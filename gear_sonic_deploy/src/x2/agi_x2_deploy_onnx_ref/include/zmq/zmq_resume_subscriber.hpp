/**
 * @file zmq_resume_subscriber.hpp
 * @brief Tiny ZMQ SUB helper for the operator's "resume from SAFE_IDLE" chord.
 *
 * The split-topology deploy on PC2 enters SAFE_IDLE when the pose-ref stream
 * from the laptop goes silent (PoseRefStarvationWatchdog::Tripped() == true).
 * Exiting SAFE_IDLE requires an explicit operator gesture so a wifi flicker
 * mid-task cannot silently re-engage CONTROL.
 *
 * The operator-side ``quest3_manager_x2.py`` recognises a button chord on
 * the Quest 3 controllers (A+B held for 1 s on the left controller) and
 * PUBLISHES a single multipart ZMQ message on ``tcp://*:5566`` topic
 * ``pose_resume``. This subscriber receives it and updates a steady_clock
 * timestamp atomic. The deploy's SAFE_IDLE exit logic combines this with
 * PoseRefStarvationWatchdog::ReadyToResume() to decide when to leave
 * SAFE_IDLE.
 *
 * ## Wire format
 *
 * Two-part ZMQ message:
 *   part 0: topic prefix (e.g. ``pose_resume``)
 *   part 1: 8-byte little-endian int64 = publisher-side ``time.monotonic_ns``
 *
 * We DON'T use the packed-binary header + nlohmann_json infrastructure
 * (``zmq_packed_message_subscriber.hpp``) because the resume signal is
 * dirt-simple -- one number, no per-tick rate sensitivity, no fielded
 * payload to decode. Less code = less to test = fewer ways for a safety
 * gate to break.
 *
 * ## Threading
 *
 * Background thread is spawned on Start(); recv() with a 200 ms timeout so
 * Stop() can join cleanly. Callers only ever read ``last_recv_monotonic_s_``
 * which is updated atomically -- no locking required on the read side.
 *
 * ## Lifecycle parity with ZmqPoseInputSource
 *
 * Mirrors ZmqPoseInputSource::Connect's static factory + RAII shutdown so
 * the deploy can wire it the same way (member std::unique_ptr<>; tear down
 * on node exit).
 */

#ifndef AGI_X2_ZMQ_RESUME_SUBSCRIBER_HPP
#define AGI_X2_ZMQ_RESUME_SUBSCRIBER_HPP

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <zmq.hpp>

namespace agi_x2 {

class ZmqResumeSubscriber {
 public:
  /// Connect to ``host:port`` and start the background receive thread.
  /// Throws std::runtime_error on socket creation / connect failure.
  static std::unique_ptr<ZmqResumeSubscriber> Connect(
      const std::string& host,
      int                port,
      const std::string& topic = "pose_resume",
      int                receive_timeout_ms = 200)
  {
    auto sub = std::unique_ptr<ZmqResumeSubscriber>(
        new ZmqResumeSubscriber(topic));
    sub->ctx_ = std::make_unique<zmq::context_t>(1);
    sub->sock_ = std::make_unique<zmq::socket_t>(*sub->ctx_, zmq::socket_type::sub);
    sub->sock_->set(zmq::sockopt::rcvtimeo, receive_timeout_ms);
    sub->sock_->set(zmq::sockopt::subscribe, topic);
    const std::string endpoint = "tcp://" + host + ":" + std::to_string(port);
    try {
      sub->sock_->connect(endpoint);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          "ZmqResumeSubscriber: connect failed on " + endpoint + ": " + e.what());
    }
    sub->endpoint_ = endpoint;
    sub->thread_ = std::thread(&ZmqResumeSubscriber::Run, sub.get());
    return sub;
  }

  ZmqResumeSubscriber(const ZmqResumeSubscriber&) = delete;
  ZmqResumeSubscriber& operator=(const ZmqResumeSubscriber&) = delete;

  ~ZmqResumeSubscriber()
  {
    stop_requested_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    sock_.reset();
    ctx_.reset();
  }

  /// Steady-clock seconds (matches ``steady_clock::now().time_since_epoch()``)
  /// when the most recent resume message was received. -1.0 if no message
  /// has ever arrived. Thread-safe (atomic load).
  double LastReceivedMonotonicS() const noexcept {
    return last_recv_monotonic_s_.load(std::memory_order_acquire);
  }

  /// Total resume messages received since Connect(). Useful for the
  /// periodic deploy status line so the operator can see chord presses
  /// landing.
  std::uint64_t total_received() const noexcept {
    return total_received_.load(std::memory_order_acquire);
  }

  /// True if the most recent resume message arrived within ``window_s``
  /// seconds of ``now_s`` (both in steady_clock seconds). Returns false
  /// if no message has ever arrived. The deploy's CONTROL-re-entry logic
  /// pairs this with PoseRefStarvationWatchdog::ReadyToResume() to require
  /// BOTH a fresh wire AND a fresh chord before exiting SAFE_IDLE.
  bool LatestFresh(double now_s, double window_s) const noexcept {
    const double last = last_recv_monotonic_s_.load(std::memory_order_acquire);
    if (last <= 0.0) return false;
    return (now_s - last) <= window_s;
  }

  const std::string& Endpoint() const noexcept { return endpoint_; }
  const std::string& Topic()    const noexcept { return topic_; }

 private:
  explicit ZmqResumeSubscriber(const std::string& topic) : topic_(topic) {}

  void Run()
  {
    zmq::message_t topic_msg, payload_msg;
    while (!stop_requested_.load(std::memory_order_acquire)) {
      // Part 0: topic. recv() returns std::nullopt on timeout (we set
      // rcvtimeo=200 ms so Stop() joins cleanly).
      auto rc = sock_->recv(topic_msg, zmq::recv_flags::none);
      if (!rc.has_value()) continue;
      // If publisher sent a single-part message, we'll keep it as the
      // "topic" buffer and skip the timestamp parse. If multipart, drain
      // remaining parts so the next iteration starts fresh.
      bool more = topic_msg.more();
      // Part 1+: payload(s). We only consume the first if it exists.
      bool got_payload = false;
      while (more) {
        rc = sock_->recv(payload_msg, zmq::recv_flags::none);
        if (!rc.has_value()) break;
        if (!got_payload) got_payload = true;
        more = payload_msg.more();
      }
      // Timestamp: we don't actually use the publisher-side value (clocks
      // aren't synchronised across machines). What matters is "we just
      // received a chord press" -- so we stamp on the deploy side using
      // steady_clock, which is the same clock the watchdog uses for its
      // freshness window comparisons.
      const auto now = std::chrono::steady_clock::now();
      const double now_s = std::chrono::duration<double>(
                               now.time_since_epoch()).count();
      last_recv_monotonic_s_.store(now_s, std::memory_order_release);
      total_received_.fetch_add(1, std::memory_order_acq_rel);
      (void)got_payload;  // payload presently informational only
    }
  }

  std::string                       topic_;
  std::string                       endpoint_;
  std::unique_ptr<zmq::context_t>   ctx_;
  std::unique_ptr<zmq::socket_t>    sock_;
  std::thread                       thread_;
  std::atomic<bool>                 stop_requested_{false};
  std::atomic<double>               last_recv_monotonic_s_{-1.0};
  std::atomic<std::uint64_t>        total_received_{0};
};

}  // namespace agi_x2

#endif  // AGI_X2_ZMQ_RESUME_SUBSCRIBER_HPP
