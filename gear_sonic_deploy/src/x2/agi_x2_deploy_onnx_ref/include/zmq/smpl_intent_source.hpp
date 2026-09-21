/**
 * @file smpl_intent_source.hpp
 * @brief ZMQ subscriber for the laptop's ``pico_intent`` stream, consumed
 * directly by the deploy (2026-09-04): replaces the PC2 Python token
 * service. Fields (pack_pose_message v4, pico_intent_sender.py):
 *   smpl_joints f32[72] | human_quat f32[4] (wxyz) | engaged f32[1] |
 *   wrist_pr f32[4] (l_pitch l_roll r_pitch r_roll) | frame_index i64[1]
 *
 * Keeps a 128-frame ring stamped with the receive time and reproduces the
 * token service's guards: freshness (< stale_s), the FROZEN-INTENT guard
 * (last 25 frames byte-identical = dead body tracking), and the
 * DELAY_S-lagged 10-frame window sampled at SMPL_DT.
 */
#ifndef AGI_X2_SMPL_INTENT_SOURCE_HPP
#define AGI_X2_SMPL_INTENT_SOURCE_HPP

#include "smpl_obs.hpp"
#include "zmq/zmq_packed_message_subscriber.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

namespace agi_x2 {

class SmplIntentSource {
 public:
  /// Bind (the laptop sender PUB-connects to this port, as it did to the
  /// token service) or connect. Throws on socket failure.
  static std::unique_ptr<SmplIntentSource> Open(const std::string& host, int port,
                                                bool bind, const std::string& topic = "pico_intent");
  ~SmplIntentSource();

  /// Seconds since the last frame (inf before the first).
  double AgeS(double now_mono) const;
  /// True iff a frame arrived within stale_s AND the stream is not frozen.
  bool Fresh(double now_mono, double stale_s) const;
  /// FROZEN-INTENT guard: false when the last 25 frames are identical.
  bool Alive() const;
  /// Latest engaged flag from the sender (0/1), 0 before the first frame.
  double LatestEngaged() const;
  /// Latest wrist targets [lp lr rp rr] (rad).
  std::array<double, 4> LatestWristPr() const;
  /// Latest wrist yaw (pronation) targets [l r] (rad); {0,0} until the
  /// sender publishes ``wrist_yaw`` (2026-09-04).
  std::array<double, 2> LatestWristYaw() const;
  bool HasWristYaw() const;
  std::int64_t FramesReceived() const { return frames_.load(); }

  /// DELAY_S-lagged window (oldest first). False if fewer than 4 frames.
  bool Window(double now_mono, std::array<SmplFrame, SMPL_WINDOW>& out) const;

 private:
  SmplIntentSource() = default;
  void HandleDecoded(const std::string& topic,
                     const ZMQPackedMessageSubscriber::DecodedHeader& header,
                     const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers);

  std::unique_ptr<ZMQPackedMessageSubscriber> sub_;
  mutable std::mutex mu_;
  std::deque<SmplFrame> ring_;              // <= 128
  double last_rx_ = -1.0;
  double engaged_ = 0.0;
  std::array<double, 4> wrist_pr_{0, 0, 0, 0};
  std::array<double, 2> wrist_yaw_{0, 0};
  bool has_wrist_yaw_ = false;
  std::atomic<std::int64_t> frames_{0};
};

}  // namespace agi_x2

#endif  // AGI_X2_SMPL_INTENT_SOURCE_HPP
