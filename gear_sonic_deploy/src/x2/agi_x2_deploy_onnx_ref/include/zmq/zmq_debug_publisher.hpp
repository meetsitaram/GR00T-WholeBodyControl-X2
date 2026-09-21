/**
 * @file zmq_debug_publisher.hpp
 * @brief Packed-binary ZMQ PUB sink for the X2 deploy's ``x2_debug`` telemetry.
 *
 * Mirrors ``ZMQPackedMessageSubscriber``'s wire format on the publish side:
 *
 *   [topic_bytes][1280-byte JSON header][concatenated binary fields]
 *
 * Why packed-binary instead of msgpack
 * -------------------------------------
 *
 * The G1 deploy publishes its ``g1_debug`` topic as msgpack and consumes
 * ``pose`` as packed-binary. That asymmetry exists for historical reasons
 * (the SMPL pose pipeline pre-dated the policy state logger) and required
 * pulling in two third-party deps in the C++ tree.
 *
 * For X2 we settled on a single wire format end-to-end:
 *
 *   * incoming ``pose`` topic: packed-binary (this header's mirror image)
 *   * outgoing ``x2_debug`` topic: packed-binary (this class)
 *
 * Pros: ``nlohmann_json`` is the only third-party dep beyond ``cppzmq``;
 * the Python tooling already speaks both directions
 * (:func:`gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder.unpack_message`
 * and :func:`gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message`);
 * and the wire-format gate ``tests/test_zmq_pose_loopback.py`` already
 * covers the dtype/shape contract.
 *
 * Cons: msgpack offers heterogeneous maps (string-keyed values can have
 * different dtypes per key); the packed-binary format is array-of-fields
 * only. We adopt msgpack-like *per-field* shape descriptors at the cost
 * of forbidding nested structures -- fine for telemetry.
 *
 * Threading
 * ---------
 *
 * ``Publish*`` calls run on the deploy harness's 50 Hz control thread.
 * The class wraps a single ``zmq::socket_t`` whose access is serialised
 * via an internal mutex so it's safe to call ``Publish`` from the control
 * thread and ``PublishConfig`` from a setup thread without explicit
 * external locking.
 */

#ifndef AGI_X2_ZMQ_DEBUG_PUBLISHER_HPP
#define AGI_X2_ZMQ_DEBUG_PUBLISHER_HPP

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

namespace agi_x2 {

/// Field descriptor identical in semantics to
/// ``ZMQPackedMessageSubscriber::FieldInfo`` -- repeated here so callers
/// of the publisher don't need to include the subscriber header just to
/// reference dtype names.
struct PackedField {
  std::string         name;
  std::string         dtype;          ///< "f32", "f64", "i32", "i64", "u8", "bool".
  std::vector<size_t> shape;
  /// Pointer to source bytes; the publisher copies them into the ZMQ frame.
  const void*         data;
  /// Override byte size; if 0, computed from dtype + shape.
  size_t              byte_size;
};

/**
 * @class ZmqDebugPublisher
 * @brief Bind a PUB socket and emit packed-binary telemetry frames.
 *
 * Usage::
 *
 *     auto pub = ZmqDebugPublisher::Bind(/&zwj;*port=*&zwj;/5557, /&zwj;*topic=*&zwj;/"x2_debug");
 *     pub->PublishDebugFrame(state, last_action, motion_token);
 *     pub->PublishConfig(robot_config_map);          // throttled to ~2 s
 */
class ZmqDebugPublisher {
 public:
  static constexpr std::size_t HEADER_SIZE = 1280;

  /// Bind PUB socket on ``tcp://*:port``. Throws on bind failure.
  static std::unique_ptr<ZmqDebugPublisher> Bind(
      int port,
      const std::string& topic = "x2_debug",
      const std::string& robot_config_topic = "robot_config")
  {
    auto pub = std::unique_ptr<ZmqDebugPublisher>(
        new ZmqDebugPublisher(topic, robot_config_topic));
    pub->context_ = std::make_unique<zmq::context_t>(1);
    pub->socket_  = std::make_unique<zmq::socket_t>(*pub->context_, zmq::socket_type::pub);
    pub->socket_->set(zmq::sockopt::sndhwm, 10);
    pub->socket_->set(zmq::sockopt::linger, 0);
    pub->socket_->bind("tcp://*:" + std::to_string(port));
    std::cout << "[ZmqDebugPublisher] bound on tcp://*:" << port
              << " topic='" << topic << "'" << std::endl;
    return pub;
  }

  ZmqDebugPublisher(const ZmqDebugPublisher&) = delete;
  ZmqDebugPublisher& operator=(const ZmqDebugPublisher&) = delete;

  ~ZmqDebugPublisher()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (socket_) {
      try { socket_->close(); } catch (...) {}
      socket_.reset();
    }
    if (context_) {
      try { context_->close(); } catch (...) {}
      context_.reset();
    }
  }

  /// Publish a packed-binary message with the configured ``topic_`` prefix.
  /// Returns true on success (zmq::send_flags::dontwait may drop in HWM).
  bool Publish(const std::vector<PackedField>& fields, int version = 1, int count = 1)
  {
    return PublishOnTopic(topic_, fields, version, count);
  }

  /// Publish a config map under the ``robot_config_topic_``. Map values
  /// must be convertible to string/int/double/bool. Throttled to once every
  /// ``CONFIG_REPUBLISH_INTERVAL_SEC`` seconds to match G1's transient_local
  /// QoS semantics.
  using ConfigValue = std::variant<std::string, int, double, bool>;
  bool PublishConfig(const std::map<std::string, ConfigValue>& config)
  {
    auto now = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (config_last_publish_time_.time_since_epoch().count() != 0) {
        auto elapsed = std::chrono::duration<double>(now - config_last_publish_time_).count();
        if (elapsed < CONFIG_REPUBLISH_INTERVAL_SEC) return false;
      }
      config_last_publish_time_ = now;
    }
    // Serialise as a single packed-binary message with one string field per
    // entry. We do NOT use msgpack here; the JSON header itself carries the
    // values. The data section is empty for string-only configs.
    nlohmann::json header;
    header["v"] = 1;
    header["endian"] = "le";
    header["count"] = 1;
    header["fields"] = nlohmann::json::array();
    nlohmann::json values_obj = nlohmann::json::object();
    for (const auto& [k, v] : config) {
      std::visit([&](auto&& vv) { values_obj[k] = vv; }, v);
    }
    header["values"] = values_obj;
    std::string header_json = header.dump();
    if (header_json.size() > HEADER_SIZE) {
      std::cerr << "[ZmqDebugPublisher] config header too large: "
                << header_json.size() << " > " << HEADER_SIZE << std::endl;
      return false;
    }
    std::vector<unsigned char> frame;
    frame.reserve(robot_config_topic_.size() + HEADER_SIZE);
    frame.insert(frame.end(), robot_config_topic_.begin(), robot_config_topic_.end());
    auto header_offset = frame.size();
    frame.insert(frame.end(), HEADER_SIZE, 0);
    std::memcpy(frame.data() + header_offset, header_json.data(), header_json.size());
    return SendFrame(frame);
  }

 private:
  ZmqDebugPublisher(const std::string& topic, const std::string& robot_config_topic)
    : topic_(topic), robot_config_topic_(robot_config_topic) {}

  bool PublishOnTopic(
      const std::string& topic,
      const std::vector<PackedField>& fields,
      int version, int count)
  {
    nlohmann::json header;
    header["v"] = version;
    header["endian"] = "le";
    header["count"] = count;
    header["fields"] = nlohmann::json::array();

    std::size_t total_data_bytes = 0;
    for (const auto& f : fields) {
      nlohmann::json entry;
      entry["name"] = f.name;
      entry["dtype"] = f.dtype;
      entry["shape"] = f.shape;
      header["fields"].push_back(std::move(entry));
      total_data_bytes += FieldByteSize(f);
    }

    std::string header_json = header.dump();
    if (header_json.size() > HEADER_SIZE) {
      std::cerr << "[ZmqDebugPublisher] header too large: "
                << header_json.size() << " > " << HEADER_SIZE << std::endl;
      return false;
    }

    std::vector<unsigned char> frame;
    frame.reserve(topic.size() + HEADER_SIZE + total_data_bytes);
    frame.insert(frame.end(), topic.begin(), topic.end());
    auto header_offset = frame.size();
    frame.insert(frame.end(), HEADER_SIZE, 0);
    std::memcpy(frame.data() + header_offset, header_json.data(), header_json.size());

    for (const auto& f : fields) {
      const std::size_t bytes = FieldByteSize(f);
      auto data_offset = frame.size();
      frame.resize(frame.size() + bytes);
      if (bytes > 0 && f.data != nullptr) {
        std::memcpy(frame.data() + data_offset, f.data, bytes);
      }
    }

    return SendFrame(frame);
  }

  bool SendFrame(const std::vector<unsigned char>& frame)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!socket_) return false;
    try {
      zmq::message_t msg(frame.size());
      std::memcpy(msg.data(), frame.data(), frame.size());
      auto result = socket_->send(msg, zmq::send_flags::dontwait);
      return result.has_value();
    } catch (const zmq::error_t& e) {
      std::cerr << "[ZmqDebugPublisher] send error: " << e.what() << std::endl;
      return false;
    }
  }

  static std::size_t DtypeBytes(const std::string& dtype)
  {
    if (dtype == "f64" || dtype == "i64") return 8;
    if (dtype == "f32" || dtype == "i32") return 4;
    if (dtype == "i16" || dtype == "f16") return 2;
    if (dtype == "u8" || dtype == "i8" || dtype == "bool") return 1;
    return 4;
  }

  static std::size_t FieldByteSize(const PackedField& f)
  {
    if (f.byte_size > 0) return f.byte_size;
    std::size_t total = 1;
    for (auto d : f.shape) total *= d;
    return total * DtypeBytes(f.dtype);
  }

  static constexpr double CONFIG_REPUBLISH_INTERVAL_SEC = 2.0;

  std::string topic_;
  std::string robot_config_topic_;

  std::unique_ptr<zmq::context_t> context_;
  std::unique_ptr<zmq::socket_t>  socket_;

  std::mutex mutex_;
  std::chrono::steady_clock::time_point config_last_publish_time_{};
};

}  // namespace agi_x2

#endif  // AGI_X2_ZMQ_DEBUG_PUBLISHER_HPP
