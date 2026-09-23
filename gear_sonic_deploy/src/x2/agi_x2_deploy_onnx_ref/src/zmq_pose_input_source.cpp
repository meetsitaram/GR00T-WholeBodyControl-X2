#include "zmq/zmq_pose_input_source.hpp"
// Pull the publisher header into the same TU so its compile errors surface
// in the offline syntax-check build (the ZmqDebugPublisher itself lives
// header-only; without this include it would only be checked against
// x2_deploy_onnx_ref.cpp during the full ROS 2 ament build).
#include "zmq/zmq_debug_publisher.hpp"  // NOLINT(misc-include-cleaner)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace agi_x2 {

namespace {

/// Per-field byte-swap helper. Walks ``count`` little-endian floats sitting
/// at ``src`` and writes their host-byte-order values into ``dst`` (which is
/// allowed to alias ``src`` -- swap is in place when they match).
void DecodeF32Array(const void* src, std::size_t count, double* dst, bool needs_swap)
{
  const auto* in = static_cast<const float*>(src);
  for (std::size_t i = 0; i < count; ++i) {
    float v = in[i];
    if (needs_swap) {
      v = byte_swap(v);
    }
    dst[i] = static_cast<double>(v);
  }
}

void DecodeF32ArrayInto(const void* src, std::size_t count, double* dst, bool needs_swap)
{
  DecodeF32Array(src, count, dst, needs_swap);
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

std::unique_ptr<ZmqPoseInputSource> ZmqPoseInputSource::Connect(
    const std::string& host,
    int                port,
    const std::string& topic,
    int                receive_timeout_ms)
{
  auto source = std::unique_ptr<ZmqPoseInputSource>(
      new ZmqPoseInputSource(host, port, topic));
  source->subscriber_ = std::make_unique<ZMQPackedMessageSubscriber>(
      host, port, topic, receive_timeout_ms,
      /*verbose=*/false, /*conflate=*/true,
      /*rcv_hwm=*/1);
  source->subscriber_->SetOnDecodedMessage(
      [self = source.get()](
          const std::string& t,
          const ZMQPackedMessageSubscriber::DecodedHeader& h,
          const std::vector<ZMQPackedMessageSubscriber::BufferView>& b) {
        self->HandleDecoded(t, h, b);
      });

  if (!source->subscriber_->Connect()) {
    throw std::runtime_error(
        "ZmqPoseInputSource: failed to connect to tcp://" + host + ":" +
        std::to_string(port));
  }
  source->subscriber_->Start();

  // Pre-fill the cache with the trained stand pose so that the very first
  // call to Sample() (before any VLA frame arrives) still returns a
  // physically reasonable reference. The subscriber's first decode flips
  // has_body_reference_ to true and overwrites this.
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    source->latest_frame_.joint_pos_mj[i] = default_angles[i];
    source->latest_frame_.joint_vel_mj[i] = 0.0;
    source->previous_frame_.joint_pos_mj[i] = default_angles[i];
    source->previous_frame_.joint_vel_mj[i] = 0.0;
  }
  source->latest_frame_.root_quat_xyzw = {0.0, 0.0, 0.0, 1.0};
  source->previous_frame_.root_quat_xyzw = {0.0, 0.0, 0.0, 1.0};
  // Pre-fill the future-window slots with the same stand pose / identity
  // quat so any Sample() that happens after has_future_window_ flips
  // true but before a subsequent decode lands sees a sane reference.
  for (auto& slot : source->latest_window_) {
    slot = source->latest_frame_;
  }
  for (auto& slot : source->tick_window_) {
    slot = source->latest_frame_;
  }
  return source;
}

ZmqPoseInputSource::ZmqPoseInputSource(
    const std::string& /*host*/, int /*port*/, const std::string& /*topic*/)
{
  // All real wiring happens in Connect(). The ctor exists purely so
  // std::unique_ptr<ZmqPoseInputSource> can be the return type of Connect().
}

ZmqPoseInputSource::~ZmqPoseInputSource()
{
  if (subscriber_) {
    subscriber_->Stop();
    subscriber_.reset();
  }
}

// ---------------------------------------------------------------------------
// LastReceivedMonotonicS -- pose-ref starvation watchdog hook
// ---------------------------------------------------------------------------
double ZmqPoseInputSource::LastReceivedMonotonicS() const
{
  std::lock_guard<std::mutex> lock(cache_mutex_);
  // No frame has ever arrived (sentinel = time_point::min()).
  if (latest_recv_.time_since_epoch().count() ==
      std::chrono::steady_clock::time_point::min().time_since_epoch().count()) {
    return -1.0;
  }
  return std::chrono::duration<double>(latest_recv_.time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

bool ZmqPoseInputSource::CopyFloat32IntoDouble(
    const ZMQPackedMessageSubscriber::FieldInfo& field,
    const ZMQPackedMessageSubscriber::BufferView& buffer,
    double* out, std::size_t expected_count)
{
  if (field.dtype != "f32") {
    std::cerr << "[ZmqPoseInputSource] field '" << field.name
              << "' has unexpected dtype '" << field.dtype
              << "', want f32" << std::endl;
    return false;
  }
  std::size_t total = 1;
  for (auto d : field.shape) total *= d;
  if (total != expected_count) {
    std::cerr << "[ZmqPoseInputSource] field '" << field.name
              << "' has " << total << " elements, want " << expected_count
              << std::endl;
    return false;
  }
  if (buffer.size != total * sizeof(float)) {
    std::cerr << "[ZmqPoseInputSource] field '" << field.name
              << "' byte size mismatch: " << buffer.size
              << " vs " << total * sizeof(float) << std::endl;
    return false;
  }
  // We assume the publisher's byte order matches ours; ZMQPackedMessageSubscriber
  // exposes a byte-swap hint via DecodedHeader::NeedsByteSwap, but we don't
  // have it here. Pass needs_swap=false unconditionally; this matches every
  // x86_64/aarch64 host we ship to (all little-endian).
  DecodeF32Array(buffer.data, expected_count, out, /*needs_swap=*/false);
  return true;
}

bool ZmqPoseInputSource::CopyInt64Scalar(
    const ZMQPackedMessageSubscriber::FieldInfo& field,
    const ZMQPackedMessageSubscriber::BufferView& buffer,
    int64_t* out)
{
  if (field.dtype != "i64") {
    std::cerr << "[ZmqPoseInputSource] field '" << field.name
              << "' has unexpected dtype '" << field.dtype
              << "', want i64" << std::endl;
    return false;
  }
  if (buffer.size < sizeof(int64_t)) return false;
  std::memcpy(out, buffer.data, sizeof(int64_t));
  return true;
}

void ZmqPoseInputSource::HandleDecoded(
    const std::string& /*topic*/,
    const ZMQPackedMessageSubscriber::DecodedHeader& header,
    const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers)
{
  if (header.fields.size() != buffers.size()) return;

  ReferenceFrame next_frame{};
  bool got_body = false;
  bool got_quat = false;
  ZmqHandJointsSnapshot next_hand{};
  std::array<double, 64> next_motion_token{};
  bool got_motion_token = false;
  double next_engaged = 0.0;
  bool got_engaged = false;
  int64_t next_frame_index = -1;

  // ---- v5 future-window staging buffers --------------------------------
  // Decoded into local arrays first; promoted into latest_window_ at the
  // end iff the message carried a complete (joint_pos + root_quat) pair
  // for all kFutureSlots strictly-future slots. window[0] is synthesized
  // from the legacy single-frame fields below.
  constexpr std::size_t kFutureSlots = NUM_FUTURE_FRAMES - 1;  // window[1..9]
  std::array<std::array<double, NUM_DOFS>, kFutureSlots> future_jpos{};
  std::array<std::array<double, NUM_DOFS>, kFutureSlots> future_jvel{};
  std::array<std::array<double, 4>,        kFutureSlots> future_quat{};
  bool got_future_jpos = false;
  bool got_future_jvel = false;
  bool got_future_quat = false;
  double next_future_dt_s = DT_FUTURE_REF;

  // Seed next_frame from cache so partial messages don't overwrite valid state.
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    next_frame = latest_frame_;
    next_hand = latest_hand_;
    next_motion_token = latest_motion_token_;
  }

  for (std::size_t i = 0; i < header.fields.size(); ++i) {
    const auto& f = header.fields[i];
    const auto& b = buffers[i];

    if (f.name == "estop") {
      // Operator e-stop (2026-08-04): latch; the control loop polls
      // EstopRequested() and slams stage-2 pure damping. 2026-09-23: the
      // planner lowers it with an EXPLICIT 0 after the operator repeats the
      // full gesture (>= 5 s later, fresh chord); an ABSENT field never
      // clears it, so a wire flap cannot lift an e-stop. The SAFE_HOLD ->
      // RECOVER gate is vetoed while it is set.
      double v = 1.0;
      if (!CopyFloat32IntoDouble(f, b, &v, 1)) v = 1.0;
      estop_requested_.store(v >= 0.5, std::memory_order_release);
    } else if (f.name == "joint_pos_mj") {
      if (CopyFloat32IntoDouble(f, b, next_frame.joint_pos_mj.data(), NUM_DOFS)) {
        got_body = true;
      }
    } else if (f.name == "root_quat_xyzw") {
      if (CopyFloat32IntoDouble(f, b, next_frame.root_quat_xyzw.data(), 4)) {
        got_quat = true;
      }
    } else if (f.name == "left_hand_joints") {
      double tmp[DEFAULT_HAND_DOF_PER_SIDE]{};
      if (CopyFloat32IntoDouble(f, b, tmp, DEFAULT_HAND_DOF_PER_SIDE)) {
        for (std::size_t k = 0; k < DEFAULT_HAND_DOF_PER_SIDE; ++k) {
          next_hand.left[k] = tmp[k];
        }
        next_hand.valid = true;
      }
    } else if (f.name == "right_hand_joints") {
      double tmp[DEFAULT_HAND_DOF_PER_SIDE]{};
      if (CopyFloat32IntoDouble(f, b, tmp, DEFAULT_HAND_DOF_PER_SIDE)) {
        for (std::size_t k = 0; k < DEFAULT_HAND_DOF_PER_SIDE; ++k) {
          next_hand.right[k] = tmp[k];
        }
        next_hand.valid = true;
      }
    } else if (f.name == "motion_token") {
      if (CopyFloat32IntoDouble(f, b, next_motion_token.data(), 64)) {
        got_motion_token = true;
      }
    } else if (f.name == "engaged") {
      double v{};
      if (CopyFloat32IntoDouble(f, b, &v, 1)) {
        next_engaged = v;
        got_engaged = true;
      }
    } else if (f.name == "wb_return") {
      // kplanner hand-over pre-roll (dual-head deploy, 2026-09-04): 1.0
      // while the planner announces / runs its rate-limited return.
      double v{};
      if (CopyFloat32IntoDouble(f, b, &v, 1)) {
        latest_wb_return_.store(v, std::memory_order_release);
        wb_return_rx_.store(std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count(),
            std::memory_order_release);
      }
    } else if (f.name == "frame_index") {
      int64_t idx{};
      if (CopyInt64Scalar(f, b, &idx)) next_frame_index = idx;
    } else if (f.name == "joint_pos_mj_future") {
      // Expected shape: [kFutureSlots, NUM_DOFS]. Validate by total count
      // inside CopyFloat32IntoDouble (which enforces field.shape product).
      if (CopyFloat32IntoDouble(
              f, b, future_jpos[0].data(), kFutureSlots * NUM_DOFS)) {
        got_future_jpos = true;
      }
    } else if (f.name == "root_quat_xyzw_future") {
      if (CopyFloat32IntoDouble(
              f, b, future_quat[0].data(), kFutureSlots * 4)) {
        got_future_quat = true;
      }
    } else if (f.name == "joint_vel_mj_future") {
      // Optional: if absent we synthesize joint_vel from a finite-diff of
      // future_jpos at consumer time.
      if (CopyFloat32IntoDouble(
              f, b, future_jvel[0].data(), kFutureSlots * NUM_DOFS)) {
        got_future_jvel = true;
      }
    } else if (f.name == "future_dt_s") {
      double tmp[1]{};
      if (CopyFloat32IntoDouble(f, b, tmp, 1)) {
        // Sanity-clip implausible values; default DT_FUTURE_REF on bad input.
        if (tmp[0] >= 0.01 && tmp[0] <= 1.0) {
          next_future_dt_s = tmp[0];
        }
      }
    }
    // Unknown fields are silently ignored (forward compat).
  }

  // Joint velocity comes from finite differencing the last two body refs.
  // Mirrors PklMotionReference's strategy. We rely on the publisher running
  // at a roughly constant rate (mock-VLA at 50 Hz); rate jitter shows up as
  // velocity noise, which the policy tolerates.
  auto now = std::chrono::steady_clock::now();
  if (got_body) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (has_body_reference_.load(std::memory_order_acquire)) {
      const auto dt = std::chrono::duration<double>(now - latest_recv_).count();
      if (dt > 0.0) {
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          next_frame.joint_vel_mj[i] =
              (next_frame.joint_pos_mj[i] - latest_frame_.joint_pos_mj[i]) / dt;
        }
      } else {
        next_frame.joint_vel_mj = latest_frame_.joint_vel_mj;
      }
    } else {
      next_frame.joint_vel_mj.fill(0.0);
    }
    previous_frame_ = latest_frame_;
    previous_recv_  = latest_recv_;
    latest_frame_   = next_frame;
    latest_recv_    = now;
    if (next_hand.valid) {
      next_hand.frame_index = next_frame_index;
      latest_hand_ = next_hand;
    }
    if (got_motion_token) {
      latest_motion_token_ = next_motion_token;
    }
    if (got_engaged) {
      latest_engaged_ = next_engaged;
    }
    has_body_reference_.store(true, std::memory_order_release);

    // ---- v5 future window: promote into latest_window_ if complete ----
    //
    // Only promote when both joint_pos and root_quat future arrays are
    // present (joint_vel is optional -- we synthesize it via
    // backward-finite-diff if missing). Partial v5 messages are
    // intentionally treated as v4: the deploy keeps using the
    // single-frame Sample() path until a fully-formed window arrives.
    if (got_future_jpos && got_future_quat) {
      latest_window_dt_ = next_future_dt_s;
      // window[0] = the message's "current" pose. Velocity is the
      // wall-clock finite-diff already computed in next_frame.
      latest_window_[0] = next_frame;
      const double dt_future = (next_future_dt_s > 1e-6)
                                   ? next_future_dt_s
                                   : DT_FUTURE_REF;
      for (std::size_t k = 0; k < kFutureSlots; ++k) {
        ReferenceFrame& slot = latest_window_[k + 1];
        slot.joint_pos_mj = future_jpos[k];
        slot.root_quat_xyzw = future_quat[k];
        if (got_future_jvel) {
          slot.joint_vel_mj = future_jvel[k];
        } else {
          // Backward finite-diff: vel[k] = (jpos[k] - jpos[k-1]) / dt
          // with jpos[-1] taken as the current frame's joint_pos.
          const auto& prev_jpos = (k == 0)
              ? next_frame.joint_pos_mj
              : future_jpos[k - 1];
          for (std::size_t i = 0; i < NUM_DOFS; ++i) {
            slot.joint_vel_mj[i] =
                (future_jpos[k][i] - prev_jpos[i]) / dt_future;
          }
        }
      }
      has_future_window_.store(true, std::memory_order_release);
    }
  } else {
    // Token-only / hand-only frame: refresh side-channel caches without
    // touching the body reference (Sample() keeps returning the last
    // good body frame).
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (next_hand.valid) {
      next_hand.frame_index = next_frame_index;
      latest_hand_ = next_hand;
    }
    if (got_motion_token) {
      latest_motion_token_ = next_motion_token;
    }
    if (got_engaged) {
      latest_engaged_ = next_engaged;
    }
    if (got_quat) {
      latest_frame_.root_quat_xyzw = next_frame.root_quat_xyzw;
    }
  }

  total_frames_received_.fetch_add(1, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

ReferenceFrame ZmqPoseInputSource::Sample(double time) const
{
  // Tick boundary detection: the C++ tokenizer obs calls Sample() 10
  // times per control tick at monotonically increasing ``time`` args
  // (current_time + k * DT_FUTURE_REF for k = 0..9). Between ticks,
  // ``time`` jumps BACKWARD (next tick's k=0 is +0.02s of the previous
  // tick's k=0, but -0.88s relative to the previous tick's k=9). We use
  // that backward jump to detect "this is the first Sample of a new
  // tick" and snapshot ``latest_window_`` once per tick rather than
  // taking ``cache_mutex_`` on every one of the 10 lookups.
  //
  // The half-DT_FUTURE_REF epsilon is generous enough to tolerate any
  // realistic jitter in policy_time arithmetic but tight enough that
  // a same-tick Sample re-call (e.g. wrist-bypass at line 1411) won't
  // be confused for a new tick.
  constexpr double kTickEpsilon = 0.5 * DT_FUTURE_REF;
  const bool new_tick =
      (prev_sample_t_ < 0.0) || (time + kTickEpsilon < prev_sample_t_);

  if (new_tick) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (has_future_window_.load(std::memory_order_acquire)) {
      tick_window_ = latest_window_;
      tick_window_dt_ = latest_window_dt_;
      tick_has_window_ = true;
    } else {
      tick_has_window_ = false;
    }
    tick_anchor_t_ = time;
  }
  prev_sample_t_ = time;

  if (tick_has_window_) {
    const double dt =
        (tick_window_dt_ > 1e-6) ? tick_window_dt_ : DT_FUTURE_REF;
    int k = static_cast<int>(std::lround((time - tick_anchor_t_) / dt));
    constexpr int kMaxK = static_cast<int>(NUM_FUTURE_FRAMES) - 1;
    if (k < 0)      k = 0;
    if (k > kMaxK)  k = kMaxK;
    return tick_window_[static_cast<std::size_t>(k)];
  }

  // Legacy single-frame fallback: pre-v5 publishers (or v5 publishers
  // emitting a token-only or partial frame) land here. Behaviour matches
  // the original v4 semantics: ``time`` is ignored, latest_frame_ is
  // returned.
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return latest_frame_;
}

ZmqHandJointsSnapshot ZmqPoseInputSource::LatestHandJoints() const
{
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return latest_hand_;
}

}  // namespace agi_x2
