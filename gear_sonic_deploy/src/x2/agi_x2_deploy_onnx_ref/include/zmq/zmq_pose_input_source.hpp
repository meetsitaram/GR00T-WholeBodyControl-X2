/**
 * @file zmq_pose_input_source.hpp
 * @brief ReferenceMotion drop-in that pulls 31-DOF body refs from a ZMQ feed.
 *
 * This is the M2 counterpart to ``StandStillReference`` and ``PklMotionReference``:
 * the X2 deploy harness consumes per-frame ``ReferenceFrame``s through the
 * abstract ``ReferenceMotion`` interface, and this class implements that
 * interface by subscribing to a ZMQ ``pose`` topic published by the VLA
 * Python process (or, during M2 bring-up, the
 * ``gear_sonic/scripts/mock_vla_publish_stand_token.py`` helper).
 *
 * ## Wire format (subset relevant to v0)
 *
 *   metadata: { v: 4, endian: "le", count: 1 }
 *   data:
 *     joint_pos_mj:   float32[31]   body reference pose, MuJoCo URDF order
 *     root_quat_xyzw: float32[4]    base link orientation, scipy convention
 *     left_hand_joints:  float32[10]  passthrough to AimDK HAL (not consumed
 *                                     by the policy ONNX -- the body-only
 *                                     decoder doesn't see fingers)
 *     right_hand_joints: float32[10]
 *     motion_token:   float32[64]   v1 hook for VLA-direct token streaming;
 *                                   currently logged but otherwise unused
 *                                   (see future-work note in
 *                                   docs/source/tutorials/vla_training.md).
 *     frame_index:    int64[1]      monotonic VLA tick counter
 *
 * ## Wire format extension (v5: future-reference window)
 *
 * The X2 policy's tokenizer obs (``tokenizer_obs.cpp``) samples a 10-frame
 * future-reference window at ``DT_FUTURE_REF = 0.1 s`` spacing on every
 * control tick: ``t_k = current_time + k * 0.1`` for k = 0..9, with k=0
 * being the current pose. ``PklMotionReference::Sample(t)`` honours the
 * time argument by indexing the loaded PKL, so the policy sees the true
 * future trajectory. The original v4 ZMQ wire carried ONE frame per
 * message and ``Sample(time)`` ignored ``time``, so the policy saw the
 * same frame replicated 10 times -- destroying the look-ahead cue and
 * causing the policy to under-commit to dynamic gait (e.g. side-steps
 * jiggling in place; see docs/source/dev_notes 2026-05-11 root-cause
 * analysis).
 *
 * v5 adds optional fields carrying the strictly-future part of that
 * window so the deploy can synthesize an in-memory mini-MotionSequence
 * matching what PklMotionReference would expose:
 *
 *   joint_pos_mj_future:    float32[NUM_FUTURE_FRAMES - 1, 31]
 *   root_quat_xyzw_future:  float32[NUM_FUTURE_FRAMES - 1, 4]
 *   joint_vel_mj_future:    float32[NUM_FUTURE_FRAMES - 1, 31]  (publisher-side
 *                                                                finite-diff)
 *   frame_index_future:     int64  [NUM_FUTURE_FRAMES - 1]
 *   future_dt_s:            float32[1]                          (== 0.1 s)
 *
 * ``window[0]`` is reconstructed from the existing single-frame
 * ``joint_pos_mj`` / ``root_quat_xyzw`` fields; ``window[1..9]`` come
 * from the new ``*_future`` arrays. When the future fields are absent
 * (legacy v4 publishers or token-only frames), ``Sample(time)`` falls
 * back to the v4 single-frame behaviour. This keeps the M2 mock VLA
 * helpers and any existing integrations working unchanged.
 *
 * Pattern parity: the G1 deploy uses ``StreamedMotionMerger`` to merge
 * multi-frame chunks into a sliding ``MotionSequence`` (see
 * ``gear_sonic_deploy/src/g1/.../streamed_motion_merger.hpp``). For X2
 * we use a smaller fixed ``NUM_FUTURE_FRAMES``-slot ring rather than
 * a full ``MotionSequence`` because the X2 policy's tokenizer only
 * looks 1 s ahead -- there's no need for arbitrary-horizon streaming
 * yet. If we wire VLA token streaming later (v1+ of motion_token)
 * we may want to graduate to ``StreamedMotionMerger``-style storage.
 *
 * Compatibility with the existing wire format
 * -------------------------------------------
 *
 * The Python helper ``pack_pose_message`` accepts a free-form payload dict;
 * this class consumes whichever fields are present and falls back to the
 * last-good cached value when an expected field is missing. That keeps M2
 * bring-up incremental:
 *
 *   - The token-only mock (``mock_vla_publish_stand_token.py``) keeps
 *     working -- when ``joint_pos_mj`` is absent, ``Sample(time)`` returns
 *     the trained ``default_angles`` stand pose.
 *   - The refframe mock (``mock_vla_publish_refframe.py``, added with this
 *     PR) drives the body to a real reference trajectory.
 *
 * ## Threading
 *
 * The subscriber runs a background thread (provided by
 * ``ZMQPackedMessageSubscriber::Start``); decoded payloads are copied into
 * ``cache_`` under ``cache_mutex_``. ``Sample(time)`` is called from the
 * 50 Hz deploy control thread and locks the same mutex for the duration
 * of a single struct copy (sub-microsecond on the X2 dev box).
 *
 * Joint velocity is reconstructed from the cached frames via finite
 * difference -- mirrors what ``PklMotionReference::Sample`` already does
 * (the file format also lacks velocity).
 */

#ifndef AGI_X2_ZMQ_POSE_INPUT_SOURCE_HPP
#define AGI_X2_ZMQ_POSE_INPUT_SOURCE_HPP

#include "policy_parameters.hpp"
#include "reference_motion.hpp"
#include "zmq/zmq_packed_message_subscriber.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace agi_x2 {

/// Hand-joint count exposed alongside the body reference. Defaults to the
/// full 10-DOF AgiBot OmniHand layout.
constexpr std::size_t DEFAULT_HAND_DOF_PER_SIDE = 10;

/// Latest hand-joint snapshot pulled from the ZMQ feed. Held as a fixed-size
/// vector so callers can pass it straight to ``aimdk_io`` without bounds
/// gymnastics. ``valid`` flips to ``true`` after the first successful decode;
/// callers that haven't seen a frame yet should fall back to "fingers open"
/// (zeros) as the deploy harness already does.
struct ZmqHandJointsSnapshot {
  std::array<double, DEFAULT_HAND_DOF_PER_SIDE> left{};
  std::array<double, DEFAULT_HAND_DOF_PER_SIDE> right{};
  bool   valid{false};
  /// Monotonic VLA tick that produced this snapshot.
  int64_t frame_index{0};
};

/**
 * @class ZmqPoseInputSource
 * @brief ReferenceMotion source that consumes body refs from a ZMQ ``pose`` topic.
 *
 * Construct via ``Connect``, then pass the returned ``unique_ptr`` to
 * ``X2Deploy`` exactly as you would a ``PklMotionReference``. Hand
 * snapshots are read out-of-band via ``LatestHandJoints()``.
 */
class ZmqPoseInputSource : public ReferenceMotion {
 public:
  /// Connect to ``host:port`` and start the background receive thread.
  /// Throws std::runtime_error on connect failure.
  static std::unique_ptr<ZmqPoseInputSource> Connect(
      const std::string& host,
      int                port,
      const std::string& topic = "pose",
      int                receive_timeout_ms = 200);

  ZmqPoseInputSource(const ZmqPoseInputSource&) = delete;
  ZmqPoseInputSource& operator=(const ZmqPoseInputSource&) = delete;

  ~ZmqPoseInputSource() override;

  // ---- ReferenceMotion interface -----------------------------------------
  ReferenceFrame Sample(double time) const override;
  std::string    Name() const override { return "zmq_pose"; }

  /// Mark the start of a control tick: the next Sample() re-snapshots the
  /// latest received window. Sample() used to detect tick boundaries ONLY
  /// from a backward jump of its ``time`` argument, which the tokenizer
  /// obs produces (10 samples per tick) -- but under whole-body TOKEN
  /// authority that obs is never built, the only callers are the wrist /
  /// head bypasses (once per tick, monotonically increasing time), so the
  /// snapshot taken at engage was served for the ENTIRE session: frozen
  /// wrist/head targets (found 2026-09-04, wb_release_stability.md 4c).
  /// Call once per control tick before any Sample(); the backward-jump
  /// heuristic stays as a fallback for callers that never call this.
  void BeginTick() const { prev_sample_t_ = -1.0; }

  /// kplanner hand-over pre-roll flag (payload field ``wb_return``): true
  /// iff the planner announced/runs its return within the last ``max_age_s``.
  bool WbReturnRequested(double now_steady_s, double max_age_s = 0.2) const {
    return latest_wb_return_.load(std::memory_order_acquire) > 0.5 &&
           (now_steady_s - wb_return_rx_.load(std::memory_order_acquire)) < max_age_s;
  }

  /// No-op: ZMQ frames are pre-anchored to whatever convention the VLA emits.
  /// Yaw alignment, if needed, lives on the publisher side (the VLA already
  /// has access to the robot's current heading via the ``x2_debug`` echo).
  void Anchor(const std::array<double, 4>& /*robot_quat_wxyz*/) override {}

  // ---- Side channel: hand joints + diagnostics ---------------------------

  /// Snapshot of the most recent hand-joint targets. Thread-safe.
  ZmqHandJointsSnapshot LatestHandJoints() const;

  /// Total decoded messages since Connect(). Useful for "no VLA frames yet"
  /// safety holds. Thread-safe.
  int64_t total_frames_received() const noexcept {
    return total_frames_received_.load(std::memory_order_acquire);
  }

  /// Monotonic seconds (std::chrono::steady_clock relative to Connect()'s
  /// "no frames yet" epoch) when the most recent ZMQ frame was decoded.
  /// Returns -1.0 if no frame has ever arrived. Thread-safe: takes
  /// ``cache_mutex_`` for the duration of a single 64-bit read.
  ///
  /// Designed for the deploy-side pose-ref starvation watchdog
  /// (PoseRefStarvationWatchdog::Update). The returned value is in the same
  /// monotonic frame as ``std::chrono::steady_clock``'s
  /// ``time_since_epoch()`` -- pair with ``SteadyNow()`` in the deploy to
  /// compute age = now - LastReceivedMonotonicS().
  double LastReceivedMonotonicS() const;

  /// Latest ``motion_token`` (64-D). Zeros until a token-bearing frame has
  /// arrived; a message WITHOUT the field keeps the previous value (same
  /// last-good-value semantics as the pose fields). Freshness is NOT
  /// tracked separately — the token rides the same pose message, so
  /// ``LastReceivedMonotonicS()`` / the pose-ref starvation watchdog govern
  /// it too. Consumed by the deploy's --whole-body-teleop mode
  /// (2026-08-29); previously this field was parsed and logged only.
  std::array<double, 64> LatestMotionToken() const {
    std::lock_guard<std::mutex> lk(cache_mutex_);
    return latest_motion_token_;
  }

  /// Latest ``engaged`` flag from the whole-body overlay sender (0.0 until a
  /// frame carries the field; last-good-value like every other field).
  /// Freshness rides LastReceivedMonotonicS() as usual.
  double LatestEngaged() const {
    std::lock_guard<std::mutex> lk(cache_mutex_);
    return latest_engaged_;
  }

  /// Whether at least one ``joint_pos_mj``-bearing frame has arrived. Until
  /// this returns true ``Sample()`` falls back to ``default_angles``.
  bool has_body_reference() const noexcept {
    return has_body_reference_.load(std::memory_order_acquire);
  }

  /// True once a v5 future-window-bearing message has been decoded; used by
  /// the deploy to log "future-aware" vs "single-frame" Sample mode.
  bool has_future_window() const noexcept {
    return has_future_window_.load(std::memory_order_acquire);
  }

 private:
  ZmqPoseInputSource(const std::string& host, int port, const std::string& topic);

  void HandleDecoded(
      const std::string& topic,
      const ZMQPackedMessageSubscriber::DecodedHeader& header,
      const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers);

  // Helpers for typed reads out of the BufferView descriptors.
  static bool CopyFloat32IntoDouble(
      const ZMQPackedMessageSubscriber::FieldInfo& field,
      const ZMQPackedMessageSubscriber::BufferView& buffer,
      double* out, std::size_t expected_count);

  static bool CopyInt64Scalar(
      const ZMQPackedMessageSubscriber::FieldInfo& field,
      const ZMQPackedMessageSubscriber::BufferView& buffer,
      int64_t* out);

  std::unique_ptr<ZMQPackedMessageSubscriber> subscriber_;

  mutable std::mutex          cache_mutex_;
  ReferenceFrame              latest_frame_{};
  ReferenceFrame              previous_frame_{};
  std::chrono::steady_clock::time_point latest_recv_{std::chrono::steady_clock::time_point::min()};
  std::chrono::steady_clock::time_point previous_recv_{std::chrono::steady_clock::time_point::min()};
  ZmqHandJointsSnapshot       latest_hand_{};
  std::array<double, 64>      latest_motion_token_{};
  double                      latest_engaged_ = 0.0;
  std::atomic<double>         latest_wb_return_{0.0};
  std::atomic<double>         wb_return_rx_{-1e9};

  // ---- v5 future-reference window --------------------------------------
  //
  // ``latest_window_[0]`` mirrors ``latest_frame_`` (k=0 == current pose
  // synthesized from the legacy single-frame fields).
  // ``latest_window_[1..NUM_FUTURE_FRAMES-1]`` come from the v5
  // ``*_future`` arrays. Fully populated only when the most recent
  // message carried the future fields; otherwise the deploy falls back
  // to ``latest_frame_`` and the legacy single-frame Sample() path.
  std::array<ReferenceFrame, NUM_FUTURE_FRAMES> latest_window_{};
  double                latest_window_dt_{DT_FUTURE_REF};
  std::atomic<bool>     has_future_window_{false};

  // ---- per-tick snapshot consumed by Sample() (control thread only) ----
  //
  // Sample() is called 10 times per policy tick at monotonically
  // increasing ``time`` arguments (current_time + k*0.1 for k=0..9).
  // We detect tick boundaries (a sudden ``time`` decrease relative to
  // the previous call) and snapshot ``latest_window_`` at the start of
  // each tick so the 10 lookups within a tick are race-free without
  // taking the cache_mutex_ on every call. ``mutable`` is sound here
  // because Sample() is logically pure WRT the wire/cache state -- the
  // snapshot is just a local optimisation private to the control thread.
  mutable std::array<ReferenceFrame, NUM_FUTURE_FRAMES> tick_window_{};
  mutable double tick_window_dt_{DT_FUTURE_REF};
  mutable bool   tick_has_window_{false};
  mutable double tick_anchor_t_{-1.0};
  mutable double prev_sample_t_{-1.0};

  std::atomic<int64_t>        total_frames_received_{0};
  std::atomic<bool>           has_body_reference_{false};

 public:
  /// Operator e-stop flag carried on the pose wire (2026-08-04): the
  /// planner latches an ``estop`` field into every payload after an
  /// operator e-stop gesture; the control loop polls this and slams
  /// stage-2 pure damping. Latched until the planner sends an explicit
  /// ``estop = 0`` (operator repeated the full gesture >= 5 s later); an
  /// absent field never clears it.
  bool EstopRequested() const {
    return estop_requested_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<bool>           estop_requested_{false};

 public:
};

}  // namespace agi_x2

#endif  // AGI_X2_ZMQ_POSE_INPUT_SOURCE_HPP
