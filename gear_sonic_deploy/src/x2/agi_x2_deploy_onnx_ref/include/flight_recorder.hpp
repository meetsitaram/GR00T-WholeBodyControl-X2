#ifndef AGI_X2_FLIGHT_RECORDER_HPP
#define AGI_X2_FLIGHT_RECORDER_HPP
/**
 * @file flight_recorder.hpp
 * @brief Pre-trigger ring buffer ("black box") dumped when an e-stop fires.
 *
 * WHY. Nothing in this stack observes above 50 Hz. x2_debug publishes one frame
 * per CONTROL tick, motor_monitor logs at 1 Hz, and the 500 Hz command writer is
 * a pure re-publish of an unchanged target. So any phenomenon faster than 25 Hz
 * -- actuator chatter, gear-lash clicking, a PD/firmware limit cycle -- is
 * ALIASED and invisible. Every high-frequency question raised during the
 * 2026-08-24 waist investigation was unanswerable for exactly this reason.
 *
 * WHAT IT CAPTURES. Two rings, both pre-allocated at construction so the
 * real-time path never allocates:
 *
 *   FAST  @ writer rate (500 Hz)  -- the STATE, sampled fast. This is the
 *         point: subscriptions are async/event-driven, so sampling state at the
 *         writer tick reveals content the 50 Hz stream folds down. Also carries
 *         the published target so command->response lag is measurable.
 *
 *   SLOW  @ control rate (50 Hz)  -- what the policy saw and asked for:
 *         raw action PRE-clip (the DEMAND, otherwise unobservable -- see the
 *         trap in the investigation notes: `last_action_il_` is captured AFTER
 *         the clip and the x2_debug field of that name is actually a target),
 *         action POST-clip, the kplanner reference, and the safety-stack output.
 *
 * ORDERING. Everything is stored in MuJoCo joint order, matching x2_debug and
 * the ritual CSVs, so no permutation is needed downstream. The ONE exception is
 * `action_pre_clip` / `action_post_clip`, which are IsaacLab-ordered because
 * that is the policy's native space; the sidecar records this per field so a
 * reader cannot get it wrong.
 *
 * THREADING. Each ring has exactly ONE producer (fast: the writer timer; slow:
 * the control timer), so a relaxed atomic write index is sufficient -- no locks
 * on the hot path. Trigger() freezes both rings and hands serialisation to a
 * detached thread, because writing ~11 MB takes tens of ms and the 500 Hz
 * writer must keep feeding damping commands during SAFE_HOLD.
 *
 * OUTPUT. `<dir>/flight_<reason>_<unix_ms>.bin` plus a `.json` sidecar giving
 * dtype, field order, widths and sample rates. Load with
 * gear_sonic/scripts/read_flight_recorder.py.
 */
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace agi_x2 {

class FlightRecorder {
 public:
  // 50 Hz, NOT 500. The fast ring originally sampled in the 500 Hz writer to
  // resolve content a 50 Hz stream aliases. That put a state snapshot + ring
  // write on the hot path, and on 2026-08-25 -- with the control loop already
  // over budget and every callback serialised in one MutuallyExclusive group
  // -- the writer collapsed to 3.4 Hz. Telemetry must not compete with the
  // control path for CPU. Recording from OnControl's EXISTING snapshot costs
  // nothing extra: no second lock acquisition, no work in the writer at all.
  // The cost is losing sub-tick resolution on the state channels.
  static constexpr int kFastHz = 50;
  static constexpr int kSlowHz = 50;

  /// @param dofs     joint count (31 on X2)
  /// @param seconds  history depth; 0 disables the recorder entirely
  FlightRecorder(std::size_t dofs, double seconds, std::string dir)
      : dofs_(dofs),
        enabled_(seconds > 0.0 && !dir.empty()),
        dir_(std::move(dir)),
        fast_n_(enabled_ ? static_cast<std::size_t>(seconds * kFastHz) : 0),
        slow_n_(enabled_ ? static_cast<std::size_t>(seconds * kSlowHz) : 0),
        fast_stride_(1 + 5 * dofs + 12),  // t | pos vel tgt eff temp | quat4 gyro3 age5
        slow_stride_(1 + 5 * dofs + 6) {  // t | pre post raw final ref | 6 scalars
    if (!enabled_) return;
    fast_.assign(fast_n_ * fast_stride_, 0.0f);
    slow_.assign(slow_n_ * slow_stride_, 0.0f);
    // Time is kept as float64 in a SEPARATE channel. float32 holds only ~7
    // significant digits, so a steady_clock value of ~1.2e5 s quantises to
    // ~10 ms -- coarser than the phenomena this recorder exists to resolve.
    fast_t_.assign(fast_n_, 0.0);
    slow_t_.assign(slow_n_, 0.0);
    t0_wall_ms_ = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    t0_mono_s_ = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  bool enabled() const { return enabled_; }

  /// Called from the 500 Hz writer. MUST NOT allocate or block.
  /// NOTE: there are NO world coordinates to record -- RobotState carries no
  /// position, no odometry and no linear acceleration. Orientation (quat) and
  /// angular velocity are the only base signals that exist. `src_age_s` is the
  /// per-source staleness (leg,waist,arm,head,imu): at 500 Hz that is a proper
  /// starvation trace, which is what actually tripped SAFE_HOLD on 2026-08-24.
  void RecordFast(double t,
                  const double* joint_pos_mj, const double* joint_vel_mj,
                  const double* target_pos_mj, const double* pd_torque_demand_mj,
                  const double* motor_temp_c,
                  const double* quat_wxyz, const double* gyro_xyz,
                  const double* src_age_s /* 5 */) {
    if (!enabled_ || frozen_.load(std::memory_order_relaxed)) return;
    const std::size_t i = fast_w_.fetch_add(1, std::memory_order_relaxed) % fast_n_;
    fast_t_[i] = t;                       // float64: full ms (and better) fidelity
    float* r = fast_.data() + i * fast_stride_;
    std::size_t k = 0;
    r[k++] = static_cast<float>(t - t0_mono_s_);   // float32 copy, seconds since start
    for (std::size_t j = 0; j < dofs_; ++j) r[k++] = static_cast<float>(joint_pos_mj[j]);
    for (std::size_t j = 0; j < dofs_; ++j) r[k++] = static_cast<float>(joint_vel_mj[j]);
    for (std::size_t j = 0; j < dofs_; ++j) r[k++] = static_cast<float>(target_pos_mj[j]);
    for (std::size_t j = 0; j < dofs_; ++j)
      r[k++] = pd_torque_demand_mj ? static_cast<float>(pd_torque_demand_mj[j]) : 0.0f;
    for (std::size_t j = 0; j < dofs_; ++j)
      r[k++] = motor_temp_c ? static_cast<float>(motor_temp_c[j]) : 0.0f;
    for (int j = 0; j < 4; ++j) r[k++] = static_cast<float>(quat_wxyz[j]);
    for (int j = 0; j < 3; ++j) r[k++] = static_cast<float>(gyro_xyz[j]);
    for (int j = 0; j < 5; ++j)
      r[k++] = src_age_s ? static_cast<float>(src_age_s[j]) : -1.0f;
  }

  /// Called from the 50 Hz control tick. MUST NOT allocate or block.
  /// Captures the WHOLE command chain, so a reader can attribute a difference
  /// to a specific stage instead of guessing:
  ///   action_pre_il    raw policy output, before --action-clip      (IL order)
  ///   action_post_il   after --action-clip                          (IL order)
  ///   target_raw_mj    default + action*action_scale, BEFORE the
  ///                    max_target_dev clamp and the output LPF      (MJ order)
  ///   target_final_mj  what is actually published on ZMQ to the
  ///                    robot, after clamp + LPF                     (MJ order)
  ///   ref_pos_mj       the kplanner reference that drove this tick  (MJ order)
  void RecordSlow(double t,
                  const double* action_pre_il, const double* action_post_il,
                  const double* target_raw_mj, const double* target_final_mj,
                  const double* ref_pos_mj,
                  float ramp_alpha, float clipped_frac, int state,
                  bool tilt_trip, bool estop, float pose_ref_age_s) {
    if (!enabled_ || frozen_.load(std::memory_order_relaxed)) return;
    const std::size_t i = slow_w_.fetch_add(1, std::memory_order_relaxed) % slow_n_;
    slow_t_[i] = t;
    float* r = slow_.data() + i * slow_stride_;
    std::size_t k = 0;
    r[k++] = static_cast<float>(t - t0_mono_s_);
    for (std::size_t j = 0; j < dofs_; ++j) r[k++] = static_cast<float>(action_pre_il[j]);
    for (std::size_t j = 0; j < dofs_; ++j) r[k++] = static_cast<float>(action_post_il[j]);
    for (std::size_t j = 0; j < dofs_; ++j)
      r[k++] = target_raw_mj ? static_cast<float>(target_raw_mj[j]) : 0.0f;
    for (std::size_t j = 0; j < dofs_; ++j)
      r[k++] = target_final_mj ? static_cast<float>(target_final_mj[j]) : 0.0f;
    for (std::size_t j = 0; j < dofs_; ++j)
      r[k++] = ref_pos_mj ? static_cast<float>(ref_pos_mj[j]) : 0.0f;
    r[k++] = ramp_alpha;
    r[k++] = clipped_frac;
    r[k++] = static_cast<float>(state);
    r[k++] = tilt_trip ? 1.0f : 0.0f;
    r[k++] = estop ? 1.0f : 0.0f;
    r[k++] = pose_ref_age_s;
  }

  /// Freeze both rings and serialise off-thread. Idempotent: only the first
  /// trigger of a run writes, so an e-stop that escalates does not clobber the
  /// evidence with a second, shorter capture.
  void Trigger(const std::string& reason) {
    if (!enabled_) return;
    bool expected = false;
    if (!triggered_.compare_exchange_strong(expected, true)) return;
    frozen_.store(true, std::memory_order_relaxed);
    std::thread([this, reason] { Dump(reason); }).detach();
  }

  /// Segment-capture variant: dump, then UNFREEZE and re-arm so every
  /// whole-body release cuts its own box (the once-latch above ate all but
  /// the first of ~10 segment dumps, capture session 2026-08-30). Recording
  /// pauses only for the dump itself (<1 s between segments). A safety
  /// Trigger() that fires first still wins and stays latched — crash
  /// evidence is never clobbered by a later segment dump.
  void TriggerReusable(const std::string& reason) {
    if (!enabled_) return;
    bool expected = false;
    if (!triggered_.compare_exchange_strong(expected, true)) return;
    frozen_.store(true, std::memory_order_relaxed);
    std::thread([this, reason] {
      Dump(reason);
      frozen_.store(false, std::memory_order_relaxed);
      triggered_.store(false);
    }).detach();
  }

 private:
  void Dump(const std::string& reason) const {
    const auto ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    const std::string stem = dir_ + "/flight_" + reason + "_" + std::to_string(ms);
    {
      std::ofstream b(stem + ".bin", std::ios::binary | std::ios::trunc);
      const std::size_t fw = fast_w_.load(), sw = slow_w_.load();
      // Unroll oldest->newest so the reader gets chronological order directly.
      EmitRingT(b, fast_t_, fast_n_, fw);
      EmitRingT(b, slow_t_, slow_n_, sw);
      EmitRing(b, fast_, fast_n_, fast_stride_, fw);
      EmitRing(b, slow_, slow_n_, slow_stride_, sw);
    }
    std::ofstream j(stem + ".json", std::ios::trunc);
    j << "{\n"
      << "  \"reason\": \"" << reason << "\",\n"
      << "  \"dtype\": \"float32\",\n"
      << "  \"layout\": \"float64 fast_t[], float64 slow_t[], then float32 fast[], float32 slow[]\",\n"
      << "  \"t0_wall_unix_ms\": " << t0_wall_ms_ << ",\n"
      << "  \"t0_mono_s\": " << std::fixed << t0_mono_s_ << ",\n"
      << "  \"t_note\": \"time channels are float64 monotonic seconds; add "
         "(t0_wall_unix_ms/1000 - t0_mono_s) to get wall-clock unix seconds\",\n"
      << "  \"dofs\": " << dofs_ << ",\n"
      << "  \"fast\": {\"rate_hz\": " << kFastHz << ", \"rows\": "
      << std::min(fast_w_.load(), fast_n_) << ", \"stride\": " << fast_stride_
      << ", \"fields\": [\"t\",\"joint_pos_mj\",\"joint_vel_mj\",\"target_pos_mj\","
         "\"pd_torque_demand_mj\",\"motor_temp_c\",\"base_quat_wxyz\","
         "\"base_gyro_xyz\",\"src_age_s\"],"
         " \"order\": \"mujoco\"},\n"
      << "  \"slow\": {\"rate_hz\": " << kSlowHz << ", \"rows\": "
      << std::min(slow_w_.load(), slow_n_) << ", \"stride\": " << slow_stride_
      << ", \"fields\": [\"t\",\"action_pre_clip_il\",\"action_post_clip_il\","
         "\"target_raw_mj\",\"target_final_mj\",\"ref_pos_mj\","
         "\"ramp_alpha\",\"clipped_frac\","
         "\"state\",\"tilt_trip\",\"estop\",\"pose_ref_age_s\"],"
         " \"order\": \"ACTIONS ARE ISAACLAB-ORDERED; the rest are mujoco\"}\n"
      << "}\n";
  }

  /// float64 time channel, oldest->newest, written BEFORE the float32 payload.
  static void EmitRingT(std::ofstream& b, const std::vector<double>& v,
                        std::size_t n, std::size_t w) {
    if (n == 0) return;
    const std::size_t filled = std::min(w, n);
    const std::size_t start = (w >= n) ? (w % n) : 0;
    for (std::size_t c = 0; c < filled; ++c) {
      const double t = v[(start + c) % n];
      b.write(reinterpret_cast<const char*>(&t), sizeof(double));
    }
  }

  static void EmitRing(std::ofstream& b, const std::vector<float>& v,
                       std::size_t n, std::size_t stride, std::size_t w) {
    if (n == 0) return;
    const std::size_t filled = std::min(w, n);
    const std::size_t start = (w >= n) ? (w % n) : 0;
    for (std::size_t c = 0; c < filled; ++c) {
      const float* r = v.data() + ((start + c) % n) * stride;
      b.write(reinterpret_cast<const char*>(r),
              static_cast<std::streamsize>(stride * sizeof(float)));
    }
  }

  std::size_t dofs_;
  bool enabled_;
  std::string dir_;
  std::size_t fast_n_, slow_n_, fast_stride_, slow_stride_;
  std::vector<float>  fast_, slow_;
  std::vector<double> fast_t_, slow_t_;
  std::uint64_t t0_wall_ms_ = 0;
  double        t0_mono_s_  = 0.0;
  std::atomic<std::size_t> fast_w_{0}, slow_w_{0};
  std::atomic<bool> frozen_{false}, triggered_{false};
};

}  // namespace agi_x2
#endif  // AGI_X2_FLIGHT_RECORDER_HPP
