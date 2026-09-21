/**
 * @file safety.hpp
 * @brief Safety scaffolding for first-real-robot bring-up.
 *
 * Three layers, ordered most-passive to least-passive:
 *
 *   1. DryRun
 *      - Runs the full obs/policy/IO pipeline and PUBLISHES the commands,
 *        but with stiffness=0 and damping=0 so the motors do nothing. Used
 *        to verify wiring (topics, joint name remap, IMU sign convention)
 *        before any torque is applied.
 *
 *   2. SoftStartRamp
 *      - Once the operator says "go", blend the live policy target with the
 *        DEFAULT_DOF standing pose over `ramp_seconds` seconds. Prevents
 *        a sudden jump from the operator-selected start pose to whatever
 *        the policy thinks is "right" on the first inference.
 *
 *   3. TiltWatchdog
 *      - Monitors gravity[2] in body frame (== quat_rotate_inverse(quat,
 *        [0,0,-1])[2]). On the real robot, "upright" means gravity[2] is
 *        approximately -1.0. If it drifts above the configured threshold
 *        (default -0.3, matching eval_x2_mujoco --fall-tilt-cos), the
 *        watchdog trips and the deploy switches to "hold default angles
 *        with high damping" mode (no policy commands). The operator can
 *        then catch the robot manually.
 *
 * All three are pure CPU-side state machines; they do not own any ROS or
 * ONNX resources. The main loop calls them from the 50 Hz control timer.
 */

#ifndef AGI_X2_SAFETY_HPP
#define AGI_X2_SAFETY_HPP

#include "policy_parameters.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <string>

namespace agi_x2 {

/// Pure-data result of running the safety stack on one tick. Consumed by
/// the IO publisher to decide what to actually send to the joints.
struct SafeCommand {
  std::array<double, NUM_DOFS> target_pos_mj;
  std::array<double, NUM_DOFS> stiffness_mj;
  std::array<double, NUM_DOFS> damping_mj;
  bool                         dry_run    = false;  ///< if true, all kp/kd zeroed
  bool                         tilt_trip  = false;  ///< watchdog fired this tick
  double                       ramp_alpha = 1.0;    ///< 0=fully default, 1=fully policy
  std::string                  reason;              ///< human-readable status
};

/// Soft-start ramp: blends a freshly-computed policy target from a
/// blend-from pose during the first `ramp_seconds` seconds after a
/// (re-)arm. The blend-from pose is the robot's measured pose at the
/// tick that armed the ramp when the caller supplies one; otherwise the
/// trained default standing pose (legacy behaviour, first CONTROL entry
/// before any state snapshot exists).
class SoftStartRamp {
 public:
  explicit SoftStartRamp(double ramp_seconds = 2.0)
      : ramp_seconds_(ramp_seconds) {}

  /// Legacy re-arm: alpha restarts at 0, blend-from = default_angles.
  void Reset() { control_start_s_ = -1.0; has_from_ = false; }

  /// Re-arm blending from the given pose (normally the measured
  /// joint_pos at the transition tick). This is the fix for the engage
  /// crouch-dip (2026-09-01): default_angles has knees at ~0.66 rad, so
  /// a default-anchored re-arm *commands a crouch* whenever authority
  /// changes while the robot stands at ~0.35 -- the weak sim twin fell
  /// inside that transient and hardware took a visible engage bump.
  void Reset(const std::array<double, NUM_DOFS>& blend_from) {
    control_start_s_ = -1.0;
    blend_from_ = blend_from;
    has_from_ = true;
  }

  /// Compute alpha in [0, 1]. alpha=0 -> output = blend-from, alpha=1 -> policy.
  double Alpha(double now_s);

  /// Blend in place: target_mj = (1 - alpha) * blend_from + alpha * target_mj.
  void Apply(double alpha, std::array<double, NUM_DOFS>& target_mj) const;

 private:
  double ramp_seconds_;
  double control_start_s_ = -1.0;  ///< set on first Alpha() call after Reset()
  std::array<double, NUM_DOFS> blend_from_{};  ///< valid iff has_from_
  bool   has_from_ = false;
};

/// Tilt watchdog: monitors body-frame gravity[2]. Once tripped, latches.
/// Use Reset() at INIT to clear.
class TiltWatchdog {
 public:
  /// @param fall_tilt_cos same sign convention as
  ///   gear_sonic/scripts/eval_x2_mujoco.py::--fall-tilt-cos. Default -0.3
  ///   means "trip when the body tilt exceeds ~72.5 degrees from upright".
  explicit TiltWatchdog(double fall_tilt_cos = -0.3)
      : fall_tilt_cos_(fall_tilt_cos) {}

  void   Reset()      { tripped_ = false; force_stage2_ = false; reason_.clear(); }
  bool   Tripped()    const { return tripped_; }
  double FallCos()    const { return fall_tilt_cos_; }
  const std::string& Reason() const { return reason_; }

  /// Returns true if the watchdog fires ON THIS TICK (transitioned from
  /// not-tripped to tripped). Subsequent calls return false but Tripped()
  /// stays true.
  bool Update(double gravity_body_z);

  /// Operator e-stop from the pose wire (2026-08-04): terminal trip with
  /// immediate stage-2 pure damping — the operator asked for everything
  /// to stop NOW; no slump stage.
  void ForceTripEstop() {
    if (tripped_ && force_stage2_) return;
    reason_ = "operator E-STOP (wire flag)";
    tripped_ = true;
    force_stage2_ = true;
  }

  /// Joint-velocity trip (G1-reference parity, g1_deploy_onnx_ref.cpp:2830:
  /// any |dq| > 35 rad/s aborts the tick; their Stop() then writes the
  /// Kp=0/Kd=8 damping command). A velocity breach means the fall or the
  /// policy divergence is ALREADY violent, so this trips terminally AND
  /// requests immediate stage-2 pure damping — no stage-1 slump.
  void ForceTripVelocity(int mj_idx, double dq_rad_s);

  /// True when stage-2 pure damping must engage immediately (velocity trip).
  bool ForceStage2() const { return force_stage2_; }

 private:
  double      fall_tilt_cos_;
  bool        tripped_ = false;
  bool        force_stage2_ = false;
  std::string reason_;
};

/// Motor thermal monitor (G1-parity port, 2026-08-04 audit #2): per-joint
/// Schmitt-trigger on max(coil, motor) temperature — enter HOT at
/// ``enter_c`` (default 90), exit below ``exit_c`` (default 85) — with
/// throttled reporting so warnings don't spam the log. WARN-ONLY for now
/// (no derating); the PC-side notifier turns reports into PC3-speaker
/// beeps + pad rumble. Also finally answers the open lower-limb heat
/// question: the wire always carried per-motor temps, we just dropped them.
class ThermalMonitor {
 public:
  explicit ThermalMonitor(double enter_c = 90.0, double exit_c = 85.0,
                          double remind_period_s = 5.0)
      : enter_c_(enter_c), exit_c_(exit_c), remind_period_s_(remind_period_s) {
    hot_.fill(false);
  }

  /// Feed current temps; returns a non-empty human-readable report string
  /// when something newly enters/exits HOT or a periodic reminder is due.
  std::string Update(const std::array<double, NUM_DOFS>& coil_c,
                     const std::array<double, NUM_DOFS>& motor_c,
                     double now_s);

  bool  AnyHot()   const { return hot_count_ > 0; }
  int   HotCount() const { return hot_count_; }

 private:
  double enter_c_, exit_c_, remind_period_s_;
  std::array<bool, NUM_DOFS> hot_{};
  int    hot_count_ = 0;
  double last_report_s_ = -1e9;
};

/// Pose-ref starvation watchdog: monitors the age of the most recent ZMQ
/// pose-reference frame received from the operator-side stack (planner /
/// recorder / Quest 3 manager) and trips into a *recoverable* idle state
/// when the wire goes silent for longer than ``stale_threshold_s`` seconds.
///
/// Unlike ``TiltWatchdog`` (which is terminal -- once tripped the robot
/// goes to SAFE_HOLD until the deploy is restarted), this watchdog is
/// designed for the split-topology deploy where the operator stack runs on
/// a separate machine and crosses wifi: a transient drop should land the
/// robot in a safe idle pose (legs locked, default angles, 4x kd) without
/// shutting down the deploy, and an explicit operator chord on the Quest 3
/// (delivered out-of-band on a separate ZMQ topic) brings it back online.
///
/// State machine on the watchdog side:
///
///   not-tripped  --(age >= stale_threshold_s)-->  tripped
///        ^                                            |
///        |                                            v
///   ClearTrip()  <--(operator chord OK AND          ReadyToResume()
///                    fresh frames flowing >=          returns true after
///                    min_fresh_window_s)              fresh frames have
///                                                     been arriving for
///                                                     ``min_fresh_window_s``
///                                                     continuously
///
/// The min-fresh window prevents a flapping wifi link from re-engaging
/// CONTROL on a single fresh frame and then immediately starving again
/// (which would produce repeated SAFE_IDLE entries with full PD ramps in
/// between -- worse than just holding idle).
class PoseRefStarvationWatchdog {
 public:
  /// @param stale_threshold_s   Pose-ref age (seconds) at which Update()
  ///                            trips. Typical 0.3-1.0 s on wifi.
  /// @param min_fresh_window_s  After a trip, ReadyToResume() returns true
  ///                            only after fresh frames have been arriving
  ///                            continuously for this long. Typical 0.5-2.0 s.
  PoseRefStarvationWatchdog(double stale_threshold_s = 0.5,
                            double min_fresh_window_s = 1.0)
      : stale_threshold_s_(stale_threshold_s),
        min_fresh_window_s_(min_fresh_window_s) {}

  void   Reset()        { tripped_ = false; fresh_since_s_ = -1.0; reason_.clear(); }
  void   ClearTrip()    { tripped_ = false; fresh_since_s_ = -1.0; reason_.clear(); }
  bool   Tripped()      const { return tripped_; }
  double StaleThresholdS()  const { return stale_threshold_s_; }
  double MinFreshWindowS()  const { return min_fresh_window_s_; }
  const std::string& Reason() const { return reason_; }

  /// Returns true if the watchdog fires ON THIS TICK (transitioned from
  /// not-tripped to tripped). Subsequent calls return false but Tripped()
  /// stays true until ClearTrip() (or Reset()).
  ///
  /// @param now_s               Current monotonic time (steady_clock).
  /// @param last_rx_monotonic_s steady_clock time of the most recent ZMQ
  ///                            frame received. If the wire has never
  ///                            received a frame, callers should pass a
  ///                            value <= 0; we treat that as ``age = +inf``
  ///                            and trip immediately (the deploy is in
  ///                            split-topology CONTROL state without any
  ///                            operator-side input, which is unsafe).
  bool Update(double now_s, double last_rx_monotonic_s);

  /// Returns true when fresh pose-ref frames have been arriving for at
  /// least ``min_fresh_window_s`` seconds continuously, signalling that
  /// the operator side is healthy and the deploy can safely re-engage
  /// CONTROL on receipt of the operator's resume chord.
  ///
  /// Must be called every tick (even when not tripped) so the watchdog
  /// can maintain the fresh-since-when bookkeeping. Returns false when
  /// not tripped (no need to "resume" if we're already running).
  ///
  /// @param now_s    Current monotonic time (steady_clock).
  /// @param ref_age_s  Pose-ref age (seconds) at this tick.
  bool ReadyToResume(double now_s, double ref_age_s);

  /// Latest computed pose-ref age. Updated by Update() on every call;
  /// reported via x2_debug for the operator/forensic tooling.
  double LatestAgeS() const { return latest_age_s_; }

 private:
  double      stale_threshold_s_;
  double      min_fresh_window_s_;
  bool        tripped_      = false;
  // Monotonic time when fresh frames started arriving again. -1 = no fresh
  // frame seen since the last trip (or since startup). Used by
  // ReadyToResume() to gate exit-from-SAFE_IDLE on a stable wire.
  double      fresh_since_s_ = -1.0;
  double      latest_age_s_  = 0.0;
  std::string reason_;
};

/// Build the post-safety command for one control tick. Encapsulates the
/// dry-run / ramp / watchdog interaction so the main loop stays small.
///

/// Fill ``cmd`` with the VENDOR-fitted STAGE-2 PURE-DAMPING profile
/// (Kp=0, per-joint Kd) -- the same table ApplySafetyStack uses when a fall is
/// committed or the operator e-stops.
///
/// EXTRACTED 2026-08-24. The table used to be a static local inside
/// ApplySafetyStack, which meant SAFE_HOLD could not reach it: SAFE_HOLD
/// returns before the safety stack runs, so an operator e-stop pressed while
/// already in SAFE_HOLD tripped the watchdog and then nothing consumed it --
/// the robot kept publishing its latched (possibly saturated) pose. Sharing
/// ONE definition instead of copying the table is deliberate: duplicated
/// constants are exactly how the deploy/trainer plant drifted apart.
void BuildPureDampingCommand(SafeCommand& cmd,
                             const std::array<double, NUM_DOFS>& default_angles);

/// Called from the 50 Hz control timer. The returned SafeCommand is then
/// published by the 500 Hz writer (which simply re-publishes the latest
/// SafeCommand without modification).
///
/// Two ApplySafetyStack overloads exist:
///
///   * Scalar (legacy): one ``max_target_dev_rad`` applied to all 31 DOFs.
///     Negative value disables the clamp. Kept for callers that don't care
///     about per-group differentiation.
///
///   * Per-DOF (preferred for real-robot deploys): a 31-element array
///     where ``per_dof[i]`` is the clamp on joint ``i`` (MuJoCo order, see
///     ``policy_parameters.hpp::mujoco_joint_names``). Entry < 0 disables
///     the clamp on that joint. Used by ``--max-target-dev-{leg,waist,
///     arm,head}`` so legs (kp ~99) can be tightly clamped while arms
///     (kp ~14) get the room they need to reach operator IK targets.
///
/// In both cases the clamp is applied AFTER the soft-start ramp blend
/// and BEFORE the dry-run gain zeroing. Skipped on a tilt-trip because
/// that branch already pinned the target to ``default_angles``.
SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             double max_target_dev_rad = -1.0);

SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             const std::array<double, NUM_DOFS>& max_target_dev_per_dof);

/// Full-control overload. Same semantics as the per-DOF max_target_dev
/// variant above, but ALSO takes per-DOF kp / kd arrays instead of using
/// the trained ``kps`` / ``kds`` from policy_parameters.hpp directly.
///
/// Use this overload when the operator wants to bump deployment-time PD
/// per joint group (e.g. ``--kp-scale-ankle 1.5`` to recover the loop gain
/// IsaacLab's implicit PD lent training). The values passed here ARE the
/// final effective gains the safety stack will publish on the bus -- any
/// per-group scaling should already be folded in by the caller (see
/// ``BuildPdScalesPerDof`` in x2_deploy_onnx_ref.cpp).
///
/// On a tilt-trip the kd part of the returned SafeCommand is still boosted
/// by 4x (over-damped slump-back), so callers don't need to reproduce that
/// branch. The dry-run zeroing also still applies AFTER the per-DOF kp/kd
/// landed, so dry-run + custom gains gives kp=0/kd=0 like before.
SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             const std::array<double, NUM_DOFS>& max_target_dev_per_dof,
                             const std::array<double, NUM_DOFS>& kp_per_dof,
                             const std::array<double, NUM_DOFS>& kd_per_dof);

}  // namespace agi_x2

#endif  // AGI_X2_SAFETY_HPP
