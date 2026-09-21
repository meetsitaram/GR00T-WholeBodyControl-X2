#include "safety.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace agi_x2 {

// ---------------------------------------------------------------------------
// SoftStartRamp
// ---------------------------------------------------------------------------
double SoftStartRamp::Alpha(double now_s)
{
  if (control_start_s_ < 0.0) {
    control_start_s_ = now_s;
    return 0.0;
  }
  if (ramp_seconds_ <= 0.0) return 1.0;
  const double a = (now_s - control_start_s_) / ramp_seconds_;
  return std::clamp(a, 0.0, 1.0);
}

void SoftStartRamp::Apply(double alpha,
                          std::array<double, NUM_DOFS>& target_mj) const
{
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    const double from = has_from_ ? blend_from_[i] : default_angles[i];
    target_mj[i] = (1.0 - alpha) * from + alpha * target_mj[i];
  }
}

// ---------------------------------------------------------------------------
// TiltWatchdog
// ---------------------------------------------------------------------------
bool TiltWatchdog::Update(double gravity_body_z)
{
  if (tripped_) return false;
  // Upright: gravity[2] ~ -1.0. Trip when it climbs toward 0 or positive.
  // Same convention as eval_x2_mujoco.py::step_once: trip if grav_z > -0.3.
  if (gravity_body_z > fall_tilt_cos_) {
    tripped_ = true;
    const double g = std::clamp(-gravity_body_z, -1.0, 1.0);
    const double tilt_deg = std::acos(g) * 180.0 / 3.14159265358979323846;
    std::ostringstream os;
    os << "tilt watchdog tripped: gravity_body[z]=" << gravity_body_z
       << " > threshold " << fall_tilt_cos_
       << " (~" << static_cast<int>(tilt_deg) << " deg from upright)";
    reason_ = os.str();
    return true;
  }
  return false;
}

std::string ThermalMonitor::Update(const std::array<double, NUM_DOFS>& coil_c,
                                   const std::array<double, NUM_DOFS>& motor_c,
                                   double now_s)
{
  std::ostringstream os;
  bool transition = false;
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    const double t = std::max(coil_c[i], motor_c[i]);
    if (!hot_[i] && t >= enter_c_) {
      hot_[i] = true; ++hot_count_; transition = true;
      os << "motor " << i << " HOT " << t << "C; ";
    } else if (hot_[i] && t < exit_c_) {
      hot_[i] = false; --hot_count_; transition = true;
      os << "motor " << i << " cooled " << t << "C; ";
    }
  }
  if (transition) { last_report_s_ = now_s; return os.str(); }
  if (hot_count_ > 0 && (now_s - last_report_s_) >= remind_period_s_) {
    last_report_s_ = now_s;
    os << hot_count_ << " motor(s) still hot:";
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      if (hot_[i]) os << " " << i << "("
                      << std::max(coil_c[i], motor_c[i]) << "C)";
    }
    return os.str();
  }
  return {};
}

void TiltWatchdog::ForceTripVelocity(int mj_idx, double dq_rad_s)
{
  if (tripped_ && force_stage2_) return;
  std::ostringstream os;
  os << "joint-velocity watchdog tripped: |dq[" << mj_idx << "]|="
     << dq_rad_s << " rad/s (G1-parity bound; fall/divergence committed)";
  reason_ = os.str();
  tripped_ = true;
  force_stage2_ = true;
}

// ---------------------------------------------------------------------------
// PoseRefStarvationWatchdog
// ---------------------------------------------------------------------------
bool PoseRefStarvationWatchdog::Update(double now_s, double last_rx_monotonic_s)
{
  // No frame has ever arrived since the deploy connected. In the split-
  // topology CONTROL path that's unsafe by construction -- the policy is
  // running with the stand-pose default, but the operator wire is dark.
  // Treat as infinitely stale.
  if (last_rx_monotonic_s <= 0.0) {
    latest_age_s_ = std::numeric_limits<double>::infinity();
  } else {
    latest_age_s_ = std::max(0.0, now_s - last_rx_monotonic_s);
  }

  if (tripped_) {
    return false;  // already tripped; subsequent calls return false
  }
  if (latest_age_s_ >= stale_threshold_s_) {
    tripped_ = true;
    fresh_since_s_ = -1.0;
    std::ostringstream os;
    if (std::isinf(latest_age_s_)) {
      os << "pose-ref starvation watchdog tripped: no frames received "
         << "since deploy connected (threshold " << stale_threshold_s_
         << " s)";
    } else {
      os << "pose-ref starvation watchdog tripped: age " << latest_age_s_
         << " s >= threshold " << stale_threshold_s_ << " s";
    }
    reason_ = os.str();
    return true;
  }
  return false;
}

bool PoseRefStarvationWatchdog::ReadyToResume(double now_s, double ref_age_s)
{
  // Update the latest_age_s_ from this call so x2_debug reflects the
  // current age even when the deploy is in SAFE_IDLE and the per-tick
  // Update() above isn't being called (we centralise stale checks here
  // while in SAFE_IDLE).
  latest_age_s_ = ref_age_s;
  if (!tripped_) {
    fresh_since_s_ = -1.0;  // not in starved mode; bookkeeping irrelevant
    return false;
  }
  // Frames count as "fresh" while their age is comfortably under the trip
  // threshold. We use the same threshold for both directions (no hysteresis)
  // because the min-fresh window already enforces stability -- a flapping
  // link can't satisfy ``min_fresh_window_s_`` seconds of continuous
  // freshness if it keeps re-crossing the threshold.
  const bool fresh_now = (ref_age_s < stale_threshold_s_);
  if (!fresh_now) {
    fresh_since_s_ = -1.0;  // freshness streak broken
    return false;
  }
  if (fresh_since_s_ < 0.0) {
    fresh_since_s_ = now_s;
    return false;
  }
  return (now_s - fresh_since_s_) >= min_fresh_window_s_;
}

// ---------------------------------------------------------------------------
// ApplySafetyStack -- full-control variant (preferred; takes per-DOF kp/kd)
// ---------------------------------------------------------------------------
// Stage-2 pure damping, shared by ApplySafetyStack and by SAFE_HOLD's operator
// e-stop path. See safety.hpp for why this is a free function and not a static
// local: SAFE_HOLD returns before the safety stack runs, so it needs its own
// way to reach this profile.
void BuildPureDampingCommand(SafeCommand& cmd,
                             const std::array<double, NUM_DOFS>& default_angles)
{
  static constexpr std::array<double, 31> kVendorDampKd = {
      9.0, 8.0, 8.0, 4.0, 7.5, 3.0,   // left  hip p/r/y, KNEE 4.0, ankle p/r
      9.0, 8.0, 8.0, 4.0, 7.5, 3.0,   // right hip p/r/y, KNEE 4.0, ankle p/r
      5.0, 6.25, 4.0,                 // waist yaw, pitch, roll
      4.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0,  // L shldr p/r/y, elbow, wrist y/p/r
      4.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0,  // R shldr p/r/y, elbow, wrist y/p/r
      0.0, 0.0,                       // head yaw, pitch (vendor: undamped)
  };
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    cmd.target_pos_mj[i] = default_angles[i];  // ignored at kp=0
    cmd.stiffness_mj[i]  = 0.0;
    cmd.damping_mj[i]    = kVendorDampKd[i];
  }
}

SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             const std::array<double, NUM_DOFS>& max_target_dev_per_dof,
                             const std::array<double, NUM_DOFS>& kp_per_dof,
                             const std::array<double, NUM_DOFS>& kd_per_dof)
{
  SafeCommand cmd{};

  // Effective kp/kd: caller passes the already-scaled values (e.g.
  // ``--kp-scale-ankle 1.5`` baked in upstream by BuildPdScalesPerDof).
  // Dry-run zeros these later; tilt-trip overrides damping below.
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    cmd.stiffness_mj[i] = kp_per_dof[i];
    cmd.damping_mj[i]   = kd_per_dof[i];
  }

  // Tilt watchdog runs first so a freshly-tripped command goes to the safe
  // hold-default branch immediately.
  watchdog.Update(current_gravity_body_z);
  if (watchdog.Tripped()) {
    cmd.tilt_trip = true;
    // SINGLE-STAGE trip response (2026-08-25). Any trip -> PURE DAMPING.
    //
    // There used to be a stage 1 "recoverable wobble" step: pin the target to
    // default_angles at UNCHANGED kp (legs ~99 Nm/rad) with kd*4, for up to
    // 0.7 s or until the tilt passed ~55 deg. Removed at operator request.
    //
    // The hazard: a trip means the robot is ALREADY past 72.5 deg from
    // upright. Commanding it back to the upright default pose at full leg kp
    // from there is a hard shove toward vertical -- the robot pushes itself
    // forward/over before damping ever engages. Operator: "an intermediate
    // idle-stand pose ... was a hazard causing the robot to push forward
    // before damping".
    //
    // THE TRADE, STATED PLAINLY: stage 1 existed as a recovery window and had
    // caught real waist-level trips. Removing it converts a recoverable
    // stumble into a committed fall. That is the deliberate choice: at 72.5 deg
    // a recovery is unlikely, and a shove from full kp makes it less likely
    // still, while a compliant fold is survivable.
    //
    // The damping profile is fitted from the VENDOR MC's own damping mode
    // (2026-08-04 hardware capture: 247k samples at 250 Hz through a commanded
    // slump + hand back-driving of every limb; least-squares tau = -Kd*vel per
    // joint, left/right agree to 0.01-0.1). The flat Kd=8 shipped first (G1
    // paper value) matched the legs but over-damped the small high-gear joints
    // 2-8x (wrists worst) -- the operator-reported motor whir/growl during the
    // e-stop damp collapse. DELIBERATE departure from vendor (operator
    // fall-shaping): knees at 4.0 instead of the vendor's 8.0 so the legs are
    // the preferential fold point -- the robot drops onto its KNEES instead of
    // staying stiff-legged and toppling onto its torso. Hips stay at vendor
    // 9/8 to keep slowing torso pitch during the fold.
    BuildPureDampingCommand(cmd, default_angles);
    cmd.reason = watchdog.Reason() + " -> PURE-DAMPING";
    cmd.ramp_alpha = 0.0;
  } else {
    // Normal path: soft-start blend.
    cmd.target_pos_mj = policy_target_mj;
    cmd.ramp_alpha    = ramp.Alpha(now_s);
    ramp.Apply(cmd.ramp_alpha, cmd.target_pos_mj);
    cmd.reason = "ok";
  }

  // Per-joint hard clamp on |target - default|, with PER-DOF thresholds.
  // Applied AFTER the ramp so a small clamp cleanly bounds the worst-case
  // command even when the policy is fully phased in (alpha = 1). Skipped
  // on a tilt-trip because that branch already pinned the target to
  // default and adding a clamp on top would be a no-op. An entry <= 0 in
  // max_target_dev_per_dof disables the clamp on that specific joint.
  //
  // Rationale: a divergent policy (or an obs-construction bug that makes a
  // sane policy look divergent) can emit per-joint targets many radians from
  // the standing pose. With kp ~ 99 Nm/rad on the legs, that becomes
  // hundreds of Nm of impulse the moment MC steps aside and the gains come
  // back from zero. With kp ~ 14 Nm/rad on the arms, the same nominal
  // deviation has 7x less torque, so arms can safely take 5x the angular
  // travel that legs can. Per-DOF clamps let us express that ratio
  // directly: legs/waist tight (e.g. 0.30 rad ~ 17 deg) for upright
  // stability, arms wide (e.g. 1.50 rad ~ 86 deg) for reach-target
  // following.
  if (!cmd.tilt_trip) {
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      const double clamp = max_target_dev_per_dof[i];
      if (clamp <= 0.0) continue;
      const double delta = cmd.target_pos_mj[i] - default_angles[i];
      const double clamped = std::max(-clamp, std::min(clamp, delta));
      cmd.target_pos_mj[i] = default_angles[i] + clamped;
    }
  }

  // Dry-run zeros the gains AFTER everything else, so the wiring (target
  // positions, joint name remap) is still exercised end-to-end while the
  // motors do nothing. Operator-supplied kp/kd scaling is honoured up to
  // this point; dry-run is the unconditional kill.
  if (dry_run) {
    cmd.dry_run = true;
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      cmd.stiffness_mj[i] = 0.0;
      cmd.damping_mj[i]   = 0.0;
    }
  }

  return cmd;
}

// ---------------------------------------------------------------------------
// ApplySafetyStack -- per-DOF clamp variant (uses trained kps/kds as-is)
// ---------------------------------------------------------------------------
SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             const std::array<double, NUM_DOFS>& max_target_dev_per_dof)
{
  // Forward to the full-control overload with the constexpr trained PD as
  // the "effective" gains. Backward-compatible with callers that don't
  // care about deployment-time PD bumps.
  std::array<double, NUM_DOFS> kp_default{};
  std::array<double, NUM_DOFS> kd_default{};
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    kp_default[i] = kps[i];
    kd_default[i] = kds[i];
  }
  return ApplySafetyStack(policy_target_mj, current_gravity_body_z,
                          ramp, watchdog, dry_run, now_s,
                          max_target_dev_per_dof, kp_default, kd_default);
}

// ---------------------------------------------------------------------------
// ApplySafetyStack -- scalar clamp variant (legacy thin wrapper)
// ---------------------------------------------------------------------------
SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             double max_target_dev_rad)
{
  std::array<double, NUM_DOFS> per_dof{};
  per_dof.fill(max_target_dev_rad);
  return ApplySafetyStack(policy_target_mj, current_gravity_body_z,
                          ramp, watchdog, dry_run, now_s, per_dof);
}

}  // namespace agi_x2
