/**
 * @file x2_deploy_onnx_ref.cpp
 * @brief Reference real-robot deploy harness for the AgiBot X2 Ultra.
 *
 * Deploy-harness phase of the X2 ONNX deploy plan.
 *
 * ## Threading model
 *
 *   Thread / Timer    | Rate    | Responsibility
 *   ------------------|---------|----------------------------------------
 *   Subscriptions     | event   | Async ingest of leg/waist/arm/head/imu
 *                     |         | (mutex-guarded write into AimdkIo state)
 *   Control timer     | 50 Hz   | Snapshot state, build obs, run ONNX,
 *                     |         | apply safety stack, push SafeCommand to
 *                     |         | the latest_command_ slot.
 *   Command writer    | 500 Hz  | Read latest_command_ and PublishCommand.
 *                     |         | (10x oversampled re-publish; the policy
 *                     |         | target itself only changes at 50 Hz, but
 *                     |         | streaming it at the firmware's command
 *                     |         | rate keeps the on-bot PD loop fed.)
 *
 * All timers attach to a MultiThreadedExecutor so subscribers can deliver
 * messages while the control timer is mid-inference.
 *
 * ## State machine
 *
 *   STANDBY -> INIT -> WAIT_FOR_CONTROL -> CONTROL -> SAFE_HOLD
 *               ^                                       |
 *               +---------------------------------------+
 *
 *   STANDBY          : OPTIONAL pre-INIT state, only entered when
 *                       --start-trigger-sentinel is set. ROS subscribers
 *                       are active, ONNX is loaded, MC-takeover detectors
 *                       armed -- but the 500 Hz writer is GATED OFF (no
 *                       joint commands published). bash uses this to
 *                       launch the binary BEFORE stop_app + safety gate
 *                       so colcon build / DDS discovery / model load
 *                       overlap with the operator's "Y" decision. Exits
 *                       to INIT when bash touches the trigger sentinel.
 *   INIT             : waiting for first leg/waist/arm/head/IMU message
 *                       (AimdkIo::AllStateFresh(0.5)). Publishes nothing.
 *   WAIT_FOR_CONTROL : have valid state, waiting for operator "go" via
 *                       --autostart-after=N (N seconds delay) OR Ctrl-C-then-rerun
 *                       OR (future) the operator service. Publishes nothing.
 *   CONTROL          : runs the policy, applies the safety stack, publishes
 *                       commands. The 500 Hz writer is allowed to publish.
 *   SAFE_HOLD        : tilt watchdog tripped or fatal error. The 500 Hz
 *                       writer publishes "hold default angles, 4x damping"
 *                       (or pure damping once the down-detect / e-stop
 *                       carve-outs fire) until either the binary is
 *                       restarted or the RECOVER gate below opens.
 *   RECOVER          : assisted stand-up (2026-09-22). Entered from
 *                       SAFE_HOLD once the IMU reads upright (grav_z <
 *                       --recover-upright-cos) AND the body is quiet
 *                       (base |w|, joint |qd| under bounds) for
 *                       --recover-dwell-s: i.e. someone is holding the
 *                       robot up, legs anywhere between crouch and stand.
 *                       Phase A stiffens IN PLACE (gains ramp from the
 *                       latched damping to deploy gains, target = measured
 *                       pose); phase B blends the target to the stand pose
 *                       at <= --recover-max-rate rad/s. Any tilt past
 *                       --tilt-cos aborts back to SAFE_HOLD pure damping.
 *                       Completion re-enters CONTROL exactly like a fresh
 *                       WAIT_FOR_CONTROL start (soft-start ramp from the
 *                       measured pose, watchdogs reset, reference re-anchored).
 *                       --no-safe-hold-recover disables the path.
 *
 * ## CLI (selected)
 *
 *   --model PATH               Path to fused g1+g1_dyn ONNX (required)
 *   --motion PATH              Optional X2M2 motion file for the tokenizer
 *                              reference. Default: StandStill.
 *   --autostart-after SECONDS        Auto-transition WAIT->CONTROL after N seconds.
 *   --dry-run                  Publish stiffness=0/damping=0 (no torque).
 *   --tilt-cos COS             Watchdog threshold (default -0.3).
 *   --ramp-seconds SECONDS     Soft-start blend duration (default 2.0).
 *   --log-dir PATH             Per-tick CSVs go here. Empty = no logs.
 */

#include "aimdk_io.hpp"
#include "deploy_logger.hpp"
#include "math_utils.hpp"
#include "onnx_actor.hpp"
#include "policy_parameters.hpp"
#include "plant_loader.hpp"
#include "proprioception_buffer.hpp"
#include "reference_motion.hpp"
#include "flight_recorder.hpp"
#include "safety.hpp"
#include "stand_pose_loader.hpp"
#include "tokenizer_obs.hpp"
#include "wrist_bypass.hpp"
#include "smpl_obs.hpp"
#include "zmq/smpl_intent_source.hpp"
#include "head_bypass.hpp"
#include "zmq/zmq_debug_publisher.hpp"
#include "zmq/zmq_pose_input_source.hpp"
#include "zmq/zmq_resume_subscriber.hpp"

#include <aimdk_msgs/srv/get_mc_action.hpp>
#include <aimdk_msgs/msg/common_request.hpp>

#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription_options.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include <unistd.h>  // ::write, STDERR_FILENO for async-signal-safe stderr

namespace agi_x2 {

// ────────────────────────────────────────────────────────────────────────────
// Soft-shutdown signal handling (graceful Ctrl-C).
//
// File-scoped atomic flag set by our custom SIGINT/SIGTERM handler and polled
// by OnControl. Must be lock-free + signal-safe; std::atomic<bool> on a
// std::sig_atomic_t-compatible underlying type satisfies that on every
// platform we target. The X2Deploy::OnControl tick (50 Hz) reads this and,
// when set together with a non-empty --soft-shutdown-trigger-sentinel, takes
// the RAMP_OUT → HOLD_FOR_MC path instead of letting rclcpp::shutdown() exit
// the process immediately.
//
// The handler is installed AFTER rclcpp::init in main() so it overrides
// rclcpp's default SIGINT handler. On the SECOND SIGINT within 5 s (or
// 30 s into a hung RAMP_OUT), we fall back to rclcpp::shutdown() so the
// operator always has a fast escape hatch.
static std::atomic<bool>      g_soft_shutdown_requested{false};
static std::atomic<int>       g_sigint_count{0};
static std::atomic<long long> g_first_sigint_ns{0};

// Async-signal-safe write helper. The cast-to-void trick does NOT suppress
// glibc's warn_unused_result attribute on ::write, so we explicitly assign
// to a sink variable and mark it volatile-discarded.
[[gnu::always_inline]]
static inline void SigSafeWriteStderr(const char* msg, std::size_t len)
{
  ssize_t written = ::write(STDERR_FILENO, msg, len);
  (void)written;
}

extern "C" void SoftShutdownSignalHandler(int signum)
{
  using namespace std::chrono;
  const long long now_ns = duration_cast<nanoseconds>(
      steady_clock::now().time_since_epoch()).count();
  const int count = g_sigint_count.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (count == 1) {
    g_first_sigint_ns.store(now_ns, std::memory_order_release);
    g_soft_shutdown_requested.store(true, std::memory_order_release);
    // Async-signal-safe stderr message. write(2) is on the
    // sig-safe list; iostream is not.
    static const char kMsg1[] =
        "\n[soft-shutdown] SIGINT/SIGTERM received -> requesting RAMP_OUT. "
        "Press Ctrl-C again within 5 s to force immediate shutdown.\n";
    SigSafeWriteStderr(kMsg1, sizeof(kMsg1) - 1);
    return;
  }
  // Second signal: check if within the escape-hatch window.
  const long long first_ns = g_first_sigint_ns.load(std::memory_order_acquire);
  const long long dt_ns    = now_ns - first_ns;
  if (dt_ns < 5LL * 1000 * 1000 * 1000) {
    static const char kMsg2[] =
        "\n[soft-shutdown] second SIGINT inside 5 s -> forcing rclcpp::shutdown(). "
        "Bus will be silent until MC restarts.\n";
    SigSafeWriteStderr(kMsg2, sizeof(kMsg2) - 1);
  } else {
    static const char kMsg3[] =
        "\n[soft-shutdown] additional SIGINT -> forcing rclcpp::shutdown().\n";
    SigSafeWriteStderr(kMsg3, sizeof(kMsg3) - 1);
  }
  // rclcpp::shutdown() is documented as signal-safe; this matches what
  // rclcpp's own default handler does.
  rclcpp::shutdown();
  (void)signum;
}


// ---------------------------------------------------------------------------
// CLI parsing (single-purpose; lightweight; no third-party deps)
// ---------------------------------------------------------------------------
struct CliArgs {
  std::string model_path;
  std::string motion_path;             // empty -> StandStillReference
  std::string log_dir;                 // empty -> logging disabled
  // Pre-trigger black box. 0 = OFF (default): opt-in, so no existing run
  // changes behaviour or memory footprint. 30 s costs ~11 MB, pre-allocated at
  // construction so the 500 Hz writer never allocates.
  double      flight_seconds = 0.0;
  // Output dir for the black box. Deliberately INDEPENDENT of --log-dir: the
  // sim stack does not pass --log-dir, so tying the two together silently
  // disabled the recorder in exactly the runs you most want it (caught
  // 2026-08-24 by running the normal sim launch). Fallback chain:
  //   --flight-recorder-dir  ->  --log-dir  ->  /tmp
  std::string flight_dir;
  double      autostart_seconds = -1.0;// negative -> wait for operator stdin "go"
  double      max_duration      = -1.0;// negative -> run until Ctrl-C; positive
                                       // -> shutdown N seconds after entering
                                       // CONTROL. Used by --dry-run smoke tests
                                       // to bound runtime so the operator isn't
                                       // expected to babysit Ctrl-C.
  bool        dry_run           = false;
  // OBSERVE-ONLY. When true the 500 Hz writer NEVER publishes, in ANY state
  // -- including SAFE_HOLD, RAMP_OUT and HOLD_FOR_MC. The policy still runs
  // and every CSV is still written, so this captures exactly what the deploy
  // WOULD have commanded without a single byte reaching the motor bus.
  //
  // NOT the same as --dry-run. --dry-run still PUBLISHES; it merely zeroes
  // the kp/kd inside the message. With MC up that is actively dangerous:
  // two publishers on one bus, and our kp=0/kd=0 at 500 Hz cancels MC's
  // holding gains, so the robot goes limp. --no-publish is the flag that is
  // safe to run alongside MC.
  //
  // HARD REQUIREMENT: MC must retain the robot for the whole run. deploy_x2.sh
  // enforces this by forcing NO_STOP_MC=true whenever --no-publish is set --
  // muting the deploy AND stopping MC would leave nothing commanding the
  // joints at all (zero torque, robot drops).
  bool        no_publish        = false;
  double      tilt_cos          = -0.3;
  double      ramp_seconds      = 2.0;
  // Per-joint hard clamp on |target - default_angles|, in radians. Negative
  // = disabled. See safety.hpp::ApplySafetyStack for rationale; intended for
  // first-powered-bring-up runs (e.g. 0.05 rad ~= 3 deg) so a divergent
  // policy or obs-construction bug cannot drive any joint more than this
  // many radians from the trained standing pose. This is the GLOBAL value
  // applied to every joint that doesn't have a per-group override below.
  double      max_target_dev    = -1.0;
  // Joint-velocity watchdog bound (rad/s); G1-reference parity value 35.
  // Breach = immediate stage-2 pure damping. <=0 disables (sim A/B only).
  double      joint_vel_trip_rad_s = 35.0;
  // Per-group overrides for max_target_dev, indexed by MuJoCo joint group.
  // -1 = inherit the global ``max_target_dev``; >0 = clamp this group at the
  // given radian deviation. Designed for the case where one joint family
  // can safely take more travel than another -- e.g. on X2 Ultra the legs
  // run kp ~99 Nm/rad and the arms kp ~14 Nm/rad, so the same nominal
  // |target - default| produces ~7x the torque on legs. Tight legs +
  // wide arms is the natural setting for any teleop-driven session.
  // Joint-group ranges (MuJoCo joint order, see policy_parameters.hpp::
  // mujoco_joint_names):
  //   leg   -> indices  0..11 (hip pitch/roll/yaw, knee, ankle pitch/roll, both sides)
  //   waist -> indices 12..14 (yaw, pitch, roll)
  //   arm   -> indices 15..28 (shoulder pitch/roll/yaw, elbow,
  //                            wrist yaw/pitch/roll, both sides)
  //   head  -> indices 29..30 (yaw, pitch)
  double      max_target_dev_leg   = -1.0;
  double      max_target_dev_waist = -1.0;
  double      max_target_dev_waist_pitch = -1.0;  // per-joint (MJ 13); wins over group
  // SAFE_HOLD down-detect: auto pure-damping when grav_z stays above this
  // for 300 ms on fresh IMU (robot genuinely down/going down). <= -1.0
  // disables (pre-2026-08-31 behaviour: stand-pose hold until e-stop).
  double      safe_hold_down_damp_cos = -0.25;
  // SAFE_HOLD -> RECOVER assisted stand-up (2026-09-22). Gate: IMU upright
  // (gravity_body[z] < recover_upright_cos; -1.0 = perfectly upright,
  // -0.95 ~ 18 deg) AND quiet (base |ang vel| and every joint |qd| under
  // the bounds) on fresh state for recover_dwell_s. Then gains ramp from
  // the latched (damping) profile to deploy gains over recover_stiffen_s
  // with the target pinned at the measured pose, and the target blends to
  // the stand pose over max(recover_standup_s, max|delta| / recover_max_rate).
  bool        safe_hold_recover       = true;
  double      recover_upright_cos     = -0.95;
  double      recover_dwell_s         = 2.0;
  double      recover_max_base_angvel = 0.5;   // rad/s
  double      recover_max_joint_vel   = 1.0;   // rad/s
  double      recover_stiffen_s       = 1.5;
  double      recover_standup_s       = 4.0;
  double      recover_max_rate        = 0.4;   // rad/s of the fastest joint
  int         recover_max_attempts    = 3;     // per run; <=0 = unlimited
  double      max_target_dev_arm   = -1.0;
  double      max_target_dev_head  = -1.0;
  // Per-group multiplicative trim on the trained PD spec (mirror of the
  // ``--kp-scale-*`` / ``--kd-scale-*`` flags in ``eval_x2_mujoco.py``).
  // The C++ deploy ships ``kps[]`` / ``kds[]`` as the BARE training values
  // (codegen'd from IsaacLab via ``codegen_x2_policy_parameters.py``), but
  // training used IsaacLab's IMPLICIT PD which integrates against the
  // joint-space inertia + armature. MuJoCo (sim profile) and the real
  // robot apply EXPLICIT ctrl-driven torque, so the same numerical KP
  // produces a softer response at deploy time -- documented at length in
  // ``gear_sonic/scripts/eval_x2_mujoco.py:155-200``. Fixed by bumping
  // deployed PD per joint group, NOT by retraining.
  //
  // Defaults to 1.0 (no trim). The G16b-validated default for X2 Ultra
  // is ``ankle=1.5`` (recovers ~+0.5 s mean survival on standing motions);
  // the operator-observed real-robot torso wobble on nudge often clears
  // up with ``waist=1.5`` on top of that (no Python sim equivalent --
  // X2 Ultra real-robot inertia distribution differs from MJCF).
  //
  // Joint family (substring match on the joint name; mirrors Python
  // ``_deployment_pd_scale``):
  //   hip       -> contains "hip"
  //   knee      -> contains "knee"
  //   ankle     -> contains "ankle"
  //   waist     -> contains "waist"
  //   shoulder  -> contains "shoulder"
  //   elbow     -> contains "elbow"
  //   wrist     -> contains "wrist"
  //   head      -> contains "head"
  //
  // Both global and per-group are MULTIPLICATIVE. Effective scale on a
  // joint = global * group (group defaults to 1.0). Negative or zero
  // values are rejected at parse time (cannot make a passive joint
  // active by negating its kp).
  double      kp_scale            = 1.0;
  double      kp_scale_hip        = 1.0;
  double      kp_scale_knee       = 1.0;
  // Ankle is split into pitch vs roll (mirror of waist below). The
  // legacy ``kp_scale_ankle`` alias multiplies BOTH subgroups so older
  // YAMLs / scripts continue to work; effective scale on ankle_pitch
  // becomes kp_scale_ankle * kp_scale_ankle_pitch. Both default to 1.0
  // so alias-only behaviour is unchanged.
  double      kp_scale_ankle        = 1.0;
  double      kp_scale_ankle_pitch  = 1.0;
  double      kp_scale_ankle_roll   = 1.0;
  // Waist is split into yaw vs pitch vs roll (see PdScaleSpec below for
  // why). Two layers of legacy alias are kept for backward compat:
  //   ``kp_scale_waist``    -- applies to ALL three waist subgroups
  //                            (predates any split; matches the very
  //                            first single-knob YAMLs we shipped).
  //   ``kp_scale_waist_pr`` -- applies to BOTH waist_pitch and
  //                            waist_roll (predates the pitch/roll
  //                            split; matches expressive.yaml and the
  //                            pre-2026-05-30 walking_recovery.yaml).
  // New presets should prefer the explicit ``kp_scale_waist_pitch``
  // and ``kp_scale_waist_roll`` knobs below. Effective scale on a
  // waist_pitch joint becomes
  //   kp_scale_waist * kp_scale_waist_pr * kp_scale_waist_pitch
  // which is identity (1*1*1) by default, so omitting the new knobs
  // exactly reproduces the pre-split behaviour.
  //
  // Why split waist_pr further: the 2026-05-30 MC-vs-SONIC nudge scan
  // showed pitch (sagittal lean) at +45% over MC and roll (lateral
  // lean) at -72% under MC -- same trained PD, same actuator, but
  // closed-loop response diverges by axis because the COM moves
  // asymmetrically about the two axes (arms hang symmetric about
  // pitch, swing the whole COM laterally about roll). A single
  // waist_pr knob cannot close the lateral gap without breaking
  // the sagittal axis. Splitting gives one knob per physical axis.
  double      kp_scale_waist        = 1.0;
  double      kp_scale_waist_yaw    = 1.0;
  double      kp_scale_waist_pr     = 1.0;
  double      kp_scale_waist_pitch  = 1.0;
  double      kp_scale_waist_roll   = 1.0;
  double      kp_scale_shoulder   = 1.0;
  double      kp_scale_elbow      = 1.0;
  double      kp_scale_wrist      = 1.0;
  double      kp_scale_head       = 1.0;
  double      kd_scale            = 1.0;
  double      kd_scale_hip        = 1.0;
  double      kd_scale_knee       = 1.0;
  double      kd_scale_ankle        = 1.0;
  double      kd_scale_ankle_pitch  = 1.0;
  double      kd_scale_ankle_roll   = 1.0;
  double      kd_scale_waist        = 1.0;
  double      kd_scale_waist_yaw    = 1.0;
  double      kd_scale_waist_pr     = 1.0;
  double      kd_scale_waist_pitch  = 1.0;
  double      kd_scale_waist_roll   = 1.0;
  double      kd_scale_shoulder   = 1.0;
  double      kd_scale_elbow      = 1.0;
  double      kd_scale_wrist      = 1.0;
  double      kd_scale_head       = 1.0;
  // Symmetric clamp on the raw ONNX action (action_il) BEFORE multiplying by
  // x2_action_scale. This mirrors IsaacLab's training-time clamp applied in
  // ManagerEnvWrapper.step (controlled by config ``action_clip_value`` in
  // gear_sonic/config/manager_env/base_env.yaml, default 20.0). Without this,
  // a saturated/diverged policy can emit O(100 rad) action_il, which the
  // ``--max-target-dev`` safety stack truncates -- but the truncated motor
  // command never matches what training observed, so the policy's
  // ``last_action`` proprioception feedback drifts away from training
  // distribution and the loop runs away. Default 20.0 matches training.
  // Set to a negative value to disable (only useful for parity tests).
  double      action_clip       = 20.0;
  // Soft-EXIT ramp (counterpart to --ramp-seconds, the soft-START). When
  // --max-duration trips we don't want to just stop publishing -- the policy
  // may have left the joints far from the trained standing pose (e.g. mid-
  // swing on a take_a_sip / hot_dog motion), and MC's PD-hold will snap
  // them back the moment start_app is POSTed during cleanup, which can fault
  // the MC unit (red-flashing-light territory on X2 Ultra). RAMP_OUT
  // linearly interpolates target_pos from the last policy command to
  // default_angles over `return_seconds` while keeping deploy-mode kp/kd
  // active, so we drive the joints back instead of letting them flop. Set
  // to <=0 to keep the legacy "shutdown immediately" behaviour.
  double      return_seconds    = 2.0;
  // Output-side first-order EMA on the published joint targets, applied
  // AFTER the policy has produced action_il and AFTER the safety stack has
  // computed sc.target_pos_mj. Purely a real-deploy jitter mitigation; the
  // policy still SEES the same observation it always has, so:
  //   * --obs-dump emits raw policy outputs (this LPF runs after the dump
  //     return path), keeping compare_deploy_vs_python_obs.py bit-exact;
  //   * sim profiles in deploy_x2.sh leave this at 0 (= bypass) so MuJoCo
  //     parity (eval_x2_mujoco.py) is preserved by construction;
  //   * RAMP_OUT and SAFE_HOLD bypass the LPF -- those states already
  //     produce a deliberate trajectory we don't want to attenuate.
  // alpha = 1 - exp(-2*pi*hz*dt) at the 50 Hz OnControl rate, so e.g.
  // hz=8 -> alpha~=0.63 (~5 Hz effective bandwidth). 0 = disabled.
  // Documented under gear_sonic_deploy/configs/real_deploy_tuning/.
  double      target_lpf_hz     = 0.0;
  // Per-group overrides for target_lpf_hz, mirroring the
  // max_target_dev_{leg,waist,arm,head} pattern above. A negative value
  // means "inherit the global ``target_lpf_hz``"; 0 = "disabled on this
  // group regardless of global"; >0 = "use this cutoff on this group".
  //
  // Motivating diagnostic (Phase 1b waist torso-push A/B scan, 2026-05-30):
  // SONIC's waist commanded target moves ~2x faster than MC's at numerically
  // identical PD (waist_roll peak velocity 1.29 vs 0.59 rad/s, +118%) and
  // overshoots its own command by an extra 25-64% in pos_p2p. Motor effort
  // peaks at <=23 N*m on a 48 N*m motor (~50% headroom), so neither kd_scale
  // bumping nor max_target_dev tightening is the right knob. The signal IS
  // command bandwidth -- which a per-group LPF controls directly. Global
  // LPF drops (8.0 -> 4.0) tried earlier hurt the legs because legs need
  // high bandwidth for heel-strike correction; the waist doesn't.
  //
  // Group ranges (MuJoCo joint order, see policy_parameters.hpp::
  // mujoco_joint_names):
  //   leg   -> indices  0..11 (hip+knee+ankle, both sides)
  //   waist -> indices 12..14 (yaw, pitch, roll)
  //   arm   -> indices 15..28 (shoulder+elbow+wrist, both sides)
  //   head  -> indices 29..30 (yaw, pitch)
  //
  // Recommended values:
  //   leg   : inherit global (8 Hz on walking_recovery.yaml). Going lower
  //           reintroduces the heel-strike-correction lag we hit when we
  //           tried global LPF=4 earlier.
  //   waist : 2-3 Hz (1.3-1.9 Hz effective bandwidth). Sits just above the
  //           observed 2.0-2.5 Hz waist_pitch/waist_roll resonance from the
  //           Phase 1b oscillation scan, so the resonance content is
  //           attenuated ~70-80% while slow operator-tracked motion still
  //           passes.
  //   arm   : inherit global, or 10-12 Hz (~6-7.5 Hz bandwidth) if teleop
  //           feels sluggish under the global default.
  //   head  : inherit global. No measured resonance worth treating
  //           differently.
  double      target_lpf_hz_leg   = -1.0;
  double      target_lpf_hz_waist = -1.0;
  double      target_lpf_hz_arm   = -1.0;
  double      target_lpf_hz_head  = -1.0;
  int         intra_op_threads  = 1;
  // Optional topic overrides. Empty string -> use AimdkIo defaults
  // (which match the canonical names in the SDK `topics_and_services`
  // registry: /aima/hal/joint/{leg,waist,arm,head}/{state,command} and
  // /aima/hal/imu/torso/state).
  std::string imu_topic;
  // If non-empty, the first CONTROL tick will write the full inference
  // payload (tokenizer 680 + proprioception 990 + raw policy output 31)
  // to PATH as a binary blob and *immediately exit* (so we don't keep
  // commanding a robot whose obs we don't trust). The companion script
  // ``gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py``
  // diffs that blob against /tmp/x2_step0_isaaclab_lastpt.pt slot-by-slot.
  // Used for debugging policy divergence on the real robot. See
  // docs/source/references/x2_deployment_code.md for the file format.
  std::string obs_dump_path;
  // ────────────────────────────────────────────────────────────────────
  // End-of-run smooth handoff (HOLD_FOR_MC).
  //
  // Default policy: when --max-duration trips, after RAMP_OUT lerps back
  // to the trained ``default_angles``, the deploy node exits and bash
  // POSTs start_app to bring MC back. The robot is therefore in zero
  // torque (PASSIVE_DEFAULT) for the ~5-15 s MC takes to boot. To close
  // that window, the deploy node can stay alive in a ``HOLD_FOR_MC``
  // state holding MC's STAND_DEFAULT pose, while bash drives MC back.
  // The deploy node detects MC's takeover by listening on
  // /aima/hal/joint/{leg,waist}/command for any non-self publisher and
  // exits cleanly the moment MC's first message arrives. Sentinel file
  // is touched on entering HOLD_FOR_MC so bash can sequence start_app.
  // ────────────────────────────────────────────────────────────────────
  // Path to the stand-pose YAML produced by the capture script. Empty
  // = fall back to default_angles for both RAMP_OUT and HOLD_FOR_MC
  // (legacy behaviour). See configs/x2_stand_default_pose.yaml.
  std::string stand_pose_path;
  // Cap on time spent in HOLD_FOR_MC. <=0 disables HOLD_FOR_MC entirely
  // (RAMP_OUT exits the process directly, like before). When >0, the
  // deploy node holds MC's stand pose for up to N seconds waiting for
  // MC's bus takeover, then exits with a warning if MC never came back.
  double      hold_for_mc_timeout_s = 0.0;
  // If non-empty, touch this file when the deploy node enters
  // HOLD_FOR_MC. Lets bash know "policy phase is done; you can now
  // start_app + SetMcAction(STAND_DEFAULT) safely" without parsing
  // stdout. Cleared on clean exit.
  std::string hold_for_mc_sentinel;
  // If non-empty, HOLD_FOR_MC stops auto-exiting on the first detected
  // MC publish (because MC boots in PASSIVE_DEFAULT = zero torque, and
  // exiting then would let the robot go limp). Instead, deploy waits
  // for THIS sentinel to appear -- bash creates it after escalating MC
  // all the way to STAND_DEFAULT. The MC-takeover detector subscribers
  // are still armed (and logged for diagnostics), they just don't
  // trigger exit. The hold_for_mc_timeout_s safety cap still applies.
  std::string hold_for_mc_exit_sentinel;
  // If non-empty, deploy touches this file the moment it sees its
  // FIRST non-self publisher on /aima/hal/joint/{leg,waist}/command
  // (= MC has started publishing again, very likely in
  // PASSIVE_DEFAULT). Bash uses this as a fast signal to start the
  // SetMcAction(JOINT_DEFAULT) escalation immediately, BYPASSING the
  // slow mc_get_action poll loop (which can lag MC's actual first
  // publish by 0.5-0.8 s on the standing-gestures runs). Combined
  // with a fast-poll mode-check in bash, this compresses the
  // PASSIVE_DEFAULT dwell time (= dual-publisher whir window) from
  // ~1.5 s to roughly the SetMcAction round-trip latency. Independent
  // of hold_for_mc_exit_sentinel; touching mc_first_publish_sentinel
  // does NOT trigger exit on its own.
  std::string mc_first_publish_sentinel;
  // ────────────────────────────────────────────────────────────────────
  // Pre-launch STANDBY (cold-warm boot).
  //
  // Default flow without these flags is "deploy starts, immediately
  // tries to enter INIT, the writer streams the safe-hold latch from
  // the moment it has fresh state". That assumes MC is already stopped
  // before the binary launches -- otherwise MC + deploy would fight on
  // the bus during the startup window. The bash script enforces this
  // by stop_app'ing MC before `ros2 run`.
  //
  // The bash script can flip the order so the C++ binary starts BEFORE
  // the operator confirms the safety gate (cold-warm boot: build, ONNX
  // load, ROS subscribers settle while the operator is still reading
  // the prompt). For that, we need a "alive but silent" state where
  // the writer publishes nothing, so deploy doesn't dual-publish with
  // MC. STANDBY is that state, and start_trigger_sentinel is how bash
  // (post-stop_app + verify) tells us "you can take the bus now". When
  // start_trigger_sentinel is empty we skip STANDBY entirely and behave
  // exactly like before.
  // ────────────────────────────────────────────────────────────────────
  // If non-empty, deploy boots into STANDBY (no publishing) and stays
  // there until this file exists. Once it does, STANDBY -> INIT and
  // the rest of the state machine runs as normal. Bash is responsible
  // for stop_app'ing MC BEFORE touching this file.
  std::string start_trigger_sentinel;
  // If non-empty, deploy touches this file once it has reached its
  // ready point (subscribers up, ONNX loaded, MC-takeover detectors
  // armed) so bash can show its safety-gate prompt with confidence
  // that the next step (touch start_trigger_sentinel) is sub-second.
  // Independent of start_trigger_sentinel; safe to use on its own.
  std::string ready_sentinel;
  // ────────────────────────────────────────────────────────────────────
  // Soft-shutdown trigger (graceful Ctrl-C).
  //
  // Default flow without this flag is "Ctrl-C → rclcpp default SIGINT
  // handler → rclcpp::shutdown() → executor.spin() returns → main()
  // exits immediately". That bypasses RAMP_OUT / HOLD_FOR_MC entirely:
  // the bus goes silent for the 1-2 s it takes MC to boot through
  // PASSIVE_DEFAULT (zero torque), so the robot drops noticeably under
  // gravity before MC reaches STAND_DEFAULT.
  //
  // With this flag, bash can request a graceful shutdown by touching
  // soft_shutdown_trigger_sentinel BEFORE sending SIGTERM. In CONTROL
  // state, deploy polls the file every OnControl tick (~50 Hz) and,
  // when it appears, transitions to RAMP_OUT using the same code path
  // that --max-duration uses. The full RAMP_OUT → HOLD_FOR_MC →
  // exit-sentinel chain then runs as normal, so the robot stays under
  // active torque through MC's PASSIVE_DEFAULT boot. Result: no
  // zero-torque window, no drop.
  // ────────────────────────────────────────────────────────────────────
  std::string soft_shutdown_trigger_sentinel;
  // ────────────────────────────────────────────────────────────────────
  // VLA / ZMQ input source (M2 acceptance gate).
  //
  // ``--input-type`` selects what drives the tokenizer's reference window:
  //   * ``motion_file`` (default, legacy): replay an X2M2 .pkl-style file
  //     via ``PklMotionReference`` (requires ``--motion``), or fall back
  //     to ``StandStillReference`` when ``--motion`` is empty.
  //   * ``zmq``: subscribe to a packed-binary ``pose`` topic on
  //     ``--zmq-pose-host:--zmq-pose-port`` and consume the body refs +
  //     hand joints + (forward-compat) motion token published there.
  //     Used by the M2 sim smoke gate (``mock_vla_publish_stand_token.py``)
  //     and, eventually, the real GR00T N1.7 VLA serving the X2.
  //
  // ``--zmq-debug-port`` enables the publisher counterpart so Python
  // tooling (``dump_x2_debug.py``) can subscribe to the deploy's
  // telemetry stream. Set to 0 to disable.
  // ────────────────────────────────────────────────────────────────────
  std::string input_type        = "motion_file";  // {motion_file, zmq}
  // ────────────────────────────────────────────────────────────────────
  // Whole-body Pico/SMPL teleop (first bring-up 2026-08-29).
  //
  // The policy's 680-D command slot is filled from the streamed
  // ``motion_token`` field ([token 64 | zeros 616]) instead of
  // BuildTokenizerObs — the LAPTOP already ran the release smpl encoder +
  // FSQ on the operator's body stream, so encoding must NOT happen again
  // here. Requires --input-type=zmq (token rides the pose message; all
  // staleness/SAFE_IDLE/resume-chord machinery applies unchanged) and a
  // TOKEN-INPUT graph (metadata graph_kind=any2any_token_deploy, stamped
  // by gear_sonic/scripts/frozen_core_token_onnx_export.py). Launching this
  // mode with a pose-encoder graph — or a normal session with a token
  // graph — is REFUSED at startup: 680 floats of the wrong meaning is a
  // silent-failure class this codebase has been burned by (see
  // tokenizer_obs.hpp's scrambled-layout history).
  // Reference poses in the same message still drive the wrist bypass and
  // every safety stage, exactly as in Quest/pad sessions.
  // ────────────────────────────────────────────────────────────────────
  // HYBRID (2026-08-29, operator design): the pad/kplanner session is
  // PRIMARY exactly as today; the pico token stream is an OPTIONAL overlay.
  // Per tick: token source fresh (<0.5 s) AND engaged flag set -> the
  // TOKEN graph drives; otherwise the pose graph + kplanner reference
  // drive (gamepad always takes precedence; pico is the optional one).
  // Transitions re-arm the soft-start ramp so authority changes blend.
  // Mirrors the Quest-VR workflow one layer lower: Quest merges command
  // AUTHORS upstream in the planner; whole-body swaps the policy INPUT,
  // which only the deploy can do.
  bool        whole_body_teleop = false;
  std::string token_model_path;                 // token-input graph (overlay)
  std::string token_zmq_host = "localhost";     // the operator laptop
  int         token_zmq_port = 5556;
  // ---- native dual-head whole-body (2026-09-04): ONE SONIC session, the
  // SMPL stream consumed in-binary, no PC2 token service ------------------
  std::string dual_model_path;                  // *_dual.onnx (graph_kind=native_dual_head)
  std::string smpl_intent_host = "0.0.0.0";     // laptop sender PUB-connects here
  int         smpl_intent_port = 0;             // >0 enables (token service used 5573)
  bool        smpl_intent_bind = true;
  double      wb_stale_s       = 0.5;           // stream silence -> release
  double      wb_warmup_s      = 1.0;           // engage warm-up (window fill)
  double      wb_switch_ramp_s = 0.2;           // action crossfade at a head switch
  bool        wb_hold          = true;          // keep the SMPL head on a link drop while stable
  std::string operator_root_level;              // off|zero|clamp:<deg>|offset:<deg> (LevelOperatorRoot)
  std::string smpl_obs_dump_path;               // gate-3 parity dump (smpl_obs_parity.py)
  std::string zmq_pose_host     = "localhost";
  int         zmq_pose_port     = 5556;
  std::string zmq_pose_topic    = "pose";
  int         zmq_debug_port    = 0;               // 0 = disabled
  std::string zmq_debug_topic   = "x2_debug";
  // ────────────────────────────────────────────────────────────────────
  // Pose-ref starvation watchdog (split-topology safety, --input-type=zmq).
  //
  // Active only when --input-type=zmq AND the deploy has reached CONTROL.
  // Trips into the recoverable SAFE_IDLE state when the age of the most
  // recent received pose frame exceeds ``pose_ref_stale_s`` seconds (= the
  // operator-side stack is dead, wifi dropped, or the publisher froze).
  // In SAFE_IDLE the deploy holds ``default_angles`` with 4x kd and does
  // NOT run policy inference. Exiting SAFE_IDLE requires BOTH:
  //   (a) fresh pose frames arriving for ``pose_ref_min_fresh_s`` seconds
  //       (watchdog reports ReadyToResume() == true), AND
  //   (b) an operator chord (A+B held 1 s on the Quest 3 left controller)
  //       delivered via ZMQ pose_resume topic (see --zmq-resume-*).
  // Both gates are required so a wifi flicker mid-task cannot silently
  // re-engage CONTROL.
  // ────────────────────────────────────────────────────────────────────
  // Watchdog trip threshold (seconds). Pose-ref age >= this → trip.
  // Default 0.5 s is comfortable for wifi: planner runs at 50 Hz (20 ms
  // per frame), so a single dropped frame doesn't trip; sustained drops
  // of ~25 frames do. Set <= 0 to disable the watchdog entirely
  // (equivalent to --disable-pose-ref-watchdog; useful for benchmarks).
  double      pose_ref_stale_s     = 0.5;
  // After a trip, the wire must carry fresh frames for at least this many
  // seconds continuously before ReadyToResume() returns true. Default
  // 1.0 s is enough to confirm a stable link (50 fresh frames at 50 Hz).
  // Pairs with the resume chord; both must hold to re-engage CONTROL.
  double      pose_ref_min_fresh_s = 1.0;
  // Hard-disable the pose-ref watchdog regardless of other flags. Keeps
  // the legacy "ZmqPoseInputSource just holds the last good frame on
  // starvation" behaviour for parity tests / benchmarks. NOT recommended
  // for split-topology production runs -- a stale operator pose is the
  // freeze-while-leaning failure mode that motivated this whole feature.
  bool        disable_pose_ref_watchdog = false;
  // ────────────────────────────────────────────────────────────────────
  // Operator resume chord (companion to the starvation watchdog above).
  //
  // The deploy SUBs to a ``pose_resume`` topic on tcp://<host>:<port> and
  // a single received message bumps an internal monotonic timestamp.
  // The SAFE_IDLE exit logic gates on ``LatestFresh(0.5 s)`` AND the
  // watchdog's ReadyToResume(), so the chord only takes effect when the
  // operator side is currently healthy. See zmq_resume_subscriber.hpp.
  // ────────────────────────────────────────────────────────────────────
  std::string zmq_resume_host   = "localhost";
  int         zmq_resume_port   = 5566;
  std::string zmq_resume_topic  = "pose_resume";
  // ────────────────────────────────────────────────────────────────────
  // MC mode poller (Phase 5b observability).
  //
  // Polls the AimRT MC service ``/aimdk_5Fmsgs/srv/GetMcAction`` at this
  // rate and surfaces the current ``McAction.value`` enum on x2_debug as
  // an int (alongside body_q / dq / action). Pure observability: the
  // deploy does not act on MC mode changes (the x2_motor_monitor.py
  // daemon on PC2 owns the alert + JSONL persistence; this just gets
  // the signal aligned to the policy tick clock for forensic
  // cross-correlation). Set <= 0 to disable polling entirely.
  // ────────────────────────────────────────────────────────────────────
  double      mc_mode_poll_s    = 1.0;
  std::string mc_mode_service   = "/aimdk_5Fmsgs/srv/GetMcAction";
  // ────────────────────────────────────────────────────────────────────
  // Wrist bypass (VR-teleop quality-of-life switch).
  //
  // SONIC's training distribution does not include diverse wrist motion
  // and the smallmotor wrist channels have an x2_action_scale of just
  // 0.0715 (vs ~0.42 on the rest of the arm), so the policy outputs near
  // a single comfort pose for wrist_pitch and pins wrist_roll at the
  // asymmetric joint-range tight side. Empirical: data/lerobot/
  // x2_quest3_sonic_v2/data/chunk-000/episode_000001.parquet shows
  // corr(commanded, executed) ~0.0 for both pitches and 98-99% of frames
  // pinned at +-41 deg for both rolls -- identical with both the iter-2k
  // and iter-25k checkpoints, ruling out a training regression.
  //
  // When --wrist-bypass=ik AND --input-type=zmq, OnControl overwrites
  // target_pos_mj for the 4 broken wrist DOFs with the latest IK
  // reference straight off the ZMQ pose feed BEFORE the safety stack so
  // soft-start ramp + max-target-dev clamp still apply uniformly. The
  // tokenizer obs is unchanged (SONIC still sees the IK reference for
  // ALL 31 dofs), only the final per-tick PD target gets the override.
  // wrist_yaw is left under SONIC because v2 telemetry shows it tracks
  // (corr ~0.8). Default off to preserve sim-to-real fidelity for the
  // motion-file replay path; record_x2_dataset.sh flips it to ``ik``.
  // ────────────────────────────────────────────────────────────────────
  enum class WristBypass { Off, Ik };
  WristBypass wrist_bypass      = WristBypass::Off;
  // Head bypass (2026-08-12): SONIC's head saturates ~+-10 deg; 'ref'
  // drives the two head DOFs straight from the ZMQ reference (the pad
  // head-look feature). Same architecture + gating as the wrist bypass.
  enum class HeadBypass { Off, Ref };
  HeadBypass  head_bypass       = HeadBypass::Off;
};

void PrintUsage()
{
  std::cout
      << "Usage: x2_deploy_onnx_ref --model PATH [options]\n"
      << "  --model PATH               (required) fused g1+g1_dyn ONNX\n"
      << "  --motion PATH              X2M2 reference motion (else stand-still)\n"
      << "  --autostart-after SECONDS        auto-go after N seconds (else wait stdin)\n"
      << "  --max-duration SECONDS     auto-shutdown N seconds after entering CONTROL\n"
      << "                             (default: run until Ctrl-C). Useful for bounded\n"
      << "                             dry-run smoke tests.\n"
      << "  --dry-run                  publish stiffness=0/damping=0\n"
      << "  --tilt-cos COS             tilt watchdog threshold (default -0.3)\n"
      << "  --no-safe-hold-recover     disable the SAFE_HOLD -> RECOVER assisted\n"
      << "                             stand-up (default: enabled). RECOVER opens\n"
      << "                             when the robot is HELD upright and still:\n"
      << "  --recover-upright-cos COS  gravity_body[z] below this = upright (-0.95)\n"
      << "  --recover-dwell-s SEC      upright+quiet dwell before recovering (2.0)\n"
      << "  --recover-max-base-angvel W  quiet bound on base |ang vel| rad/s (0.5)\n"
      << "  --recover-max-joint-vel QD quiet bound on joint |qd| rad/s (1.0)\n"
      << "  --recover-stiffen-s SEC    phase A: gains damping -> deploy in place (1.5)\n"
      << "  --recover-standup-s SEC    phase B: minimum stand-up duration (4.0)\n"
      << "  --recover-max-rate RAD_S   phase B: cap on the fastest joint (0.4)\n"
      << "  --recover-max-attempts N   recoveries per run before SAFE_HOLD is final (3; <=0 unlimited)\n"
      << "  --ramp-seconds SECONDS     soft-start ramp (default 2.0)\n"
      << "  --max-target-dev RAD       per-joint hard clamp on |target-default|,\n"
      << "                             in radians. Negative/omitted = disabled.\n"
      << "                             Use 0.05 (~3 deg) for first powered runs\n"
      << "                             so a divergent policy cannot drive any joint\n"
      << "                             more than RAD away from the standing pose.\n"
      << "                             Acts as the GLOBAL default for joints with no\n"
      << "                             per-group override below.\n"
      << "  --max-target-dev-leg RAD   per-group override (MJ joints 0..11 = both\n"
      << "                             hips, knees, ankles). >0 wins over the global\n"
      << "                             --max-target-dev for these joints; -1/omitted\n"
      << "                             = inherit. Typical pairing: leg=0.30 (~17 deg),\n"
      << "                             arm=1.50 (~86 deg). Legs need to stay tight\n"
      << "                             because their kp ~99 Nm/rad; the same\n"
      << "                             nominal travel produces ~7x the torque arms do.\n"
      << "  --max-target-dev-waist-pitch RAD per-JOINT override for waist_pitch\n"
         "                              (MJ joint 13 only; wins over the waist group\n"
         "                              value -- rated range 0.314 needs a tighter\n"
         "                              clamp than roll/yaw can live with)\n"
      << "  --max-target-dev-waist RAD per-group override (MJ joints 12..14 = waist\n"
      << "                             yaw/pitch/roll). Same semantics as --leg.\n"
      << "  --max-target-dev-arm RAD   per-group override (MJ joints 15..28 = both\n"
      << "                             shoulders, elbows, wrists). Set this loose\n"
      << "                             (e.g. 1.50) for teleop where IK can drive\n"
      << "                             the wrist far from default.\n"
      << "  --max-target-dev-head RAD  per-group override (MJ joints 29..30 = head\n"
      << "                             yaw, pitch). Same semantics as --leg.\n"
      << "  --kp-scale FACTOR          multiplicative trim on the trained KP for ALL\n"
      << "                             joints (default 1.0). The shipped kps[] is the\n"
      << "                             raw IsaacLab training value; deploying it\n"
      << "                             unchanged loses ~1.3-1.5x of effective loop\n"
      << "                             gain because IsaacLab integrates PD implicitly\n"
      << "                             against joint inertia + armature, while real\n"
      << "                             X2 / MuJoCo apply explicit ctrl-driven torque.\n"
      << "                             See gear_sonic/scripts/eval_x2_mujoco.py:155.\n"
      << "                             Per-group scales below are ON TOP of this.\n"
      << "  --kp-scale-hip FACTOR      per-family override (joint-name contains 'hip').\n"
      << "  --kp-scale-knee FACTOR     per-family override (joint-name contains 'knee').\n"
      << "  --kp-scale-ankle FACTOR    LEGACY ankle alias. Multiplies BOTH the\n"
      << "                             ankle_pitch and ankle_roll subgroups (backward\n"
      << "                             compat with the pre-split single-knob YAMLs).\n"
      << "                             Prefer the split knobs below for new presets\n"
      << "                             because MC uses ASYMMETRIC PD across ankle\n"
      << "                             axes (pitch kp=40 kd=3.0, roll kp=30 kd=2.0).\n"
      << "  --kp-scale-ankle-pitch FACTOR  ankle_pitch_joint ONLY. Trained kp=21.38;\n"
      << "                             1.87 matches MC's 40 N*m/rad. The G16b-validated\n"
      << "                             default for sagittal-plane recovery.\n"
      << "  --kp-scale-ankle-roll FACTOR  ankle_roll_joint ONLY. Trained kp=21.38;\n"
      << "                             1.40 matches MC's 30 N*m/rad (deliberately\n"
      << "                             softer than pitch so frontal-plane disturbances\n"
      << "                             absorb without snapping the foot sideways).\n"
      << "  --kp-scale-waist FACTOR    LEGACY waist alias. Multiplies BOTH the waist_yaw\n"
      << "                             and waist_pr subgroups (backward compat with the\n"
      << "                             pre-split single-knob YAMLs). Prefer the split\n"
      << "                             knobs below for new presets -- the trained kps[]\n"
      << "                             differ 2.81x between waist_yaw (40 Nm/rad) and\n"
      << "                             waist_pitch/roll (14.25 Nm/rad), so a single\n"
      << "                             alias cannot match MC's uniform 40 Nm/rad\n"
      << "                             without over-stiffening yaw.\n"
      << "  --kp-scale-waist-yaw FACTOR  waist_yaw_joint ONLY. Trained kp=40.18\n"
      << "                             (matches MC's 40 exactly), so the typical value\n"
      << "                             is 1.00 -- bumping it is more likely to ring than\n"
      << "                             to help.\n"
      << "  --kp-scale-waist-pr FACTOR   LEGACY alias: applies to BOTH waist_pitch and\n"
      << "                             waist_roll. Trained kp=14.25, MC uses 40, so\n"
      << "                             2.81 matches MC exactly. Prefer the per-axis\n"
      << "                             knobs below for new presets -- the 2026-05-30\n"
      << "                             MC-vs-SONIC scan showed pitch and roll close\n"
      << "                             the MC gap in opposite directions (+45% on\n"
      << "                             pitch, -72% on roll), which the combined knob\n"
      << "                             cannot express.\n"
      << "  --kp-scale-waist-pitch FACTOR  waist_pitch_joint ONLY. Composes with the\n"
      << "                             legacy --kp-scale-waist and --kp-scale-waist-pr\n"
      << "                             multiplicatively (default 1.0 = inherit). Use\n"
      << "                             this to tune sagittal-lean (fwd/back) stiffness\n"
      << "                             independently of roll.\n"
      << "  --kp-scale-waist-roll FACTOR  waist_roll_joint ONLY. Composes with the\n"
      << "                             legacy --kp-scale-waist and --kp-scale-waist-pr\n"
      << "                             multiplicatively (default 1.0 = inherit). Use\n"
      << "                             this to tune lateral-lean (left/right)\n"
      << "                             stiffness; closes the -72% lateral compliance\n"
      << "                             gap vs MC that the combined waist_pr knob\n"
      << "                             can't reach without breaking pitch.\n"
      << "  --kp-scale-shoulder FACTOR per-family override (shoulders L/R).\n"
      << "  --kp-scale-elbow FACTOR    per-family override (elbows L/R).\n"
      << "  --kp-scale-wrist FACTOR    per-family override (wrists L/R yaw/pitch/roll).\n"
      << "  --kp-scale-head FACTOR     per-family override (head yaw/pitch).\n"
      << "  --kd-scale FACTOR          like --kp-scale but for damping (default 1.0).\n"
      << "                             Watch the kp/kd ratio: bumping kp without kd\n"
      << "                             reduces effective damping ratio and can ring.\n"
      << "  --kd-scale-{hip,knee,ankle,waist,shoulder,elbow,wrist,head}\n"
      << "                             per-family kd overrides; same group definitions\n"
      << "                             as the kp variants above. The ankle and waist\n"
      << "                             aliases multiply their split subgroups together\n"
      << "                             (same backward-compat semantics as --kp-scale-*).\n"
      << "  --kd-scale-ankle-pitch FACTOR  ankle_pitch_joint ONLY. MC publishes kd=3.0\n"
      << "                             vs trained 0.907 -> 3.31 matches MC. Under-damped\n"
      << "                             ankle_pitch is the usual cause of 'foot-feels-\n"
      << "                             springy' on fwd/back nudges at the ankle.\n"
      << "  --kd-scale-ankle-roll FACTOR  ankle_roll_joint ONLY. MC publishes kd=2.0\n"
      << "                             vs trained 0.907 -> 2.20 matches MC. Should be\n"
      << "                             LESS than the pitch knob for the same reason\n"
      << "                             the KP is lower (frontal plane is intrinsically\n"
      << "                             more rigid; less damping needed).\n"
      << "  --kd-scale-waist-yaw FACTOR  waist_yaw_joint ONLY. MC publishes kd=8.0 vs\n"
      << "                             trained 2.56 -> 3.13 matches MC. Trunk damping\n"
      << "                             is critical for nudge rejection: bumping kd is\n"
      << "                             usually safer than bumping kp.\n"
      << "  --kd-scale-waist-pr FACTOR   LEGACY alias: applies to BOTH waist_pitch and\n"
      << "                             waist_roll. MC publishes kd=5.0 vs trained\n"
      << "                             0.907 -> 5.51 matches MC. Prefer the per-axis\n"
      << "                             knobs below for new presets (see\n"
      << "                             --kp-scale-waist-pr rationale).\n"
      << "  --kd-scale-waist-pitch FACTOR  waist_pitch_joint ONLY. Composes with the\n"
      << "                             legacy --kd-scale-waist and --kd-scale-waist-pr\n"
      << "                             multiplicatively (default 1.0 = inherit). The\n"
      << "                             sagittal-axis push response is the ONE waist\n"
      << "                             metric where SONIC already exceeds MC (+45%\n"
      << "                             stronger); don't bump this above MC-match\n"
      << "                             without explicit fwd/back-nudge scan evidence.\n"
      << "  --kd-scale-waist-roll FACTOR  waist_roll_joint ONLY. Composes with the\n"
      << "                             legacy --kd-scale-waist and --kd-scale-waist-pr\n"
      << "                             multiplicatively (default 1.0 = inherit). The\n"
      << "                             lateral-axis push response under SONIC at the\n"
      << "                             combined waist_pr knob sits at -72% of MC's\n"
      << "                             resistance, AND the policy moves its commanded\n"
      << "                             target with the disturbance (gait tracker),\n"
      << "                             so the failure mode is not simply 'too soft'\n"
      << "                             or 'too damped'. Direction of trim must be\n"
      << "                             determined by a post-fix MC-vs-SONIC lateral\n"
      << "                             nudge scan; do NOT prescribe an a priori\n"
      << "                             value here without that data.\n"
      << "  --action-clip RAD          symmetric clip on the raw ONNX action\n"
      << "                             (action_il) BEFORE x2_action_scale (default\n"
      << "                             20.0, matches training-time\n"
      << "                             config.action_clip_value in\n"
      << "                             gear_sonic/config/manager_env/base_env.yaml).\n"
      << "                             Negative = disabled (parity tests only).\n"
      << "  --return-seconds SECONDS   soft-exit ramp duration (default 2.0). When\n"
      << "                             --max-duration trips, lerp target_pos from\n"
      << "                             the last policy command to default_angles\n"
      << "                             over SECONDS (deploy-mode PD active) before\n"
      << "                             shutdown -- prevents MC from snapping joints\n"
      << "                             back at handoff (red-fault on X2 Ultra).\n"
      << "                             Set 0 to disable (legacy immediate-shutdown).\n"
      << "  --target-lpf-hz HZ         REAL-DEPLOY ONLY: first-order EMA cutoff\n"
      << "                             applied to the published joint targets to\n"
      << "                             tame leg/waist jitter caused by noisy real\n"
      << "                             sensor obs. Bypassed in RAMP_OUT/SAFE_HOLD.\n"
      << "                             Default 0 (disabled). Sim parity profiles\n"
      << "                             MUST leave this at 0 -- the LPF is invisible\n"
      << "                             to --obs-dump (raw target preserved) but it\n"
      << "                             changes what the bus sees, which would\n"
      << "                             diverge from eval_x2_mujoco.py's reference.\n"
      << "                             Acts as the GLOBAL default for joints with\n"
      << "                             no per-group override (see --target-lpf-hz-*).\n"
      << "  --target-lpf-hz-leg HZ     Per-group override for legs (MJ joints 0..11\n"
      << "                             = hip+knee+ankle, both sides). <0 = inherit\n"
      << "                             global; 0 = disabled on this group; >0 = use\n"
      << "                             this cutoff. Default <0 (= inherit).\n"
      << "  --target-lpf-hz-waist HZ   Per-group override for the waist (MJ joints\n"
      << "                             12..14 = yaw/pitch/roll). Recommended 2-3 Hz\n"
      << "                             to kill the observed 2.0-2.5 Hz waist\n"
      << "                             resonance under SONIC active walking.\n"
      << "  --target-lpf-hz-arm HZ     Per-group override for the arms (MJ joints\n"
      << "                             15..28). <0 = inherit global. Raise above\n"
      << "                             global if teleop arm motion feels sluggish.\n"
      << "  --target-lpf-hz-head HZ    Per-group override for the head (MJ joints\n"
      << "                             29..30). <0 = inherit global.\n"
      << "  --log-dir PATH             write per-tick CSVs to PATH\n"
      << "  --intra-op-threads N       ONNX session threads (default 1)\n"
      << "  --imu-topic NAME           override IMU topic (default /aima/hal/imu/torso/state;\n"
      << "                             use /aima/hal/imu/torse/state on firmware that\n"
      << "                             ships with the SDK-example typo)\n"
      << "  --obs-dump PATH            DEBUG: dump the first CONTROL-tick obs\n"
      << "                             (tokenizer + proprioception + raw action) to\n"
      << "                             PATH as a binary blob and exit immediately.\n"
      << "                             Pair with --dry-run + --autostart-after for a\n"
      << "                             deterministic capture from a known robot pose.\n"
      << "                             See compare_deploy_vs_isaaclab_obs.py.\n"
      << "  --stand-default-pose PATH  YAML file capturing MC's STAND_DEFAULT pose\n"
      << "                             (configs/x2_stand_default_pose.yaml). When\n"
      << "                             provided, RAMP_OUT lerps to *this* pose and\n"
      << "                             HOLD_FOR_MC publishes it -- so the joints land\n"
      << "                             exactly where MC will resume from, eliminating\n"
      << "                             the takeover step. Empty = use default_angles\n"
      << "                             (legacy: ~12-34 deg snap on takeover).\n"
      << "  --hold-for-mc-timeout-s SECONDS\n"
      << "                             Stay alive after RAMP_OUT for up to SECONDS,\n"
      << "                             holding MC's STAND_DEFAULT pose, until MC's\n"
      << "                             first joint command arrives on the bus (then\n"
      << "                             exit cleanly). 0 = disabled (legacy: exit at\n"
      << "                             end of RAMP_OUT, MC bus is silent until bash\n"
      << "                             POSTs start_app). Recommended: 15.\n"
      << "  --hold-for-mc-sentinel PATH\n"
      << "                             Touch PATH on entering HOLD_FOR_MC. Used by\n"
      << "                             deploy_x2.sh to sequence start_app +\n"
      << "                             SetMcAction(STAND_DEFAULT) at the right\n"
      << "                             moment. Empty = no sentinel.\n"
      << "  --hold-for-mc-exit-sentinel PATH\n"
      << "                             When set, HOLD_FOR_MC stops auto-exiting on\n"
      << "                             the first MC publish (because MC boots in\n"
      << "                             PASSIVE_DEFAULT = zero torque -- exiting then\n"
      << "                             would let the robot go limp). Instead, the\n"
      << "                             deploy node waits for PATH to appear; bash\n"
      << "                             creates it once MC has been escalated all the\n"
      << "                             way back to STAND_DEFAULT. The hold-for-mc\n"
      << "                             timeout is still a hard upper bound.\n"
      << "  --mc-first-publish-sentinel PATH\n"
      << "                             Touch PATH the moment the takeover detector\n"
      << "                             sees its first non-self publisher on\n"
      << "                             /aima/hal/joint/{leg,waist}/command (= MC\n"
      << "                             has come back online, likely in PASSIVE).\n"
      << "                             bash uses this as a fast trigger to start\n"
      << "                             SetMcAction(JOINT_DEFAULT) without waiting\n"
      << "                             for MC's mode service to respond. Does NOT\n"
      << "                             trigger exit on its own (see exit-sentinel).\n"
      << "  --start-trigger-sentinel PATH\n"
      << "                             Boot into STANDBY (writer suppressed) and wait\n"
      << "                             for PATH to exist before advancing to INIT.\n"
      << "                             Lets bash launch deploy AHEAD of stop_app +\n"
      << "                             safety gate to overlap colcon build / ONNX\n"
      << "                             load / DDS discovery with the operator's\n"
      << "                             confirmation. Empty = boot straight to INIT.\n"
      << "  --ready-sentinel PATH      Touch PATH on the first STANDBY tick to tell\n"
      << "                             bash that subscribers / ONNX / takeover\n"
      << "                             detectors are armed and the safety gate can\n"
      << "                             be shown. Independent of start-trigger; safe\n"
      << "                             to use on its own.\n"
      << "  --soft-shutdown-trigger-sentinel PATH\n"
      << "                             Polled at 50 Hz in CONTROL state. When PATH\n"
      << "                             appears, deploy transitions to RAMP_OUT (same\n"
      << "                             code path as --max-duration) and follows the\n"
      << "                             full RAMP_OUT -> HOLD_FOR_MC -> exit-sentinel\n"
      << "                             handoff chain. Lets bash convert Ctrl-C into a\n"
      << "                             graceful shutdown so the robot stays under\n"
      << "                             torque through MC's PASSIVE_DEFAULT boot (no\n"
      << "                             zero-torque drop). Empty = legacy behaviour\n"
      << "                             (immediate exit on SIGINT, ~1-2 s drop while\n"
      << "                             MC boots back to STAND_DEFAULT).\n"
      << "  --input-type TYPE          {motion_file,zmq} (default motion_file). When\n"
      << "                             'zmq', the tokenizer reference window is\n"
      << "                             driven by a ZMQ 'pose' topic publisher\n"
      << "                             (mock-VLA helper or real GR00T VLA). With\n"
      << "                             'motion_file' (legacy) the deploy uses\n"
      << "                             --motion (or StandStill if --motion empty).\n"
      << "  --whole-body-teleop        HYBRID whole-body overlay: pad/kplanner session\n"
      << "                             runs as primary (unchanged); when the laptop's\n"
      << "                             token stream is FRESH and ENGAGED, the token\n"
      << "                             graph (--token-model) drives instead. Stream\n"
      << "                             loss or disengage falls back to the pad.\n"
      << "  --token-model PATH         Token-input ONNX for the overlay (metadata\n"
      << "                             graph_kind=any2any_token_deploy required).\n"
      << "  --token-zmq-host HOST      Token sender host (the operator laptop).\n"
      << "  --token-zmq-port PORT      Token sender port (default 5556).\n"
      << "  --dual-model PATH          NATIVE DUAL-HEAD graph (graph_kind=native_dual_head,\n"
      << "                             native_dual_head_onnx_export.py): one SONIC session\n"
      << "                             with the pose-ref AND smpl heads; the laptop's SMPL\n"
      << "                             intent stream is consumed in-binary (no token\n"
      << "                             service). Requires --whole-body-teleop and\n"
      << "                             --smpl-intent-port; --model/--token-model unused.\n"
      << "  --smpl-intent-port PORT    SUB endpoint for the laptop's pico_intent stream\n"
      << "                             (bound here, the sender connects; 5573).\n"
      << "  --smpl-intent-host HOST    Bind/connect host (default 0.0.0.0).\n"
      << "  --smpl-intent-connect      Connect instead of bind.\n"
      << "  --wb-stale-s S             Stream silence that releases the smpl head (0.5).\n"
      << "  --wb-warmup-s S            Engage warm-up before the smpl head drives (1.0).\n"
      << "  --wb-switch-ramp-s S       Action crossfade at a head switch (0.2).\n"
      << "  --operator-root-level M    Level the OPERATOR root (Pico SMPL pelvis) before the smpl obs:\n"
      << "                             off|zero|clamp:<deg>|offset:<deg> (default off; 2026-09-15).\n"
      << "  --no-wb-hold               On a link drop, release at once instead of\n"
      << "                             holding the smpl head on the latched window\n"
      << "                             while the robot is stable.\n"
      << "  --smpl-obs-dump PATH       Dual-head parity dump: per tick the window, base\n"
      << "                             quat, yaw_align, wrists and the 840-d smpl obs\n"
      << "                             (checked by gear_sonic/scripts/smpl_obs_parity.py).\n"
      << "  --zmq-pose-host HOST       Host of the ZMQ pose publisher (default localhost).\n"
      << "  --zmq-pose-port PORT       Port of the ZMQ pose publisher (default 5556).\n"
      << "  --zmq-pose-topic TOPIC     Topic prefix to subscribe to (default 'pose').\n"
      << "  --zmq-debug-port PORT      If > 0, bind a PUB socket on this port and\n"
      << "                             publish per-tick x2_debug telemetry frames\n"
      << "                             in the packed-binary wire format consumed by\n"
      << "                             gear_sonic/scripts/dump_x2_debug.py. 0 disables.\n"
      << "  --zmq-debug-topic TOPIC    Topic prefix for telemetry frames (default 'x2_debug').\n"
      << "  --wrist-bypass MODE        {off, ik} (default 'off'). When 'ik' AND\n"
      << "                             --input-type=zmq, OnControl overwrites\n"
      << "                             target_pos_mj for the 4 broken wrist DOFs\n"
      << "                             (left/right wrist_pitch + wrist_roll, MJ\n"
      << "                             indices {20,21,27,28}) with the latest IK\n"
      << "                             reference from the ZMQ pose feed BEFORE the\n"
      << "                             safety stack. SONIC still drives the other 27\n"
      << "                             DOFs (legs, waist, shoulders, elbows,\n"
      << "                             wrist_yaw, head). Use 'ik' for VR teleop /\n"
      << "                             VLA dataset recording where SONIC's wrist\n"
      << "                             attractor masks the operator's hand pose;\n"
      << "                             keep 'off' for sim-to-real fidelity tests so\n"
      << "                             the policy's own commands reach every joint.\n"
      << "  --pose-ref-stale-s SEC     Split-topology safety (--input-type=zmq).\n"
      << "                             Pose-ref age (s) at which the deploy trips\n"
      << "                             into SAFE_IDLE (legs locked, default angles,\n"
      << "                             4x kd, no policy inference). Default 0.5 s\n"
      << "                             (~25 dropped frames at 50 Hz). Set <=0 to\n"
      << "                             disable (equivalent to --disable-pose-ref-\n"
      << "                             watchdog). Pairs with --pose-ref-min-fresh-s\n"
      << "                             and the resume chord for recovery.\n"
      << "  --pose-ref-min-fresh-s SEC After a SAFE_IDLE trip, the wire must carry\n"
      << "                             fresh frames for this many seconds continuously\n"
      << "                             before ReadyToResume() flips true. Default 1.0.\n"
      << "                             Prevents a flapping wifi link from oscillating\n"
      << "                             in/out of SAFE_IDLE on each fresh frame.\n"
      << "  --disable-pose-ref-watchdog Hard-disable the pose-ref watchdog regardless\n"
      << "                             of other flags. Restores legacy 'hold last good\n"
      << "                             frame on starvation' behaviour. Use only for\n"
      << "                             parity tests / benchmarks; NOT recommended for\n"
      << "                             split-topology production (the freeze-while-\n"
      << "                             leaning failure mode is exactly what this\n"
      << "                             watchdog catches).\n"
      << "  --zmq-resume-host HOST     Operator chord PUB host (the laptop running\n"
      << "                             quest3_manager_x2.py). Default 'localhost'.\n"
      << "                             In split topology pass the laptop IP.\n"
      << "  --zmq-resume-port PORT     Resume PUB port (default 5566).\n"
      << "  --zmq-resume-topic TOPIC   Resume topic (default 'pose_resume'). The\n"
      << "                             Quest 3 manager publishes a single multipart\n"
      << "                             message [topic, ts_monotonic_ns_le_i64]\n"
      << "                             when the operator holds A+B for 1 s on the\n"
      << "                             left controller.\n"
      << "  --mc-mode-poll-s SEC       Poll the MC GetMcAction service every SEC\n"
      << "                             seconds in a background timer (default 1.0).\n"
      << "                             Publishes the current McAction.value on\n"
      << "                             x2_debug as mc_action_mode for forensic\n"
      << "                             cross-correlation with policy/state. Set <=0\n"
      << "                             to disable (e.g. for sim runs where MC isn't\n"
      << "                             present). Pure observability -- the deploy\n"
      << "                             never acts on the result; the motor monitor\n"
      << "                             daemon owns the alerts + JSONL persistence.\n"
      << "  --mc-mode-service NAME     Override the MC mode service name\n"
      << "                             (default '/aimdk_5Fmsgs/srv/GetMcAction').\n"
      << "  --help, -h                 show this help\n";
}

CliArgs ParseCli(int argc, char** argv)
{
  CliArgs a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
      }
      return argv[++i];
    };
    if (s == "--help" || s == "-h") { PrintUsage(); std::exit(0); }
    else if (s == "--model")             a.model_path        = next("--model");
    else if (s == "--motion")            a.motion_path       = next("--motion");
    else if (s == "--log-dir")           a.log_dir           = next("--log-dir");
    else if (s == "--flight-recorder-seconds")
      a.flight_seconds = std::stod(next("--flight-recorder-seconds"));
    else if (s == "--flight-recorder-dir")
      a.flight_dir = next("--flight-recorder-dir");
    else if (s == "--autostart-after")         a.autostart_seconds = std::stod(next("--autostart-after"));
    else if (s == "--max-duration")      a.max_duration      = std::stod(next("--max-duration"));
    else if (s == "--dry-run")           a.dry_run           = true;
    else if (s == "--no-publish")        a.no_publish        = true;
    else if (s == "--tilt-cos")          a.tilt_cos          = std::stod(next("--tilt-cos"));
    else if (s == "--ramp-seconds")      a.ramp_seconds      = std::stod(next("--ramp-seconds"));
    else if (s == "--max-target-dev")    a.max_target_dev    = std::stod(next("--max-target-dev"));
    else if (s == "--joint-vel-trip")    a.joint_vel_trip_rad_s = std::stod(next("--joint-vel-trip"));
    else if (s == "--max-target-dev-leg")   a.max_target_dev_leg   = std::stod(next("--max-target-dev-leg"));
    else if (s == "--max-target-dev-waist") a.max_target_dev_waist = std::stod(next("--max-target-dev-waist"));
    else if (s == "--max-target-dev-waist-pitch") a.max_target_dev_waist_pitch = std::stod(next("--max-target-dev-waist-pitch"));
    else if (s == "--safe-hold-down-damp-cos") a.safe_hold_down_damp_cos = std::stod(next("--safe-hold-down-damp-cos"));
    else if (s == "--no-safe-hold-recover")  a.safe_hold_recover = false;
    else if (s == "--safe-hold-recover")     a.safe_hold_recover = true;
    else if (s == "--recover-upright-cos")   a.recover_upright_cos = std::stod(next("--recover-upright-cos"));
    else if (s == "--recover-dwell-s")       a.recover_dwell_s = std::stod(next("--recover-dwell-s"));
    else if (s == "--recover-max-base-angvel") a.recover_max_base_angvel = std::stod(next("--recover-max-base-angvel"));
    else if (s == "--recover-max-joint-vel") a.recover_max_joint_vel = std::stod(next("--recover-max-joint-vel"));
    else if (s == "--recover-stiffen-s")     a.recover_stiffen_s = std::stod(next("--recover-stiffen-s"));
    else if (s == "--recover-standup-s")     a.recover_standup_s = std::stod(next("--recover-standup-s"));
    else if (s == "--recover-max-rate")      a.recover_max_rate = std::stod(next("--recover-max-rate"));
    else if (s == "--recover-max-attempts")  a.recover_max_attempts = std::stoi(next("--recover-max-attempts"));
    else if (s == "--max-target-dev-arm")   a.max_target_dev_arm   = std::stod(next("--max-target-dev-arm"));
    else if (s == "--max-target-dev-head")  a.max_target_dev_head  = std::stod(next("--max-target-dev-head"));
    else if (s == "--kp-scale")             a.kp_scale          = std::stod(next("--kp-scale"));
    else if (s == "--kp-scale-hip")         a.kp_scale_hip      = std::stod(next("--kp-scale-hip"));
    else if (s == "--kp-scale-knee")        a.kp_scale_knee     = std::stod(next("--kp-scale-knee"));
    else if (s == "--kp-scale-ankle")       a.kp_scale_ankle        = std::stod(next("--kp-scale-ankle"));
    else if (s == "--kp-scale-ankle-pitch") a.kp_scale_ankle_pitch  = std::stod(next("--kp-scale-ankle-pitch"));
    else if (s == "--kp-scale-ankle-roll")  a.kp_scale_ankle_roll   = std::stod(next("--kp-scale-ankle-roll"));
    else if (s == "--kp-scale-waist")        a.kp_scale_waist        = std::stod(next("--kp-scale-waist"));
    else if (s == "--kp-scale-waist-yaw")    a.kp_scale_waist_yaw    = std::stod(next("--kp-scale-waist-yaw"));
    else if (s == "--kp-scale-waist-pr")     a.kp_scale_waist_pr     = std::stod(next("--kp-scale-waist-pr"));
    else if (s == "--kp-scale-waist-pitch")  a.kp_scale_waist_pitch  = std::stod(next("--kp-scale-waist-pitch"));
    else if (s == "--kp-scale-waist-roll")   a.kp_scale_waist_roll   = std::stod(next("--kp-scale-waist-roll"));
    else if (s == "--kp-scale-shoulder")    a.kp_scale_shoulder   = std::stod(next("--kp-scale-shoulder"));
    else if (s == "--kp-scale-elbow")       a.kp_scale_elbow    = std::stod(next("--kp-scale-elbow"));
    else if (s == "--kp-scale-wrist")       a.kp_scale_wrist    = std::stod(next("--kp-scale-wrist"));
    else if (s == "--kp-scale-head")        a.kp_scale_head     = std::stod(next("--kp-scale-head"));
    else if (s == "--kd-scale")             a.kd_scale          = std::stod(next("--kd-scale"));
    else if (s == "--kd-scale-hip")         a.kd_scale_hip      = std::stod(next("--kd-scale-hip"));
    else if (s == "--kd-scale-knee")        a.kd_scale_knee     = std::stod(next("--kd-scale-knee"));
    else if (s == "--kd-scale-ankle")       a.kd_scale_ankle        = std::stod(next("--kd-scale-ankle"));
    else if (s == "--kd-scale-ankle-pitch") a.kd_scale_ankle_pitch  = std::stod(next("--kd-scale-ankle-pitch"));
    else if (s == "--kd-scale-ankle-roll")  a.kd_scale_ankle_roll   = std::stod(next("--kd-scale-ankle-roll"));
    else if (s == "--kd-scale-waist")        a.kd_scale_waist        = std::stod(next("--kd-scale-waist"));
    else if (s == "--kd-scale-waist-yaw")    a.kd_scale_waist_yaw    = std::stod(next("--kd-scale-waist-yaw"));
    else if (s == "--kd-scale-waist-pr")     a.kd_scale_waist_pr     = std::stod(next("--kd-scale-waist-pr"));
    else if (s == "--kd-scale-waist-pitch")  a.kd_scale_waist_pitch  = std::stod(next("--kd-scale-waist-pitch"));
    else if (s == "--kd-scale-waist-roll")   a.kd_scale_waist_roll   = std::stod(next("--kd-scale-waist-roll"));
    else if (s == "--kd-scale-shoulder")    a.kd_scale_shoulder   = std::stod(next("--kd-scale-shoulder"));
    else if (s == "--kd-scale-elbow")       a.kd_scale_elbow    = std::stod(next("--kd-scale-elbow"));
    else if (s == "--kd-scale-wrist")       a.kd_scale_wrist    = std::stod(next("--kd-scale-wrist"));
    else if (s == "--kd-scale-head")        a.kd_scale_head     = std::stod(next("--kd-scale-head"));
    else if (s == "--action-clip")       a.action_clip       = std::stod(next("--action-clip"));
    else if (s == "--return-seconds")    a.return_seconds    = std::stod(next("--return-seconds"));
    else if (s == "--target-lpf-hz")     a.target_lpf_hz     = std::stod(next("--target-lpf-hz"));
    else if (s == "--target-lpf-hz-leg")   a.target_lpf_hz_leg   = std::stod(next("--target-lpf-hz-leg"));
    else if (s == "--target-lpf-hz-waist") a.target_lpf_hz_waist = std::stod(next("--target-lpf-hz-waist"));
    else if (s == "--target-lpf-hz-arm")   a.target_lpf_hz_arm   = std::stod(next("--target-lpf-hz-arm"));
    else if (s == "--target-lpf-hz-head")  a.target_lpf_hz_head  = std::stod(next("--target-lpf-hz-head"));
    else if (s == "--intra-op-threads")  a.intra_op_threads  = std::stoi(next("--intra-op-threads"));
    else if (s == "--imu-topic")         a.imu_topic         = next("--imu-topic");
    else if (s == "--obs-dump")          a.obs_dump_path     = next("--obs-dump");
    else if (s == "--stand-default-pose") a.stand_pose_path  = next("--stand-default-pose");
    else if (s == "--hold-for-mc-timeout-s")
      a.hold_for_mc_timeout_s = std::stod(next("--hold-for-mc-timeout-s"));
    else if (s == "--hold-for-mc-sentinel")
      a.hold_for_mc_sentinel = next("--hold-for-mc-sentinel");
    else if (s == "--hold-for-mc-exit-sentinel")
      a.hold_for_mc_exit_sentinel = next("--hold-for-mc-exit-sentinel");
    else if (s == "--mc-first-publish-sentinel")
      a.mc_first_publish_sentinel = next("--mc-first-publish-sentinel");
    else if (s == "--start-trigger-sentinel")
      a.start_trigger_sentinel = next("--start-trigger-sentinel");
    else if (s == "--ready-sentinel")
      a.ready_sentinel = next("--ready-sentinel");
    else if (s == "--soft-shutdown-trigger-sentinel")
      a.soft_shutdown_trigger_sentinel = next("--soft-shutdown-trigger-sentinel");
    else if (s == "--input-type")
      a.input_type = next("--input-type");
    else if (s == "--zmq-pose-host")
      a.zmq_pose_host = next("--zmq-pose-host");
    else if (s == "--zmq-pose-port")
      a.zmq_pose_port = std::stoi(next("--zmq-pose-port"));
    else if (s == "--zmq-pose-topic")
      a.zmq_pose_topic = next("--zmq-pose-topic");
    else if (s == "--zmq-debug-port")
      a.zmq_debug_port = std::stoi(next("--zmq-debug-port"));
    else if (s == "--zmq-debug-topic")
      a.zmq_debug_topic = next("--zmq-debug-topic");
    else if (s == "--wrist-bypass") {
      const std::string v = next("--wrist-bypass");
      if      (v == "off") a.wrist_bypass = CliArgs::WristBypass::Off;
      else if (v == "ik")  a.wrist_bypass = CliArgs::WristBypass::Ik;
      else throw std::runtime_error(
          "--wrist-bypass must be 'off' or 'ik', got: " + v);
    }
    else if (s == "--head-bypass") {
      const std::string v = next("--head-bypass");
      if      (v == "off") a.head_bypass = CliArgs::HeadBypass::Off;
      else if (v == "ref") a.head_bypass = CliArgs::HeadBypass::Ref;
      else throw std::runtime_error(
          "--head-bypass must be 'off' or 'ref', got: " + v);
    }
    else if (s == "--whole-body-teleop")
      a.whole_body_teleop = true;
    else if (s == "--token-model")
      a.token_model_path = next("--token-model");
    else if (s == "--token-zmq-host")
      a.token_zmq_host = next("--token-zmq-host");
    else if (s == "--token-zmq-port")
      a.token_zmq_port = std::stoi(next("--token-zmq-port"));
    else if (s == "--dual-model")
      a.dual_model_path = next("--dual-model");
    else if (s == "--smpl-intent-port")
      a.smpl_intent_port = std::stoi(next("--smpl-intent-port"));
    else if (s == "--smpl-intent-host")
      a.smpl_intent_host = next("--smpl-intent-host");
    else if (s == "--smpl-intent-connect")
      a.smpl_intent_bind = false;
    else if (s == "--wb-stale-s")
      a.wb_stale_s = std::stod(next("--wb-stale-s"));
    else if (s == "--wb-warmup-s")
      a.wb_warmup_s = std::stod(next("--wb-warmup-s"));
    else if (s == "--wb-switch-ramp-s")
      a.wb_switch_ramp_s = std::stod(next("--wb-switch-ramp-s"));
    else if (s == "--operator-root-level")
      a.operator_root_level = next("--operator-root-level");
    else if (s == "--no-wb-hold")
      a.wb_hold = false;
    else if (s == "--smpl-obs-dump")
      a.smpl_obs_dump_path = next("--smpl-obs-dump");
    else if (s == "--pose-ref-stale-s")
      a.pose_ref_stale_s = std::stod(next("--pose-ref-stale-s"));
    else if (s == "--pose-ref-min-fresh-s")
      a.pose_ref_min_fresh_s = std::stod(next("--pose-ref-min-fresh-s"));
    else if (s == "--disable-pose-ref-watchdog")
      a.disable_pose_ref_watchdog = true;
    else if (s == "--zmq-resume-host")
      a.zmq_resume_host = next("--zmq-resume-host");
    else if (s == "--zmq-resume-port")
      a.zmq_resume_port = std::stoi(next("--zmq-resume-port"));
    else if (s == "--zmq-resume-topic")
      a.zmq_resume_topic = next("--zmq-resume-topic");
    else if (s == "--mc-mode-poll-s")
      a.mc_mode_poll_s = std::stod(next("--mc-mode-poll-s"));
    else if (s == "--mc-mode-service")
      a.mc_mode_service = next("--mc-mode-service");
    else {
      throw std::runtime_error("unknown argument: " + s);
    }
  }
  if (a.model_path.empty()) {
    throw std::runtime_error("--model is required");
  }
  if (a.input_type != "motion_file" && a.input_type != "zmq") {
    throw std::runtime_error(
        "--input-type must be 'motion_file' or 'zmq', got: " + a.input_type);
  }
  if (a.wrist_bypass == CliArgs::WristBypass::Ik && a.input_type != "zmq") {
    throw std::runtime_error(
        "--wrist-bypass=ik requires --input-type=zmq (no IK reference is "
        "available on the motion_file path; the bypass would be a no-op)");
  }
  if (a.head_bypass == CliArgs::HeadBypass::Ref && a.input_type != "zmq") {
    throw std::runtime_error(
        "--head-bypass=ref requires --input-type=zmq (no reference is "
        "available on the motion_file path; the bypass would be a no-op)");
  }

  // Reject non-positive PD scales. Zero would silently disable a joint
  // family (kp = 0 = passive), which is almost certainly an operator typo
  // and would let the policy push the robot around with only damping in
  // the loop. Negative values would invert the sign of the corrective
  // torque, which is unrecoverable. Better to fail at parse time.
  auto reject_nonpos = [](const char* flag, double v) {
    if (v <= 0.0) {
      throw std::runtime_error(std::string(flag) +
                               " must be > 0 (got " + std::to_string(v) +
                               "); use 1.0 for no trim");
    }
  };
  reject_nonpos("--kp-scale",          a.kp_scale);
  reject_nonpos("--kp-scale-hip",      a.kp_scale_hip);
  reject_nonpos("--kp-scale-knee",     a.kp_scale_knee);
  reject_nonpos("--kp-scale-ankle",       a.kp_scale_ankle);
  reject_nonpos("--kp-scale-ankle-pitch", a.kp_scale_ankle_pitch);
  reject_nonpos("--kp-scale-ankle-roll",  a.kp_scale_ankle_roll);
  reject_nonpos("--kp-scale-waist",        a.kp_scale_waist);
  reject_nonpos("--kp-scale-waist-yaw",    a.kp_scale_waist_yaw);
  reject_nonpos("--kp-scale-waist-pr",     a.kp_scale_waist_pr);
  reject_nonpos("--kp-scale-waist-pitch",  a.kp_scale_waist_pitch);
  reject_nonpos("--kp-scale-waist-roll",   a.kp_scale_waist_roll);
  reject_nonpos("--kp-scale-shoulder",  a.kp_scale_shoulder);
  reject_nonpos("--kp-scale-elbow",    a.kp_scale_elbow);
  reject_nonpos("--kp-scale-wrist",    a.kp_scale_wrist);
  reject_nonpos("--kp-scale-head",     a.kp_scale_head);
  reject_nonpos("--kd-scale",          a.kd_scale);
  reject_nonpos("--kd-scale-hip",      a.kd_scale_hip);
  reject_nonpos("--kd-scale-knee",     a.kd_scale_knee);
  reject_nonpos("--kd-scale-ankle",       a.kd_scale_ankle);
  reject_nonpos("--kd-scale-ankle-pitch", a.kd_scale_ankle_pitch);
  reject_nonpos("--kd-scale-ankle-roll",  a.kd_scale_ankle_roll);
  reject_nonpos("--kd-scale-waist",        a.kd_scale_waist);
  reject_nonpos("--kd-scale-waist-yaw",    a.kd_scale_waist_yaw);
  reject_nonpos("--kd-scale-waist-pr",     a.kd_scale_waist_pr);
  reject_nonpos("--kd-scale-waist-pitch",  a.kd_scale_waist_pitch);
  reject_nonpos("--kd-scale-waist-roll",   a.kd_scale_waist_roll);
  reject_nonpos("--kd-scale-shoulder",  a.kd_scale_shoulder);
  reject_nonpos("--kd-scale-elbow",    a.kd_scale_elbow);
  reject_nonpos("--kd-scale-wrist",    a.kd_scale_wrist);
  reject_nonpos("--kd-scale-head",     a.kd_scale_head);

  return a;
}

// ---------------------------------------------------------------------------
// Per-DOF max_target_dev synthesizer.
//
// Maps the (global, leg, waist, arm, head) CLI scalars onto the 31-element
// per-DOF array consumed by ApplySafetyStack. A per-group value <= 0 means
// "inherit the global". The global itself can also be <= 0 (= disabled);
// in that case any joint without a positive group override gets -1 in the
// output array, which ApplySafetyStack interprets as "no clamp on this DOF".
//
// Group ranges (MuJoCo joint order, see policy_parameters.hpp::
// mujoco_joint_names):
//   leg   -> indices  0..11
//   waist -> indices 12..14
//   arm   -> indices 15..28
//   head  -> indices 29..30
//
// Free function (not a class member) so it's trivially testable in isolation
// and visible to whoever's reading the safety wiring without having to step
// inside X2Deploy.
struct MaxTargetDevSpec {
  double global;
  double leg;
  double waist;
  double arm;
  double head;
  // Per-JOINT override inside the waist group (2026-08-31, §22.6 values
  // finally earned): waist_pitch's rated range (0.314) is smaller than
  // any group value that leaves roll/yaw usable — the ONE joint whose
  // clamp must differ from its group. Applied AFTER the group fill, so
  // it wins over --max-target-dev-waist for MJ joint 13 only.
  double waist_pitch;
};

static std::array<double, NUM_DOFS>
BuildMaxTargetDevPerDof(const MaxTargetDevSpec& s)
{
  std::array<double, NUM_DOFS> arr{};
  arr.fill(s.global);
  if (s.leg   > 0.0) for (std::size_t i = 0;  i <= 11; ++i) arr[i] = s.leg;
  if (s.waist > 0.0) for (std::size_t i = 12; i <= 14; ++i) arr[i] = s.waist;
  if (s.arm   > 0.0) for (std::size_t i = 15; i <= 28; ++i) arr[i] = s.arm;
  if (s.head  > 0.0) for (std::size_t i = 29; i <= 30; ++i) arr[i] = s.head;
  if (s.waist_pitch > 0.0) arr[13] = s.waist_pitch;  // per-joint, wins over group
  return arr;
}

// ---------------------------------------------------------------------------
// Per-DOF target_lpf alpha synthesizer.
//
// Mirrors BuildMaxTargetDevPerDof: maps the (global, leg, waist, arm, head)
// CLI cutoff scalars onto a 31-element per-DOF EMA-alpha array consumed by
// the OnControl hot loop. Convention for the input scalars (matches CliArgs
// docs and --target-lpf-hz-* help):
//   < 0  -> "inherit global"
//   == 0 -> "disabled on this group" (no smoothing, output = raw target)
//   > 0  -> "use this Hz cutoff on this group"
//
// The global itself follows the original target_lpf_hz convention: > 0
// = enabled at that cutoff, otherwise disabled. This means a robot deployed
// with the global at 0 (default) and only --target-lpf-hz-waist=3 will
// smooth only the waist DOFs and leave every other DOF passthrough.
//
// alpha is the standard first-order EMA coefficient at the OnControl tick
// rate ``dt`` (50 Hz, dt = 0.02 s): alpha = 1 - exp(-2*pi*hz*dt). alpha = 0
// in the output array means "no smoothing on this DOF" (passthrough);
// 0 < alpha < 1 means "EMA with this coefficient"; the hot path branches
// on alpha == 0 to skip the EMA for unsmoothed DOFs cheaply.
//
// Group ranges (MuJoCo joint order, see policy_parameters.hpp::
// mujoco_joint_names):
//   leg   -> indices  0..11
//   waist -> indices 12..14
//   arm   -> indices 15..28
//   head  -> indices 29..30
struct TargetLpfSpec {
  double global;  // > 0 = enabled, otherwise disabled
  double leg;     // < 0 inherit, == 0 disabled, > 0 = cutoff Hz
  double waist;
  double arm;
  double head;
  double dt;      // OnControl tick period in seconds (0.02 for 50 Hz)
};

static std::array<double, NUM_DOFS>
BuildTargetLpfAlphaPerDof(const TargetLpfSpec& s)
{
  constexpr double pi = 3.14159265358979323846;
  auto hz_to_alpha = [&](double hz) -> double {
    if (hz <= 0.0) return 0.0;  // disabled or passthrough
    return 1.0 - std::exp(-2.0 * pi * hz * s.dt);
  };

  // Step 1: fill with the global (or 0 if global is disabled).
  const double alpha_global = hz_to_alpha(s.global > 0.0 ? s.global : 0.0);
  std::array<double, NUM_DOFS> arr{};
  arr.fill(alpha_global);

  // Step 2: per-group overrides. Negative = inherit (do nothing here);
  // zero or positive = explicit override (write into the slice). A zero
  // override deliberately CLEARS the global on that group, which is the
  // whole point of supporting a "disabled on this group" semantic
  // independent of the global default.
  auto apply = [&](double hz_group, std::size_t lo, std::size_t hi) {
    if (hz_group < 0.0) return;  // inherit global
    const double alpha_group = hz_to_alpha(hz_group);
    for (std::size_t i = lo; i <= hi; ++i) arr[i] = alpha_group;
  };
  apply(s.leg,    0,  11);
  apply(s.waist, 12,  14);
  apply(s.arm,   15,  28);
  apply(s.head,  29,  30);
  return arr;
}

// ---------------------------------------------------------------------------
// Per-DOF PD-scale synthesizer.
//
// Mirrors ``gear_sonic/scripts/eval_x2_mujoco.py::_deployment_pd_scale``:
// match each joint name against family substrings ("hip", "knee", "ankle",
// "waist", "shoulder", "elbow", "wrist", "head") and use the matching
// per-family scale (default 1.0). Then multiply by the global scale.
//
// The result is the FINAL effective PD that the safety stack will publish
// on the bus. Positive only (validated at parse time). Used independently
// for kp and kd via two ``PdScaleSpec`` instances.
struct PdScaleSpec {
  double global;
  double hip;
  double knee;
  // Ankle is split into pitch vs roll because MC uses ASYMMETRIC PD on
  // those subgroups (ankle_pitch kp=40 kd=3.0; ankle_roll kp=30 kd=2.0;
  // 2026-05-15 scan). Trained kps are uniform (both 21.38) so a single
  // ``ankle`` knob forces operator to pick "match pitch and over-stiffen
  // roll" or "match roll and under-stiffen pitch". Split lets us hit
  // both MC values exactly.
  double ankle_pitch;
  double ankle_roll;
  // Waist is split THREE ways: yaw, pitch, roll. Two reasons stacked:
  //
  // (1) Trained kps for yaw vs pitch/roll differ 2.81x (waist_yaw
  //     kp=40.18, waist_pitch/roll kp=14.25). MC publishes a uniform
  //     kp=40 across all three (operator scan 2026-05-15:
  //     mc_motor_scan_1778884089.jsonl), so matching MC requires
  //     ~1.0x on waist_yaw and ~2.81x on waist_pitch/roll. A single
  //     ``waist`` knob couldn't express that.
  //
  // (2) The 2026-05-30 MC-vs-SONIC torso-nudge scan showed pitch and
  //     roll diverge in opposite directions from MC even at numerically
  //     identical PD: pitch (sagittal lean) ran +45% STRONGER than MC
  //     in push-resistance, while roll (lateral lean) ran -72% WEAKER.
  //     The asymmetry comes from the load distribution (arms hang
  //     symmetric about pitch, swing the whole COM laterally about
  //     roll) AND from the policy being a gait-tracker that moves
  //     WITH lateral disturbance. The waist_pr knob (which moved
  //     pitch and roll together) couldn't close the lateral gap
  //     without over-tightening pitch and re-introducing the
  //     forward/back wobble we'd already tuned out.
  //
  // Splitting waist_pr -> waist_pitch + waist_roll gives one knob per
  // physical axis, matching the ankle_pitch/ankle_roll split we
  // already do.
  double waist_yaw;
  double waist_pitch;
  double waist_roll;
  double shoulder;
  double elbow;
  double wrist;
  double head;
};

static double FamilyScaleForJointName(const PdScaleSpec& s,
                                      const std::string& jname)
{
  // Order matters only for substrings that overlap. The waist split is
  // checked subgroup-by-subgroup -- waist_yaw / waist_pitch /
  // waist_roll are mutually exclusive on the X2 joint name list, so
  // each falls into its own branch. All other families are pairwise
  // disjoint, so iteration order doesn't matter for them.
  if (jname.find("ankle")    != std::string::npos) {
    // ankle_pitch_joint vs ankle_roll_joint (no "ankle_yaw" exists on
    // X2). Identical structure to the waist split below.
    if (jname.find("ankle_pitch") != std::string::npos) return s.ankle_pitch;
    return s.ankle_roll;  // ankle_roll_joint (and any future ankle subgroup
                          // not yet enumerated; defensive default)
  }
  if (jname.find("knee")     != std::string::npos) return s.knee;
  if (jname.find("hip")      != std::string::npos) return s.hip;
  if (jname.find("waist")    != std::string::npos) {
    if (jname.find("waist_yaw")   != std::string::npos) return s.waist_yaw;
    if (jname.find("waist_pitch") != std::string::npos) return s.waist_pitch;
    return s.waist_roll;  // waist_roll_joint (defensive default for any
                          // future waist subgroup name we haven't enumerated)
  }
  if (jname.find("shoulder") != std::string::npos) return s.shoulder;
  if (jname.find("elbow")    != std::string::npos) return s.elbow;
  if (jname.find("wrist")    != std::string::npos) return s.wrist;
  if (jname.find("head")     != std::string::npos) return s.head;
  return 1.0;  // Unknown family -> no trim. Defensive; should not happen
               // for the codegen'd 31-joint X2 set.
}

static std::array<double, NUM_DOFS>
BuildPdScalesPerDof(const PdScaleSpec& s)
{
  std::array<double, NUM_DOFS> arr{};
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    arr[i] = s.global * FamilyScaleForJointName(s, mujoco_joint_names[i]);
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Top-level controller -- mirrors the G1Deploy class but vastly slimmer.
// ---------------------------------------------------------------------------
class X2Deploy {
 public:
  // STANDBY -> INIT -> WAIT_FOR_CONTROL -> CONTROL -> RAMP_OUT -> HOLD_FOR_MC
  //              ^                                       |
  //              +---------------------------------------+
  //                                                      |
  //                                                      v
  //                                                  SAFE_HOLD (terminal)
  //
  // STANDBY is an optional "alive but silent" state at the head of the
  // state machine, used when --start-trigger-sentinel is set. In STANDBY:
  //   * subscribers are active (ROS state is being collected so the
  //     INIT->WAIT freshness check passes immediately on transition)
  //   * the 500 Hz writer is GATED OFF (no joint commands published)
  //   * we poll the start-trigger sentinel each control tick (20 ms)
  // bash uses this to launch deploy in the background AHEAD of the
  // safety gate (parallelising colcon build / ONNX load / DDS discovery
  // with the operator reading the prompt). Once the operator confirms
  // and bash POSTs stop_app + verify, bash touches the trigger sentinel
  // and we move STANDBY -> INIT in the next tick.
  //
  // Without --start-trigger-sentinel deploy boots straight to INIT
  // (legacy path): bash must stop_app BEFORE launching the binary.
  //
  // HOLD_FOR_MC keeps the deploy node alive after RAMP_OUT publishing
  // MC's STAND_DEFAULT pose with MC-stand kp/kd (firmer than deploy
  // gains, matching MC's own stiffness for a stable static hold),
  // until any non-self publisher appears on /aima/hal/joint/{leg,waist}/
  // command (= MC has taken back over). The exit is fast: the DDS
  // callback for MC's first publish fires sub-ms after MC enters
  // JOINT_DEFAULT, and the next OnControl tick exits within <= 20 ms
  // (50 Hz). The exit-sentinel (touched by deploy_x2.sh once MC is in
  // STAND_DEFAULT) is honoured as a redundant backup so an external
  // mode switch (mobile app, ROS service from another shell) can free
  // us even if the takeover detector misses. On takeover (or on
  // hold-for-mc-timeout-s timeout) we shut down cleanly. If
  // --hold-for-mc-timeout-s is 0,
  // RAMP_OUT exits the process directly (legacy behaviour).
  enum class State {
    STANDBY,
    INIT,
    WAIT_FOR_CONTROL,
    CONTROL,
    SAFE_IDLE,    // ← NEW: recoverable starvation hold (split-topology safety)
    RAMP_OUT,
    HOLD_FOR_MC,
    SAFE_HOLD,
    RECOVER,      // ← assisted stand-up out of SAFE_HOLD (see header)
  };

  // ────────────────────────────────────────────────────────────────────
  // SAFE_IDLE (Phase 2 of the split-topology safety plan).
  //
  // Reached from CONTROL when the pose-ref starvation watchdog trips
  // (PoseRefStarvationWatchdog::Update returns true on the
  // current age == now - LastReceivedMonotonicS()). In SAFE_IDLE:
  //
  //   * the writer keeps publishing at 500 Hz, latched to default_angles
  //     with deploy-mode kp and 4x kd (matches the tilt-trip slump
  //     branch), so the body stays under torque -- it doesn't go limp
  //     like PASSIVE_DEFAULT would on MC. Operator can let go of the
  //     gantry and the robot continues to hold the standing pose.
  //   * the policy is NOT inferred (we skip the rest of the OnControl
  //     fast path, including obs construction). The deploy is "alive
  //     but coasting" -- minimal CPU + no commitment to whatever
  //     stale pose the operator was last commanding.
  //   * pose-ref freshness keeps being tracked. As soon as the wire
  //     resumes and the watchdog's ReadyToResume() flips true AND an
  //     operator chord arrives on the resume topic within the last
  //     0.5 s, deploy re-enters CONTROL with ramp_alpha reset to 0
  //     (so SoftStartRamp blends back from default to the live policy
  //     output over --ramp-seconds; no torque shock).
  //
  // Unlike SAFE_HOLD (which is terminal -- deploy stays there until
  // restart), SAFE_IDLE is explicitly recoverable. The dual-gate exit
  // (fresh wire + operator chord) is a deliberate design choice: a wifi
  // flicker on its own can never re-engage CONTROL silently, no matter
  // how briefly the link recovers.
  // ────────────────────────────────────────────────────────────────────

  X2Deploy(rclcpp::Node::SharedPtr node, const CliArgs& cli)
      : node_(node),
        cli_(cli),
        ramp_(cli.ramp_seconds),
        watchdog_(cli.tilt_cos),
        pose_ref_watchdog_(cli.pose_ref_stale_s, cli.pose_ref_min_fresh_s),
        logger_(cli.log_dir.empty() ? std::string{} : cli.log_dir,
                /*enabled=*/!cli.log_dir.empty()),
        recorder_(NUM_DOFS, cli.flight_seconds,
                  !cli.flight_dir.empty() ? cli.flight_dir
                  : (!cli.log_dir.empty() ? cli.log_dir : std::string("/tmp")))
  {
    if (cli.imu_topic.empty()) {
      aimdk_io_ = std::make_unique<AimdkIo>(node_);
    } else {
      aimdk_io_ = std::make_unique<AimdkIo>(
          node_,
          /*leg=*/  "/aima/hal/joint/leg",
          /*waist=*/"/aima/hal/joint/waist",
          /*arm=*/  "/aima/hal/joint/arm",
          /*head=*/ "/aima/hal/joint/head",
          /*imu=*/  cli.imu_topic);
      RCLCPP_INFO(node_->get_logger(),
                  "AimdkIo: IMU topic overridden via CLI -> '%s'",
                  cli.imu_topic.c_str());
    }

    if (cli.input_type == "zmq") {
      // VLA-driven path. The ZmqPoseInputSource subscribes to the pose
      // topic and publishes itself as a ReferenceMotion drop-in: until the
      // first body-bearing frame arrives, Sample() returns default_angles
      // + identity quat (matches StandStillReference exactly), so the
      // tokenizer obs is well-defined even with no VLA on the wire.
      auto zmq_src = ZmqPoseInputSource::Connect(
          cli.zmq_pose_host, cli.zmq_pose_port, cli.zmq_pose_topic);
      RCLCPP_WARN(node_->get_logger(),
                  "Reference motion: ZmqPoseInputSource bound to "
                  "tcp://%s:%d (topic='%s'). Until the first body-bearing "
                  "pose frame arrives, Sample() returns the trained stand "
                  "pose -- safe to start CONTROL on a sub-second window "
                  "without a VLA on the wire.",
                  cli.zmq_pose_host.c_str(), cli.zmq_pose_port,
                  cli.zmq_pose_topic.c_str());
      zmq_pose_source_ = zmq_src.get();  // observer pointer for hand-joint readback
      ref_motion_ = std::move(zmq_src);
    } else if (cli.motion_path.empty()) {
      ref_motion_ = std::make_unique<StandStillReference>();
      RCLCPP_INFO(node_->get_logger(),
                  "Reference motion: StandStill (default standing pose)");
    } else {
      ref_motion_ = PklMotionReference::Load(cli.motion_path);
      RCLCPP_INFO(node_->get_logger(),
                  "Reference motion: PklMotionReference '%s'",
                  cli.motion_path.c_str());
    }

    // Optional packed-binary x2_debug PUB sink. Mirrors the input wire
    // format end-to-end; gear_sonic/scripts/dump_x2_debug.py is its
    // reference consumer. Must run on a different port from the input
    // source so the SUB/PUB direction can't loop back on itself.
    if (cli.zmq_debug_port > 0) {
      try {
        zmq_debug_pub_ = ZmqDebugPublisher::Bind(
            cli.zmq_debug_port, cli.zmq_debug_topic);
        RCLCPP_WARN(node_->get_logger(),
                    "x2_debug telemetry: PUB bound on tcp://*:%d (topic='%s'). "
                    "Use gear_sonic/scripts/dump_x2_debug.py --port %d --topic %s "
                    "to inspect.",
                    cli.zmq_debug_port, cli.zmq_debug_topic.c_str(),
                    cli.zmq_debug_port, cli.zmq_debug_topic.c_str());
      } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(),
                     "x2_debug PUB failed to bind on port %d: %s",
                     cli.zmq_debug_port, e.what());
      }
    }

    onnx_actor_ = std::make_unique<OnnxActor>(cli.model_path, cli.intra_op_threads);
    // ── whole-body-teleop mode/graph cross-check (both directions) ──────
    // A token graph fed pose-built obs — or a pose-encoder graph fed a
    // token — is 680 floats of the wrong MEANING with the right shape:
    // exactly the silent failure class of the scrambled-layout incident
    // (tokenizer_obs.hpp). Refuse at startup, before anything moves.
    {
      const std::string kind = onnx_actor_->GetCustomMetadata("graph_kind");
      if (kind == "any2any_token_deploy") {
        throw std::runtime_error(
            "primary model " + cli.model_path + " is a TOKEN-input graph; "
            "the primary must be the pose-encoder graph (the token graph "
            "goes in --token-model for the whole-body overlay).");
      }
      if (cli.whole_body_teleop) {
        if (cli.input_type != "zmq") {
          throw std::runtime_error(
              "--whole-body-teleop requires --input-type=zmq (the pad/"
              "kplanner pose channel stays primary).");
        }
        if (!cli.dual_model_path.empty()) {
          // NATIVE DUAL-HEAD (2026-09-04): replace the pose-graph session
          // with the dual graph and consume the laptop's SMPL stream here.
          if (cli.smpl_intent_port <= 0) {
            throw std::runtime_error(
                "--dual-model requires --smpl-intent-port (the laptop's "
                "pico_intent stream, 5573).");
          }
          onnx_actor_ = std::make_unique<OnnxActor>(cli.dual_model_path,
                                                    cli.intra_op_threads);
          const std::string dkind = onnx_actor_->GetCustomMetadata("graph_kind");
          if (dkind != "native_dual_head") {
            throw std::runtime_error(
                "--dual-model must carry metadata graph_kind=native_dual_head "
                "(native_dual_head_onnx_export.py); got '" + dkind + "'");
          }
          smpl_ori_mode_ = onnx_actor_->GetCustomMetadata("ori_mode");
          if (smpl_ori_mode_ != "full" && smpl_ori_mode_ != "heading") {
            throw std::runtime_error(
                "--dual-model metadata ori_mode must be 'full' or 'heading', got '" +
                smpl_ori_mode_ + "'");
          }
          smpl_source_ = SmplIntentSource::Open(cli.smpl_intent_host, cli.smpl_intent_port,
                                                cli.smpl_intent_bind);
          if (!cli.smpl_obs_dump_path.empty()) {
            smpl_dump_.open(cli.smpl_obs_dump_path, std::ios::binary | std::ios::trunc);
            if (!smpl_dump_) {
              throw std::runtime_error("--smpl-obs-dump: cannot open " + cli.smpl_obs_dump_path);
            }
            RCLCPP_WARN(node_->get_logger(), "SMPL obs parity dump -> %s",
                        cli.smpl_obs_dump_path.c_str());
          }
          RCLCPP_WARN(node_->get_logger(),
                      "WHOLE-BODY DUAL-HEAD armed: %s (ori_mode=%s); SMPL intent %s "
                      "tcp://%s:%d; pad/kplanner PRIMARY; smpl head drives while the "
                      "stream is fresh (<%.2fs) AND engaged (warm-up %.1fs); switch "
                      "crossfade %.2fs; hold-on-drop %s.",
                      cli.dual_model_path.c_str(), smpl_ori_mode_.c_str(),
                      cli.smpl_intent_bind ? "bound at" : "connected to",
                      cli.smpl_intent_host.c_str(), cli.smpl_intent_port,
                      cli.wb_stale_s, cli.wb_warmup_s, cli.wb_switch_ramp_s,
                      cli.wb_hold ? "ON" : "off");
        } else {
        if (cli.token_model_path.empty()) {
          throw std::runtime_error(
              "--whole-body-teleop requires --token-model PATH (the "
              "token-input ONNX for the overlay).");
        }
        token_actor_ = std::make_unique<OnnxActor>(cli.token_model_path,
                                                   cli.intra_op_threads);
        const std::string tkind =
            token_actor_->GetCustomMetadata("graph_kind");
        if (tkind != "any2any_token_deploy") {
          throw std::runtime_error(
              "--token-model must be a token-input graph (metadata "
              "graph_kind=any2any_token_deploy, exported by "
              "frozen_core_token_onnx_export.py); got graph_kind='" + tkind +
              "' from " + cli.token_model_path);
        }
        token_source_owned_ = ZmqPoseInputSource::Connect(
            cli.token_zmq_host, cli.token_zmq_port, "pose");
        token_source_ = token_source_owned_.get();
        RCLCPP_WARN(node_->get_logger(),
                    "WHOLE-BODY OVERLAY armed: token source tcp://%s:%d; "
                    "pad/kplanner PRIMARY. Overlay drives only while the "
                    "stream is fresh (<0.5 s) AND engaged; otherwise the "
                    "pad has authority. Transitions blend via the "
                    "soft-start ramp.",
                    cli.token_zmq_host.c_str(), cli.token_zmq_port);
        }  // legacy token path
      }
    }
    {
      // Provenance on the wire (operator rule: ship proof = the Loaded
      // ONNX line, never an md5 alone). Stamped by the native exporters;
      // gear_sonic/scripts/onnx_provenance.py reads the same fields.
      std::string ck = onnx_actor_->GetCustomMetadata("checkpoint");
      if (ck.empty()) ck = onnx_actor_->GetCustomMetadata("source_checkpoint");
      if (ck.empty()) ck = "UNSTAMPED";
      std::string fp = onnx_actor_->GetCustomMetadata("codec_fingerprint");
      if (fp.empty()) fp = "-";
      std::string kind = onnx_actor_->GetCustomMetadata("graph_kind");
      if (kind.empty()) kind = "pose-graph";
      RCLCPP_INFO(node_->get_logger(),
                  "Loaded ONNX: %s  (input='%s' [1, %ld])  kind=%s  from %s  fp %s",
                  onnx_actor_->model_path().c_str(),
                  onnx_actor_->input_name().c_str(),
                  static_cast<long>(onnx_actor_->expected_obs_dim()),
                  kind.c_str(), ck.c_str(), fp.c_str());
    }

    // Synthesise the per-DOF max_target_dev array once from the CLI
    // (global + per-group). Done after CLI parsing but before any
    // OnControl tick can fire; the array is read-only thereafter.
    max_target_dev_per_dof_ = BuildMaxTargetDevPerDof(MaxTargetDevSpec{
        cli_.max_target_dev,
        cli_.max_target_dev_leg,
        cli_.max_target_dev_waist,
        cli_.max_target_dev_arm,
        cli_.max_target_dev_head,
        cli_.max_target_dev_waist_pitch});

    // Synthesise the per-DOF effective PD by multiplying the constexpr
    // trained kps[] / kds[] by the global+per-family scales. Done once
    // here so the OnControl hot path is unaffected.
    // Family aliases compose MULTIPLICATIVELY with their split-subgroup
    // knobs, so ``--kp-scale-ankle 1.5`` alone still scales BOTH ankle
    // subgroups by 1.5x (backward compat with the pre-split single-knob
    // YAMLs); same for ``--kp-scale-waist``. Setting both an alias AND
    // a subgroup value multiplies them, which is almost certainly an
    // operator typo but we don't reject it -- the SAFETY warn log
    // surfaces the final effective scale per subgroup so the operator
    // can verify what landed.
    // Waist subgroup composition: the legacy ``waist_pr`` alias is
    // kept as a multiplier on BOTH pitch and roll for backward compat
    // with pre-2026-05-30 walking_recovery.yaml / expressive.yaml.
    // Effective scale on a waist_pitch joint becomes
    //   waist (alias) * waist_pr (alias) * waist_pitch (split knob)
    // and similarly for roll. All three default to 1.0, so the
    // alias-only path (e.g. ``kd_scale_waist_pr: 5.51``) reproduces
    // the pre-split behaviour bit-for-bit.
    const auto kp_scales = BuildPdScalesPerDof(PdScaleSpec{
        cli_.kp_scale,
        cli_.kp_scale_hip,      cli_.kp_scale_knee,
        cli_.kp_scale_ankle * cli_.kp_scale_ankle_pitch,
        cli_.kp_scale_ankle * cli_.kp_scale_ankle_roll,
        cli_.kp_scale_waist * cli_.kp_scale_waist_yaw,
        cli_.kp_scale_waist * cli_.kp_scale_waist_pr * cli_.kp_scale_waist_pitch,
        cli_.kp_scale_waist * cli_.kp_scale_waist_pr * cli_.kp_scale_waist_roll,
        cli_.kp_scale_shoulder, cli_.kp_scale_elbow,
        cli_.kp_scale_wrist,    cli_.kp_scale_head});
    const auto kd_scales = BuildPdScalesPerDof(PdScaleSpec{
        cli_.kd_scale,
        cli_.kd_scale_hip,      cli_.kd_scale_knee,
        cli_.kd_scale_ankle * cli_.kd_scale_ankle_pitch,
        cli_.kd_scale_ankle * cli_.kd_scale_ankle_roll,
        cli_.kd_scale_waist * cli_.kd_scale_waist_yaw,
        cli_.kd_scale_waist * cli_.kd_scale_waist_pr * cli_.kd_scale_waist_pitch,
        cli_.kd_scale_waist * cli_.kd_scale_waist_pr * cli_.kd_scale_waist_roll,
        cli_.kd_scale_shoulder, cli_.kd_scale_elbow,
        cli_.kd_scale_wrist,    cli_.kd_scale_head});
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      kps_scaled_[i] = kps[i] * kp_scales[i];
      kds_scaled_[i] = kds[i] * kd_scales[i];
    }

    // Loud-and-proud announcement of the safety knobs that materially affect
    // worst-case actuation. Operator should see these in the deploy log
    // before saying "go", so a missing --max-target-dev on a powered run is
    // visible at a glance instead of buried in the help text. Per-group
    // overrides are surfaced even when the global is disabled, so a
    // partial-coverage configuration ("arm=1.50, leg=disabled") is hard
    // to overlook.
    constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
    auto fmt_clamp = [&](double rad) {
      char buf[64];
      if (rad > 0.0) {
        std::snprintf(buf, sizeof(buf), "%.3f rad (%.1f deg)", rad, rad * kRadToDeg);
      } else {
        std::snprintf(buf, sizeof(buf), "DISABLED");
      }
      return std::string(buf);
    };
    auto group_effective = [&](double group_val) {
      return group_val > 0.0 ? group_val : cli_.max_target_dev;
    };
    if (cli_.max_target_dev > 0.0
        || cli_.max_target_dev_leg > 0.0
        || cli_.max_target_dev_waist > 0.0
        || cli_.max_target_dev_arm > 0.0
        || cli_.max_target_dev_head > 0.0
        || cli_.max_target_dev_waist_pitch > 0.0) {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: per-joint target clamp ENABLED. Effective per-group "
                  "|target - default| limits: leg=%s, waist=%s (pitch=%s), arm=%s, head=%s "
                  "(global default --max-target-dev=%s; per-group overrides win when > 0)",
                  fmt_clamp(group_effective(cli_.max_target_dev_leg)).c_str(),
                  fmt_clamp(group_effective(cli_.max_target_dev_waist)).c_str(),
                  fmt_clamp(cli_.max_target_dev_waist_pitch > 0.0
                            ? cli_.max_target_dev_waist_pitch
                            : group_effective(cli_.max_target_dev_waist)).c_str(),
                  fmt_clamp(group_effective(cli_.max_target_dev_arm)).c_str(),
                  fmt_clamp(group_effective(cli_.max_target_dev_head)).c_str(),
                  fmt_clamp(cli_.max_target_dev).c_str());
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: per-joint target clamp DISABLED "
                  "(no --max-target-dev{,-leg,-waist,-arm,-head} set). "
                  "Policy can drive any joint to any value the ONNX session emits.");
    }

    // Surface any non-unity PD trim so the operator can verify ankle=1.5,
    // waist=1.5, etc. landed before saying "go". Suppressed entirely when
    // every scale is exactly 1.0 (the default; unscaled trained PD).
    auto any_non_unity = [](std::initializer_list<double> xs) {
      for (double x : xs) if (x != 1.0) return true;
      return false;
    };
    const bool kp_trimmed = any_non_unity({
        cli_.kp_scale, cli_.kp_scale_hip, cli_.kp_scale_knee,
        cli_.kp_scale_ankle, cli_.kp_scale_ankle_pitch, cli_.kp_scale_ankle_roll,
        cli_.kp_scale_waist, cli_.kp_scale_waist_yaw, cli_.kp_scale_waist_pr,
        cli_.kp_scale_waist_pitch, cli_.kp_scale_waist_roll,
        cli_.kp_scale_shoulder, cli_.kp_scale_elbow,
        cli_.kp_scale_wrist, cli_.kp_scale_head});
    const bool kd_trimmed = any_non_unity({
        cli_.kd_scale, cli_.kd_scale_hip, cli_.kd_scale_knee,
        cli_.kd_scale_ankle, cli_.kd_scale_ankle_pitch, cli_.kd_scale_ankle_roll,
        cli_.kd_scale_waist, cli_.kd_scale_waist_yaw, cli_.kd_scale_waist_pr,
        cli_.kd_scale_waist_pitch, cli_.kd_scale_waist_roll,
        cli_.kd_scale_shoulder, cli_.kd_scale_elbow,
        cli_.kd_scale_wrist, cli_.kd_scale_head});
    if (kp_trimmed || kd_trimmed) {
      // Print effective per-subgroup scales (with alias folded in) so the
      // operator doesn't have to multiply the alias by the subgroup knob
      // in their head to know what landed. Also dump sample joints so
      // the absolute Nm/rad value is visible at a glance for each
      // wobble-axis joint family.
      const double kp_eff_ankle_pitch = cli_.kp_scale_ankle * cli_.kp_scale_ankle_pitch;
      const double kp_eff_ankle_roll  = cli_.kp_scale_ankle * cli_.kp_scale_ankle_roll;
      const double kd_eff_ankle_pitch = cli_.kd_scale_ankle * cli_.kd_scale_ankle_pitch;
      const double kd_eff_ankle_roll  = cli_.kd_scale_ankle * cli_.kd_scale_ankle_roll;
      const double kp_eff_waist_yaw   = cli_.kp_scale_waist * cli_.kp_scale_waist_yaw;
      // waist_pitch and waist_roll effective scales fold in both the
      // legacy ``waist`` and ``waist_pr`` aliases as well as the new
      // per-axis knobs, so what gets logged here is exactly what
      // BuildPdScalesPerDof above feeds into BuildPdScalesPerDof.
      const double kp_eff_waist_pitch = cli_.kp_scale_waist * cli_.kp_scale_waist_pr * cli_.kp_scale_waist_pitch;
      const double kp_eff_waist_roll  = cli_.kp_scale_waist * cli_.kp_scale_waist_pr * cli_.kp_scale_waist_roll;
      const double kd_eff_waist_yaw   = cli_.kd_scale_waist * cli_.kd_scale_waist_yaw;
      const double kd_eff_waist_pitch = cli_.kd_scale_waist * cli_.kd_scale_waist_pr * cli_.kd_scale_waist_pitch;
      const double kd_eff_waist_roll  = cli_.kd_scale_waist * cli_.kd_scale_waist_pr * cli_.kd_scale_waist_roll;
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: deployment-time PD trim ENABLED. Effective gains = "
                  "trained kps[]/kds[] * (global * family). KP scales: "
                  "global=%.3f hip=%.3f knee=%.3f "
                  "ankle_pitch=%.3f ankle_roll=%.3f "
                  "waist_yaw=%.3f waist_pitch=%.3f waist_roll=%.3f "
                  "shoulder=%.3f elbow=%.3f wrist=%.3f head=%.3f. "
                  "KD scales: global=%.3f hip=%.3f knee=%.3f "
                  "ankle_pitch=%.3f ankle_roll=%.3f "
                  "waist_yaw=%.3f waist_pitch=%.3f waist_roll=%.3f "
                  "shoulder=%.3f elbow=%.3f wrist=%.3f head=%.3f. "
                  "Sample effective gains: "
                  "ankle_pitch kp=%.3f kd=%.3f, ankle_roll kp=%.3f kd=%.3f, "
                  "waist_yaw kp=%.3f kd=%.3f, waist_pitch kp=%.3f kd=%.3f, "
                  "waist_roll kp=%.3f kd=%.3f.",
                  cli_.kp_scale, cli_.kp_scale_hip, cli_.kp_scale_knee,
                  kp_eff_ankle_pitch, kp_eff_ankle_roll,
                  kp_eff_waist_yaw, kp_eff_waist_pitch, kp_eff_waist_roll,
                  cli_.kp_scale_shoulder, cli_.kp_scale_elbow,
                  cli_.kp_scale_wrist, cli_.kp_scale_head,
                  cli_.kd_scale, cli_.kd_scale_hip, cli_.kd_scale_knee,
                  kd_eff_ankle_pitch, kd_eff_ankle_roll,
                  kd_eff_waist_yaw, kd_eff_waist_pitch, kd_eff_waist_roll,
                  cli_.kd_scale_shoulder, cli_.kd_scale_elbow,
                  cli_.kd_scale_wrist, cli_.kd_scale_head,
                  kps_scaled_[4],  kds_scaled_[4],   // left_ankle_pitch
                  kps_scaled_[5],  kds_scaled_[5],   // left_ankle_roll
                  kps_scaled_[12], kds_scaled_[12],  // waist_yaw
                  kps_scaled_[13], kds_scaled_[13], // waist_pitch
                  kps_scaled_[14], kds_scaled_[14]); // waist_roll
    } else {
      RCLCPP_INFO(node_->get_logger(),
                  "SAFETY: deployment-time PD trim DISABLED (all scales = 1.0). "
                  "Using trained kps[]/kds[] from policy_parameters.hpp as-is. "
                  "Note: IsaacLab's implicit PD makes the same numerical KP "
                  "behave ~1.3-1.5x stiffer at training than at deploy; if you "
                  "see torso wobble on nudge, try the MC-matched values shipped "
                  "in configs/real_deploy_tuning/expressive.yaml.");
    }

    if (cli_.action_clip > 0.0) {
      if (recorder_.enabled()) {
        RCLCPP_WARN(node_->get_logger(),
                    "BLACK BOX ARMED: %.0f s pre-trigger ring (fast %d Hz state, "
                    "slow %d Hz command chain). Dumps to %s on e-stop OR SAFE_HOLD.",
                    cli_.flight_seconds,
                    agi_x2::FlightRecorder::kFastHz,
                    agi_x2::FlightRecorder::kSlowHz,
                    (!cli_.flight_dir.empty() ? cli_.flight_dir
                     : (!cli_.log_dir.empty() ? cli_.log_dir
                        : std::string("/tmp"))).c_str());
      } else {
        RCLCPP_INFO(node_->get_logger(),
                    "black box OFF (--flight-recorder-seconds %.1f, dir '%s')",
                    cli_.flight_seconds,
                    (!cli_.flight_dir.empty() ? cli_.flight_dir
                     : cli_.log_dir).c_str());
      }
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: raw action clip ENABLED at |action_il| <= %.3f "
                  "(matches training action_clip_value=20.0 from base_env.yaml)",
                  cli_.action_clip);
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: raw action clip DISABLED "
                  "(--action-clip <= 0). Deploy will diverge from training "
                  "wrapper behavior; only use for parity tests.");
    }

    if (cli_.return_seconds > 0.0) {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: soft-exit ramp ENABLED (--return-seconds %.2fs). "
                  "When --max-duration trips, joints will be lerped back to "
                  "default_angles before shutdown so MC handoff doesn't fault.",
                  cli_.return_seconds);
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: soft-exit ramp DISABLED (--return-seconds <= 0). "
                  "Deploy will shut down immediately on --max-duration; if the "
                  "policy left joints far from default_angles, the next MC "
                  "start_app POST may snap them back and trip a red fault.");
    }

    // ─── End-of-run smooth handoff: load MC's STAND_DEFAULT pose ────────
    // If --stand-default-pose was given, we use the captured pose+kp+kd
    // for both RAMP_OUT (lerp target) and HOLD_FOR_MC (static publish).
    // Otherwise we fall back to default_angles + deploy-mode kp/kd, which
    // mismatches MC's pose by up to 34 deg at the elbows -- the operator
    // sees an audible pop on takeover. Print a loud diff summary so the
    // operator notices when the YAML is missing.
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      stand_pose_target_[i]    = default_angles[i];
      stand_pose_stiffness_[i] = kps_scaled_[i];
      stand_pose_damping_[i]   = kds_scaled_[i];
    }
    if (!cli_.stand_pose_path.empty()) {
      try {
        const auto sp = LoadStandPose(cli_.stand_pose_path);
        double max_diff_rad = 0.0;
        std::size_t worst_idx = 0;
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          stand_pose_target_[i]    = sp.position[i];
          stand_pose_stiffness_[i] = sp.stiffness[i];
          stand_pose_damping_[i]   = sp.damping[i];
          const double d = std::abs(sp.position[i] - default_angles[i]);
          if (d > max_diff_rad) { max_diff_rad = d; worst_idx = i; }
        }
        RCLCPP_WARN(node_->get_logger(),
                    "HANDOFF: loaded MC STAND_DEFAULT pose from '%s' "
                    "(31 joints). Worst delta vs default_angles: %.3f rad "
                    "(%.1f deg) at '%s'. RAMP_OUT and HOLD_FOR_MC will "
                    "target this pose so MC takeover is step-free.",
                    cli_.stand_pose_path.c_str(),
                    max_diff_rad,
                    max_diff_rad * 180.0 / 3.14159265358979323846,
                    mujoco_joint_names[worst_idx]);
      } catch (const std::exception& e) {
        RCLCPP_FATAL(node_->get_logger(),
                     "HANDOFF: failed to load --stand-default-pose '%s': %s. "
                     "Aborting (refusing to start with an unknown handoff "
                     "target). Pass --stand-default-pose '' to fall back to "
                     "default_angles, or fix the YAML.",
                     cli_.stand_pose_path.c_str(), e.what());
        throw;
      }
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "HANDOFF: --stand-default-pose not provided; RAMP_OUT and "
                  "HOLD_FOR_MC will target default_angles. MC takeover may "
                  "snap up to 34 deg at the elbows. Pass "
                  "--stand-default-pose configs/x2_stand_default_pose.yaml "
                  "to eliminate the snap.");
    }

    if (cli_.hold_for_mc_timeout_s > 0.0) {
      if (cli_.hold_for_mc_exit_sentinel.empty()) {
        RCLCPP_WARN(node_->get_logger(),
                    "HANDOFF: HOLD_FOR_MC enabled (timeout %.1fs). After "
                    "RAMP_OUT, deploy will keep publishing MC's STAND_DEFAULT "
                    "pose until MC's first joint command arrives on the bus, "
                    "then exit cleanly (legacy fast-exit path).",
                    cli_.hold_for_mc_timeout_s);
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "HANDOFF: HOLD_FOR_MC enabled (timeout %.1fs). After "
                    "RAMP_OUT, deploy will keep publishing MC's STAND_DEFAULT "
                    "pose until bash touches the exit-sentinel '%s' (set after "
                    "MC has escalated all the way to STAND_DEFAULT). No zero-"
                    "torque window during MC boot's PASSIVE -> JOINT -> STAND.",
                    cli_.hold_for_mc_timeout_s,
                    cli_.hold_for_mc_exit_sentinel.c_str());
      }
      InitMcTakeoverDetectors();
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "HANDOFF: HOLD_FOR_MC disabled (--hold-for-mc-timeout-s "
                  "<= 0). RAMP_OUT will exit the process; the bus will be "
                  "silent until bash POSTs start_app.");
    }
    if (!cli_.hold_for_mc_sentinel.empty()) {
      RCLCPP_INFO(node_->get_logger(),
                  "HANDOFF: HOLD_FOR_MC sentinel = '%s' "
                  "(touched on entering HOLD_FOR_MC).",
                  cli_.hold_for_mc_sentinel.c_str());
    }
    if (!cli_.soft_shutdown_trigger_sentinel.empty()) {
      RCLCPP_WARN(node_->get_logger(),
                  "HANDOFF: soft-shutdown trigger ENABLED -- polling '%s' "
                  "in CONTROL state at 50 Hz. When bash touches this file "
                  "(e.g. on Ctrl-C), deploy transitions to RAMP_OUT and "
                  "follows the full RAMP_OUT -> HOLD_FOR_MC -> exit-"
                  "sentinel chain. Robot stays under torque across the "
                  "MC restart; no zero-torque drop.",
                  cli_.soft_shutdown_trigger_sentinel.c_str());
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "HANDOFF: soft-shutdown trigger DISABLED. Ctrl-C will "
                  "exit immediately via rclcpp default SIGINT handler; "
                  "bus will be silent for ~1-2 s while MC boots back "
                  "through PASSIVE_DEFAULT (zero torque, robot drops "
                  "under gravity). Pass --soft-shutdown-trigger-sentinel "
                  "PATH to opt in to the graceful path.");
    }

    // STANDBY support. If --start-trigger-sentinel is set, boot into
    // STANDBY (writer suppressed, no joint commands published) and wait
    // for bash to touch the file before advancing to INIT. This lets
    // bash launch the binary AHEAD of the safety gate to overlap colcon
    // build / ONNX load / DDS discovery with the operator's "Y" decision.
    if (!cli_.start_trigger_sentinel.empty()) {
      state_.store(State::STANDBY);
      if (cli_.no_publish) {
        RCLCPP_WARN(node_->get_logger(),
                    "OBSERVE-ONLY: --no-publish is SET. The 500 Hz writer "
                    "will NEVER publish, in ANY state (including SAFE_HOLD "
                    "and RAMP_OUT). The policy runs and all CSVs are written, "
                    "but nothing reaches the motor bus. MC MUST retain the "
                    "robot for this entire run -- if MC is stopped, NOTHING "
                    "will be commanding the joints.");
      }
      RCLCPP_WARN(node_->get_logger(),
                  "STANDBY: --start-trigger-sentinel='%s' provided; deploy "
                  "is alive but the 500 Hz writer is GATED OFF. Will "
                  "advance STANDBY -> INIT when this file appears.",
                  cli_.start_trigger_sentinel.c_str());
      // Best-effort cleanup of any stale trigger left over from a prior
      // crash. If the file is owned by another user we may fail; that's
      // harmless -- bash recreates it fresh.
      std::remove(cli_.start_trigger_sentinel.c_str());
    }

    // Compute the per-DOF EMA coefficient array now so OnControl can apply
    // it without re-deriving every tick. dt is fixed at 1/50 s (the
    // OnControl rate); alpha = 1 - exp(-2*pi*hz*dt) is the standard
    // discrete first-order low-pass coefficient. Per-group overrides
    // (--target-lpf-hz-{leg,waist,arm,head}) follow the max_target_dev
    // semantics: <0 inherit global, ==0 disabled on that group, >0 use
    // explicit cutoff. See BuildTargetLpfAlphaPerDof for the full table.
    {
      constexpr double kOnControlDt = 1.0 / 50.0;
      const TargetLpfSpec spec{
          cli_.target_lpf_hz,        // global
          cli_.target_lpf_hz_leg,    // leg override (<0 = inherit)
          cli_.target_lpf_hz_waist,  // waist override
          cli_.target_lpf_hz_arm,    // arm override
          cli_.target_lpf_hz_head,   // head override
          kOnControlDt,
      };
      target_lpf_alpha_per_dof_ = BuildTargetLpfAlphaPerDof(spec);

      // Cache the "anything active" flag so the OnControl hot loop can
      // skip the entire EMA block when every DOF is in passthrough mode.
      target_lpf_any_active_ = false;
      for (double a : target_lpf_alpha_per_dof_) {
        if (a > 0.0) {
          target_lpf_any_active_ = true;
          break;
        }
      }

      // Pretty-print the effective Hz per group for the warm-up log, so an
      // operator who's reading the deploy logs at startup can sanity-check
      // what they shipped without having to re-derive alpha back into Hz.
      auto group_eff_hz = [&](double override_hz) {
        return override_hz < 0.0 ? std::max(spec.global, 0.0) : std::max(override_hz, 0.0);
      };
      const double hz_leg   = group_eff_hz(cli_.target_lpf_hz_leg);
      const double hz_waist = group_eff_hz(cli_.target_lpf_hz_waist);
      const double hz_arm   = group_eff_hz(cli_.target_lpf_hz_arm);
      const double hz_head  = group_eff_hz(cli_.target_lpf_hz_head);

      if (target_lpf_any_active_) {
        RCLCPP_WARN(node_->get_logger(),
                    "REAL-DEPLOY: output target LPF ACTIVE on at least one group. "
                    "Effective cutoffs (Hz) -- leg=%.2f waist=%.2f arm=%.2f head=%.2f "
                    "(global=%.2f). Bypassed in RAMP_OUT/SAFE_HOLD. Sim parity "
                    "(eval_x2_mujoco.py) preserved -- this filter is strictly "
                    "downstream of the policy and never affects --obs-dump.",
                    hz_leg, hz_waist, hz_arm, hz_head,
                    std::max(cli_.target_lpf_hz, 0.0));
      } else {
        RCLCPP_INFO(node_->get_logger(),
                    "REAL-DEPLOY: output target LPF DISABLED on every group "
                    "(global=%.2f, leg=%.2f, waist=%.2f, arm=%.2f, head=%.2f Hz).",
                    cli_.target_lpf_hz, cli_.target_lpf_hz_leg,
                    cli_.target_lpf_hz_waist, cli_.target_lpf_hz_arm,
                    cli_.target_lpf_hz_head);
      }
    }

    // Initial safe command: PASSIVE (kp=0, kd=0) until any state arrives.
    //
    // The 500 Hz writer starts publishing as soon as the timer fires, which
    // is *before* INIT clears (i.e. before any joint state has been
    // received from HAL). If we latched full kp/kd here against
    // default_angles, the writer would yank every joint toward
    // default_angles for the entire pre-INIT window -- which on the real
    // robot fights MC's standing pose, and in sim mode (with the bridge's
    // standby PD already holding a non-default pose, e.g. --init-pose=
    // gantry_hang) tips the body over before the policy ever gets a tick
    // (verified: 95 deg tilt in 3 s of autostart). Publishing kp=kd=0 at
    // startup means the writer applies zero torque, leaving HAL / MC / the
    // sim bridge fully in charge until INIT->WAIT_FOR_CONTROL.
    //
    // The actual safe-hold pose is latched in OnControl() at the
    // INIT->WAIT_FOR_CONTROL transition, using the *current* observed
    // joint pose (rs.joint_pos_mj). That way the deploy holds whatever
    // pose the operator/MC/bridge had at the moment of handoff, so
    // WAIT_FOR_CONTROL is genuinely safe regardless of how far the start
    // pose is from DEFAULT_DOF.
    {
      std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
      for (std::size_t i = 0; i < NUM_DOFS; ++i) {
        latest_cmd_.target_pos_mj[i] = default_angles[i];
        latest_cmd_.stiffness_mj[i]  = 0.0;
        latest_cmd_.damping_mj[i]    = 0.0;
      }
      latest_cmd_.dry_run    = cli_.dry_run;
      latest_cmd_.tilt_trip  = false;
      latest_cmd_.ramp_alpha = 0.0;
      latest_cmd_.reason     = "init_passive";
    }

    // Timers. Both attach to whatever executor spins this node.
    control_timer_ = node_->create_wall_timer(
        std::chrono::milliseconds(20),  // 50 Hz
        std::bind(&X2Deploy::OnControl, this));
    writer_timer_ = node_->create_wall_timer(
        std::chrono::milliseconds(2),   // 500 Hz
        std::bind(&X2Deploy::OnWriter, this));

    // Optional autostart watchdog: flips WAIT->CONTROL after N seconds.
    if (cli_.autostart_seconds >= 0.0) {
      autostart_target_s_ = SteadyNow() + cli_.autostart_seconds;
    }

    // ─── Pose-ref starvation watchdog (split-topology safety) ──────────
    // Active only on the ZMQ input path. Logging is loud-and-proud at
    // startup so the operator can confirm the thresholds before saying
    // "go". The actual per-tick Update() lives in OnControl below, so
    // there is nothing else to wire here.
    pose_ref_watchdog_active_ =
        (cli_.input_type == "zmq")
        && (cli_.pose_ref_stale_s > 0.0)
        && (!cli_.disable_pose_ref_watchdog);
    if (cli_.input_type == "zmq") {
      if (pose_ref_watchdog_active_) {
        RCLCPP_WARN(node_->get_logger(),
                    "SAFETY: pose-ref starvation watchdog ENABLED. CONTROL "
                    "trips into SAFE_IDLE when pose-ref age >= %.3f s; "
                    "exit requires fresh frames for >= %.3f s AND an "
                    "operator resume chord. SAFE_IDLE holds default_angles "
                    "with 4x kd (no policy inference).",
                    cli_.pose_ref_stale_s, cli_.pose_ref_min_fresh_s);
      } else if (cli_.disable_pose_ref_watchdog) {
        RCLCPP_WARN(node_->get_logger(),
                    "SAFETY: pose-ref starvation watchdog DISABLED "
                    "(--disable-pose-ref-watchdog). Legacy 'hold last good "
                    "frame' behaviour active; freeze-while-leaning failure "
                    "mode is possible if the operator wire stalls.");
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "SAFETY: pose-ref starvation watchdog DISABLED "
                    "(--pose-ref-stale-s <= 0). Set --pose-ref-stale-s 0.5 "
                    "for split-topology production.");
      }
    } else {
      // motion_file path; watchdog doesn't apply.
      RCLCPP_INFO(node_->get_logger(),
                  "SAFETY: pose-ref starvation watchdog NOT applicable "
                  "(--input-type=motion_file). Watchdog activates only "
                  "on the ZMQ input path.");
    }

    // ─── Operator resume chord SUB (split-topology safety) ─────────────
    // Bound only when the watchdog is active. The subscriber listens for
    // a single multipart message on tcp://<zmq_resume_host>:<zmq_resume_port>
    // topic <zmq_resume_topic>; the receipt itself (not the payload value)
    // is what counts. Connect failures are warned-and-continued so a
    // missing operator stack at deploy startup doesn't block the whole
    // CONTROL path -- the watchdog will simply trip into SAFE_IDLE on
    // the first starvation, and re-bind happens on every deploy restart.
    if (pose_ref_watchdog_active_) {
      try {
        zmq_resume_sub_ = ZmqResumeSubscriber::Connect(
            cli_.zmq_resume_host, cli_.zmq_resume_port, cli_.zmq_resume_topic);
        RCLCPP_WARN(node_->get_logger(),
                    "SAFETY: operator resume SUB bound to %s topic='%s'. "
                    "SAFE_IDLE exit requires a resume frame within the "
                    "last 0.5 s.",
                    zmq_resume_sub_->Endpoint().c_str(),
                    zmq_resume_sub_->Topic().c_str());
      } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(),
                     "SAFETY: failed to bind operator resume SUB on "
                     "tcp://%s:%d topic='%s': %s. SAFE_IDLE will be reachable "
                     "but recovery requires deploy restart until this is "
                     "fixed.",
                     cli_.zmq_resume_host.c_str(), cli_.zmq_resume_port,
                     cli_.zmq_resume_topic.c_str(), e.what());
      }
    }

    // ─── MC mode poller (Phase 5b observability) ───────────────────────
    // Async service client + a wall timer that fires every mc_mode_poll_s
    // seconds. Each tick either: (a) starts a new async request if the
    // previous one finished or never started, OR (b) reaps the pending
    // future when it completes. mc_action_mode_ is the latest known
    // McAction.value (= int enum, see McAction.msg). Published on
    // x2_debug as ``mc_action_mode``. Pure observability -- the deploy
    // never reads mc_action_mode_ to make control decisions.
    if (cli_.mc_mode_poll_s > 0.0) {
      mc_get_action_client_ =
          node_->create_client<aimdk_msgs::srv::GetMcAction>(
              cli_.mc_mode_service);
      const auto period_ms = static_cast<int>(
          std::lround(cli_.mc_mode_poll_s * 1000.0));
      mc_mode_poll_timer_ = node_->create_wall_timer(
          std::chrono::milliseconds(std::max(period_ms, 100)),
          std::bind(&X2Deploy::OnMcModePoll, this));
      RCLCPP_WARN(node_->get_logger(),
                  "OBS: MC mode poller ENABLED at %.2f s on '%s'. "
                  "Latest value published on x2_debug as mc_action_mode "
                  "(-1 = unknown / service not yet seen).",
                  cli_.mc_mode_poll_s, cli_.mc_mode_service.c_str());
    } else {
      RCLCPP_INFO(node_->get_logger(),
                  "OBS: MC mode poller DISABLED (--mc-mode-poll-s <= 0). "
                  "x2_debug mc_action_mode field will be -1 throughout.");
    }
  }

  State state() const { return state_.load(); }

  /// Operator-triggered transition (stdin "go" or future service call).
  /// Safe to call from any thread; takes effect on the next OnControl tick.
  void RequestGo()
  {
    if (state_.load() != State::WAIT_FOR_CONTROL) return;
    autostart_target_s_ = SteadyNow();
    RCLCPP_INFO(node_->get_logger(),
                "Operator GO received; transitioning on next tick.");
  }

 private:
  static double SteadyNow()
  {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  // ----- 50 Hz control loop -------------------------------------------------
  void OnControl()
  {
    const double now = SteadyNow();
    const State  cur = state_.load();

    RobotState rs{};
    const bool fresh = aimdk_io_->SnapshotState(rs);

    // ---- CONTROL-phase staleness guard (2026-08-04, G1-parity audit #1;
    // RETUNED after the 2026-08-04 live incident). ``fresh`` above is a
    // have-I-ever-seen-it flag, NOT an age check. Original port used a
    // 150 ms window — 3.3x tighter than G1's LOW_STATE_ABSENT (500 ms) —
    // and tripped ~13 s into the first real-robot session: SAFE_HOLD ->
    // stage-2 damping -> standing robot slumped with a violent gear whir,
    // operator pulled the battery. Real aimdk streams can gap past 150 ms;
    // sim never does, so preflight could not catch it. Now: G1-parity
    // 500 ms AND 3 consecutive stale ticks (debounce: one hiccup at the
    // 50 Hz loop never trips), with a per-stream age report on every
    // strike so the offending stream is named in the log.
    if (cur == State::CONTROL) {
      if (!aimdk_io_->AllStateFresh(0.5)) {
        ++stale_state_ticks_;
        RCLCPP_WARN(node_->get_logger(),
                    "robot state stale >500 ms in CONTROL (strike %d/3): %s",
                    stale_state_ticks_,
                    aimdk_io_->StateAgeReport().c_str());
        if (stale_state_ticks_ >= 3) {
          RCLCPP_FATAL(node_->get_logger(),
                       "robot state stale 3 consecutive ticks — E-STOP -> PURE "
                       "DAMPING (%s)", aimdk_io_->StateAgeReport().c_str());
          // STATE-BLIND => DAMP, do not hold. (2026-08-25.)
          //
          // This path used to store(SAFE_HOLD) and return, latching NOTHING --
          // so the 500 Hz writer kept republishing whatever command was last
          // set. On 2026-08-25 that was a RUNAWAY target: with observations
          // frozen the policy had integrated open-loop on its own last_action
          // and driven the waist pitch/roll command to 129 deg / 162 deg,
          // saturating the 0.45 rad clamp. Holding a diverged pose at full kp
          // is the worst of both worlds -- rigid AND wrong.
          //
          // Damping (not holding) is correct here: if the state is stale we do
          // not know where the robot IS, so commanding any position is a guess.
          // kVendorDampKd is the vendor per-joint profile with knee
          // fold-shaping (knee 4.0 vs hip 9.0), so the body folds instead of
          // toppling stiff -- the same profile the operator e-stop uses and
          // which is robot-verified.
          //
          // The 2026-08-04 slump ("violent gear whir, operator pulled the
          // battery") was a UNIFORM stage-2 damping that predates this profile
          // and the e-stop system. It is not an argument against this.
          SafeCommand dc;
          BuildPureDampingCommand(dc, default_angles);
          dc.dry_run    = cli_.dry_run;
          dc.tilt_trip  = true;
          dc.ramp_alpha = 0.0;
          dc.reason     = "robot state stale -> PURE-DAMPING";
          if (cli_.dry_run) {
            for (std::size_t i = 0; i < NUM_DOFS; ++i) {
              dc.stiffness_mj[i] = 0.0;
              dc.damping_mj[i]   = 0.0;
            }
          }
          {
            std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
            latest_cmd_ = dc;
          }
          latest_cmd_ready_.store(true, std::memory_order_release);
          safe_hold_damped_ = true;   // SAFE_HOLD must not re-latch over this
          recorder_.Trigger("stale_state");
          state_.store(State::SAFE_HOLD);
          return;
        }
      } else {
        stale_state_ticks_ = 0;
      }
    }

    // ---- Operator e-stop from the pose wire -- POLLED IN EVERY STATE.
    //
    // MOVED HERE 2026-08-23. It previously lived after the state switch, in
    // the fall-through that only State::CONTROL reaches: every other case
    // returns before it. That meant the operator e-stop was silently ignored
    // in STANDBY, INIT, WAIT_FOR_CONTROL, RAMP_OUT, SAFE_IDLE and
    // HOLD_FOR_MC -- the deploy would hold its last commanded pose and not
    // drop to damping, which is exactly the "e-stop pressed, back stayed
    // bent" behaviour reported at 04:39 on 2026-08-23.
    //
    // SCOPE, STATED HONESTLY: this does NOT fix that specific incident. There
    // the control thread had wedged after one tick, so no polling site of any
    // kind was executing, and the 500 Hz writer latched the stale command.
    // Covering a wedged control thread needs the WRITER to consult the e-stop
    // independently -- tracked separately. This change closes the adjacent
    // and simpler hole: a LIVE control thread sitting in a non-CONTROL state
    // must still honour the operator.
    //
    // Cost is one atomic-ish read per tick against the ZMQ source, in a 50 Hz
    // loop. The guard order is unchanged (skip if the watchdog already
    // tripped), so a tripped run cannot re-trip and re-log.
      // GUARD WIDENED 2026-08-24: was `!watchdog_.Tripped()`, which meant a run
      // that had ALREADY tripped (e.g. tilt watchdog -> SAFE_HOLD) ignored every
      // subsequent operator e-stop. A stage-1 trip does NOT engage pure damping,
      // so an operator pressing e-stop in that state expected everything to go
      // limp and got nothing. Gate on ForceStage2() instead: ForceTripEstop early-
      // returns once stage 2 is set, so this cannot re-log, but it CAN escalate an
      // existing trip to pure damping.
      if (!watchdog_.ForceStage2()
        && zmq_pose_source_ != nullptr
        && zmq_pose_source_->EstopRequested()) {
      RCLCPP_FATAL(node_->get_logger(),
                   "operator E-STOP on pose wire (state=%d) — pure damping NOW",
                   static_cast<int>(cur));
      // Black box: capture the run-up to the OPERATOR e-stop. Idempotent,
      // so whichever of e-stop / SAFE_HOLD fires first owns the dump.
      recorder_.Trigger("estop");
      watchdog_.ForceTripEstop();
    }

    switch (cur) {
      case State::STANDBY: {
        // Touch the ready sentinel exactly once on the first STANDBY tick
        // so bash knows our subscribers/timers are live. ROS subscribers
        // are receiving state in the background while we sit here.
        if (!standby_ready_logged_) {
          standby_ready_logged_ = true;
          if (!cli_.ready_sentinel.empty()) {
            std::ofstream r(cli_.ready_sentinel, std::ios::trunc);
            if (!r) {
              RCLCPP_ERROR(node_->get_logger(),
                           "STANDBY: failed to touch ready-sentinel '%s' "
                           "(errno=%d). Bash may show its safety gate "
                           "without confirmation that deploy is ready.",
                           cli_.ready_sentinel.c_str(), errno);
            } else {
              r << "ready " << now << "\n";
              RCLCPP_WARN(node_->get_logger(),
                          "STANDBY: ready-sentinel touched at '%s'. "
                          "Writer is GATED OFF; waiting for trigger.",
                          cli_.ready_sentinel.c_str());
            }
          } else {
            RCLCPP_WARN(node_->get_logger(),
                        "STANDBY: ready (no ready-sentinel configured). "
                        "Writer is GATED OFF; waiting for trigger.");
          }
        }
        // Poll the trigger sentinel. std::ifstream open is the cheapest
        // portable existence check; no allocations on the steady-state
        // miss path.
        std::ifstream probe(cli_.start_trigger_sentinel);
        if (probe.good()) {
          RCLCPP_WARN(node_->get_logger(),
                      "STANDBY: start-trigger-sentinel '%s' detected; "
                      "advancing STANDBY -> INIT.",
                      cli_.start_trigger_sentinel.c_str());
          // Best-effort cleanup so a second invocation doesn't
          // immediately re-trigger.
          std::remove(cli_.start_trigger_sentinel.c_str());
          state_.store(State::INIT);
        }
        return;
      }
      case State::INIT: {
        if (aimdk_io_->AllStateFresh(0.5)) {
          // Latch the current observed joint pose as the safe-hold target
          // *before* WAIT_FOR_CONTROL begins. The 500 Hz writer will keep
          // publishing this latched command for the entire WAIT window
          // (and as the long-tail of SAFE_HOLD if we ever fall back).
          //
          // Capturing the actual pose -- rather than the constructor's
          // initial latch of target=default_angles, kp=0 -- means HAL
          // smoothly takes ownership at the operator's current pose with
          // no jolt:
          //   * Real robot: if MC has been driving firmware-stand (knees
          //     +28 deg, elbows -67 deg), the deploy now holds firmware-
          //     stand. When the operator stops MC and the deploy's
          //     commands actually take effect, there is no 33 deg elbow
          //     snap toward DEFAULT_DOF.
          //   * Sim --init-pose=gantry_hang / gantry_dangle / motion-RSI:
          //     the bridge has spawned the body at a non-default pose;
          //     the deploy now reads that pose from rs and latches it.
          //     The body stays put through the autostart window.
          //
          // The soft-start ramp at CONTROL now also blends from the
          // measured pose (SoftStartRamp::Reset(blend_from), 2026-09-01)
          // -- the Phase 4d follow-up is done. The fix here only
          // addresses the WAIT-time yank.
          if (fresh) {
            std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
            for (std::size_t i = 0; i < NUM_DOFS; ++i) {
              latest_cmd_.target_pos_mj[i] = rs.joint_pos_mj[i];
              latest_cmd_.stiffness_mj[i]  = cli_.dry_run ? 0.0 : kps_scaled_[i];
              latest_cmd_.damping_mj[i]    = cli_.dry_run ? 0.0 : kds_scaled_[i];
            }
            latest_cmd_.reason = "wait_for_control_hold";
          } else {
            RCLCPP_ERROR(
                node_->get_logger(),
                "INIT->WAIT advance fired but SnapshotState reported the "
                "state is stale; safe-hold latch left at constructor's "
                "passive default. Will retry on next tick if the state "
                "freshness re-asserts.");
            return;
          }
          state_.store(State::WAIT_FOR_CONTROL);
          control_entry_s_ = now;
          RCLCPP_INFO(node_->get_logger(),
                      "INIT -> WAIT_FOR_CONTROL (all state sources fresh; "
                      "safe-hold latched at current observed pose)");
        }
        return;
      }
      case State::WAIT_FOR_CONTROL: {
        if (autostart_target_s_ > 0.0 && now >= autostart_target_s_) {
          RCLCPP_WARN(node_->get_logger(),
                      "Autostart elapsed (%.2fs) -> CONTROL%s",
                      cli_.autostart_seconds,
                      cli_.dry_run ? " (DRY-RUN)" : "");
          // Blend from the pose we are actually holding, not from
          // default_angles (which would command a knee dip; see
          // SoftStartRamp::Reset(blend_from) in safety.hpp).
          if (fresh) ramp_.Reset(rs.joint_pos_mj); else ramp_.Reset();
          watchdog_.Reset();
          prop_buf_.Reset();
          last_action_il_.fill(0.0);
          control_entry_s_ = now;

          // Yaw-anchor the reference motion to the robot's current heading.
          // This is the C++ stand-in for IsaacLab's Reference State
          // Initialization (RSI): training teleports the simulated robot to
          // motion[0] every episode, so the policy never sees the motion's
          // absolute world yaw drift away from the robot's. On hardware the
          // robot is wherever the gantry left it, so without this call the
          // policy would see the recorded yaw of the .pkl file (often tens
          // of degrees off — e.g. ~96 deg for x2_ultra_idle_stand) as the
          // motion-anchor diff at t=0. See ReferenceMotion::Anchor() for
          // the full rationale and gear_sonic_deploy/scripts/
          // compare_deploy_vs_isaaclab_obs.py for the parity check that
          // surfaced the missing anchor.
          //
          // Only `fresh && AllStateFresh` could have made INIT advance us to
          // WAIT_FOR_CONTROL, so rs.base_quat_wxyz here is the same IMU
          // sample the operator sees on the robot — exactly what we want.
          if (fresh) {
            ref_motion_->Anchor(rs.base_quat_wxyz);
            RCLCPP_WARN(
                node_->get_logger(),
                "Reference motion '%s' yaw-anchored to robot heading "
                "(robot yaw = %.2f deg, applied Δyaw = %.2f deg). "
                "Motion pitch/roll left as recorded (gravity-grounded).",
                ref_motion_->Name().c_str(),
                yaw_from_quat_wxyz(rs.base_quat_wxyz) * 180.0 / 3.14159265358979323846,
                ref_motion_->yaw_anchor_delta() * 180.0 / 3.14159265358979323846);
          } else {
            RCLCPP_ERROR(
                node_->get_logger(),
                "WAIT->CONTROL fired on a STALE IMU sample; reference "
                "motion will NOT be yaw-anchored. Expect the motion-anchor "
                "diff to start far OOD. Aborting transition; will retry on "
                "next tick.");
            return;
          }

          state_.store(State::CONTROL);
        }
        return;
      }
          case State::SAFE_HOLD: {
            // Terminal: the writer republishes the latched command forever.
            //
            // OPERATOR E-STOP IN SAFE_HOLD (2026-08-24). This used to return
            // immediately, so an e-stop pressed while in SAFE_HOLD was silently
            // discarded: the poll above tripped the watchdog, but the ONLY consumer
            // of ForceStage2() is ApplySafetyStack, which runs solely on the CONTROL
            // path that SAFE_HOLD never reaches. The robot kept publishing whatever
            // pose was latched on entry -- including a saturated one. Reported from
            // hardware as "it got into this weird pose and doesn't respond any more".
            //
            // Now: on an explicit stage-2 request, latch the SHARED pure-damping
            // profile once and publish that instead.
            //
            // DELIBERATELY NOT automatic on SAFE_HOLD entry. Auto-damping every
            // SAFE_HOLD was the original behaviour and caused the 2026-08-04 incident
            // recorded above: a standing robot slumped with a violent gear whir and
            // the operator pulled the battery. Only an explicit operator e-stop
            // escalates to limp here -- WITH ONE CARVE-OUT below.
            //
            // AUTO PURE-DAMPING WHEN UNAMBIGUOUSLY DOWN (operator directive
            // 2026-08-31: "hold default angles, 4x damping -- we need to fix
            // this"). Holding the STAND pose while lying on the ground is
            // actively wrong: every joint fights the floor at full stiffness
            // (sim rehearsal showed legs frozen mid-air on a fallen robot).
            // The 2026-08-04 incident scenario -- a STANDING robot slumping on
            // a false trip -- cannot satisfy this gate: it requires
            // gravity_body[z] > safe_hold_down_damp_cos (default -0.25, just
            // past the -0.3 trip threshold; upright is -1.0) SUSTAINED on
            // FRESH IMU state (100 ms; 30 ms once past horizontal). A standing IMU glitch recovers
            // below the gate within the dwell; a deep crouch sits near -0.6
            // and never reaches it; only a robot genuinely going/gone down
            // holds it for 300 ms. Operator refinement 2026-08-31: the gate
            // deliberately covers the MID-FALL band too, not just flat-down
            // -- a full-stiffness stand-pose yank mid-crouch or mid-fall
            // makes things WORSE (the documented tilt-trip stage-1 hazard:
            // hold-default-at-full-kp launches the robot from a crouch).
            // --safe-hold-down-damp-cos <= -1.0 disables the carve-out
            // entirely (restores pre-2026-08-31 behaviour).
            if (!safe_hold_damped_ && fresh &&
                cli_.safe_hold_down_damp_cos > -1.0) {
              const auto g_shd = body_frame_gravity_from_quat_wxyz(rs.base_quat_wxyz);
              if (g_shd[2] > cli_.safe_hold_down_damp_cos) {
                if (safe_hold_down_since_s_ < 0.0) safe_hold_down_since_s_ = now;
                // Dwell (operator 2026-08-31: 300 ms let the robot hit the
                // floor stiff): 100 ms filters a standing IMU glitch while
                // catching a real fall ~mid-descent; past HORIZONTAL
                // (grav_z > 0) two consecutive fresh ticks suffice -- no
                // upright robot ever reads past horizontal twice in a row.
                const double dwell_s = (g_shd[2] > 0.0) ? 0.03 : 0.10;
                if (now - safe_hold_down_since_s_ >= dwell_s) {
                  safe_hold_damped_ = true;
                  SafeCommand dc;
                  BuildPureDampingCommand(dc, default_angles);
                  dc.dry_run    = cli_.dry_run;
                  dc.tilt_trip  = true;
                  dc.ramp_alpha = 0.0;
                  dc.reason     = watchdog_.Reason() +
                      " -> DOWN-DETECT (grav_z sustained past trip threshold) -> auto PURE-DAMPING";
                  if (cli_.dry_run) {
                    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
                      dc.stiffness_mj[i] = 0.0;
                      dc.damping_mj[i]   = 0.0;
                    }
                  }
                  {
                    std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
                    latest_cmd_ = dc;
                  }
                  latest_cmd_ready_.store(true, std::memory_order_release);
                  RCLCPP_FATAL(node_->get_logger(),
                               "SAFE_HOLD down-detect: gravity_body[z]=%.3f > %.3f for 0.3 s "
                               "-> robot is DOWN/FALLING -> auto pure damping (standing false-trips "
                               "still get the stand-pose hold)",
                               g_shd[2], cli_.safe_hold_down_damp_cos);
                }
              } else {
                safe_hold_down_since_s_ = -1.0;
              }
            }
            if (watchdog_.ForceStage2() && !safe_hold_damped_) {
              safe_hold_damped_ = true;
              SafeCommand dc;
              BuildPureDampingCommand(dc, default_angles);
              dc.dry_run    = cli_.dry_run;
              dc.tilt_trip  = true;
              dc.ramp_alpha = 0.0;
              dc.reason     = watchdog_.Reason() + " -> PURE-DAMPING (SAFE_HOLD e-stop)";
              if (cli_.dry_run) {
                for (std::size_t i = 0; i < NUM_DOFS; ++i) {
                  dc.stiffness_mj[i] = 0.0;
                  dc.damping_mj[i]   = 0.0;
                }
              }
              {
                std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
                latest_cmd_ = dc;
              }
              latest_cmd_ready_.store(true, std::memory_order_release);
              RCLCPP_FATAL(node_->get_logger(),
                           "operator E-STOP honoured in SAFE_HOLD -> pure damping (%s)",
                           watchdog_.Reason().c_str());
            }
            // ---- SAFE_HOLD -> RECOVER gate (assisted stand-up) ----------
            // Someone has picked the robot up: IMU upright AND the body
            // quiet, sustained. Works from the pure-damping profile (the
            // legs are limp, the helper carries the weight) and from the
            // stand-pose hold of a false trip alike; the gains ramp starts
            // from whatever is latched, so there is never a kp step.
            // Two things veto the gate outright (robot-verified 2026-09-23:
            // a pad E-STOP is a LATCH on the pose wire, so recovering under
            // it just re-enters CONTROL, gets e-stopped on the next tick and
            // loops every ~8 s):
            //   * the operator E-STOP flag is asserted on the wire, or
            //   * this run has already used its recovery attempts.
            const bool estop_on_wire = (zmq_pose_source_ != nullptr)
                                       && zmq_pose_source_->EstopRequested();
            const bool attempts_left = (cli_.recover_max_attempts <= 0)
                                       || (recover_count_ < cli_.recover_max_attempts);
            if (cli_.safe_hold_recover && fresh && (estop_on_wire || !attempts_left)) {
              if (!recover_blocked_logged_) {
                recover_blocked_logged_ = true;
                RCLCPP_WARN(node_->get_logger(),
                            "SAFE_HOLD: RECOVER blocked (%s) -- staying in SAFE_HOLD; "
                            "restart the stack to run again",
                            estop_on_wire ? "operator E-STOP asserted on the pose wire"
                                          : "recovery attempts exhausted");
              }
              recover_upright_since_s_ = -1.0;
            } else if (cli_.safe_hold_recover && fresh) {
              recover_blocked_logged_ = false;
              const auto g_rc = body_frame_gravity_from_quat_wxyz(rs.base_quat_wxyz);
              const double w_rc = std::sqrt(rs.base_ang_vel[0] * rs.base_ang_vel[0]
                                            + rs.base_ang_vel[1] * rs.base_ang_vel[1]
                                            + rs.base_ang_vel[2] * rs.base_ang_vel[2]);
              double qd_rc = 0.0;
              for (std::size_t i = 0; i < NUM_DOFS; ++i)
                qd_rc = std::max(qd_rc, std::abs(rs.joint_vel_mj[i]));
              const bool upright_rc = g_rc[2] < cli_.recover_upright_cos;
              const bool quiet_rc   = (w_rc < cli_.recover_max_base_angvel)
                                      && (qd_rc < cli_.recover_max_joint_vel);
              if (upright_rc && quiet_rc) {
                if (recover_upright_since_s_ < 0.0) {
                  recover_upright_since_s_ = now;
                  RCLCPP_WARN(node_->get_logger(),
                              "SAFE_HOLD: robot reads upright (grav_z %.3f) and quiet "
                              "(|w| %.2f rad/s, max|qd| %.2f rad/s) -> RECOVER in %.1f s if it stays so",
                              g_rc[2], w_rc, qd_rc, cli_.recover_dwell_s);
                }
                if (now - recover_upright_since_s_ >= cli_.recover_dwell_s) {
                  EnterRecover(now, rs);
                }
              } else if (recover_upright_since_s_ >= 0.0) {
                recover_upright_since_s_ = -1.0;
              }
            }
            return;
          }
      case State::RECOVER: {
        // Assisted stand-up. Abort to SAFE_HOLD pure damping the moment the
        // body tilts past the trip threshold (the helper let go, or the
        // stand-up is failing); stale state for > 0.5 s aborts as well.
        const auto g_rv = fresh ? body_frame_gravity_from_quat_wxyz(rs.base_quat_wxyz)
                                : std::array<double, 3>{0.0, 0.0, -1.0};
        const bool tilted_rv = fresh && (g_rv[2] > watchdog_.FallCos());
        if (tilted_rv || (!fresh && stale_state_ticks_ > 25)) {
          SafeCommand dc;
          BuildPureDampingCommand(dc, default_angles);
          dc.dry_run    = cli_.dry_run;
          dc.tilt_trip  = true;
          dc.ramp_alpha = 0.0;
          dc.reason     = tilted_rv ? "RECOVER aborted: tilt past threshold -> PURE-DAMPING"
                                    : "RECOVER aborted: state stale -> PURE-DAMPING";
          if (cli_.dry_run) {
            for (std::size_t i = 0; i < NUM_DOFS; ++i) {
              dc.stiffness_mj[i] = 0.0;
              dc.damping_mj[i]   = 0.0;
            }
          }
          {
            std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
            latest_cmd_ = dc;
          }
          latest_cmd_ready_.store(true, std::memory_order_release);
          safe_hold_damped_        = true;
          safe_hold_down_since_s_  = -1.0;
          recover_upright_since_s_ = -1.0;
          RCLCPP_FATAL(node_->get_logger(), "%s (grav_z %.3f, %.1f s into the stand-up) -> SAFE_HOLD",
                       dc.reason.c_str(), g_rv[2], now - recover_entry_s_);
          recorder_.Trigger("recover_abort");
          state_.store(State::SAFE_HOLD);
          return;
        }
        const double A  = std::max(cli_.recover_stiffen_s, 1e-3);
        const double B  = std::max(recover_standup_total_s_, 1e-3);
        const double t  = now - recover_entry_s_;
        const double ga = std::clamp(t / A, 0.0, 1.0);
        const double pl = std::clamp((t - A) / B, 0.0, 1.0);
        const double pa = 0.5 - 0.5 * std::cos(pl * 3.14159265358979323846);  // half-cosine ease
        SafeCommand sc;
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          sc.target_pos_mj[i] = (1.0 - pa) * recover_start_pos_[i] + pa * stand_pose_target_[i];
          sc.stiffness_mj[i]  = cli_.dry_run ? 0.0 : ((1.0 - ga) * recover_kp_start_[i] + ga * kps_scaled_[i]);
          sc.damping_mj[i]    = cli_.dry_run ? 0.0 : ((1.0 - ga) * recover_kd_start_[i] + ga * kds_scaled_[i]);
        }
        sc.dry_run    = cli_.dry_run;
        sc.tilt_trip  = false;
        sc.ramp_alpha = pa;           // tick.csv: 0 = stiffening in place, 1 = at the stand pose
        sc.reason     = (t < A) ? "recover_stiffen" : "recover_standup";
        {
          std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
          latest_cmd_ = sc;
        }
        latest_cmd_ready_.store(true, std::memory_order_release);
        logger_.Log(now, rs.joint_pos_mj, rs.joint_vel_mj,
                    rs.base_quat_wxyz, rs.base_ang_vel,
                    last_action_il_, sc);
        if (zmq_debug_pub_) {
          PublishDebugFrame(now, rs, last_action_il_, sc, /*policy_time=*/now);
        }
        if (pl >= 1.0) {
          // Hand back to the policy exactly like WAIT_FOR_CONTROL -> CONTROL:
          // soft-start ramp from the measured pose, watchdogs reset, LPF
          // re-seeded, reference re-anchored to the current heading.
          if (fresh) ramp_.Reset(rs.joint_pos_mj); else ramp_.Reset();
          watchdog_.Reset();
          prop_buf_.Reset();
          last_action_il_.fill(0.0);
          target_lpf_initialized_ = false;
          pose_ref_watchdog_.ClearTrip();
          safe_hold_damped_        = false;
          safe_hold_down_since_s_  = -1.0;
          recover_upright_since_s_ = -1.0;
          if (fresh && ref_motion_) ref_motion_->Anchor(rs.base_quat_wxyz);
          control_entry_s_ = now;
          RCLCPP_WARN(node_->get_logger(),
                      "RECOVER complete (%.1f s: stiffen %.1f s + stand-up %.1f s, attempt #%d) "
                      "-> CONTROL with a fresh soft-start ramp from the measured pose",
                      t, A, B, recover_count_);
          recorder_.Trigger("recover_done");
          state_.store(State::CONTROL);
        }
        return;
      }
      case State::SAFE_IDLE: {
        // Recoverable starvation hold (split-topology safety). Writer
        // keeps publishing the latched safe_idle_cmd_ (default_angles
        // + deploy-mode kp + 4x kd) at 500 Hz so the body stays under
        // torque. We update the pose-ref age + ReadyToResume() every
        // tick so x2_debug reflects the live state, then check whether
        // BOTH gates (fresh wire AND recent operator chord) have closed
        // -- if so, transition back to CONTROL with a fresh soft-start
        // ramp and a reset starvation watchdog.
        double age_s = std::numeric_limits<double>::infinity();
        if (zmq_pose_source_ != nullptr) {
          const double last_rx = zmq_pose_source_->LastReceivedMonotonicS();
          if (last_rx > 0.0) age_s = std::max(0.0, now - last_rx);
        }
        const bool ready_to_resume =
            pose_ref_watchdog_.ReadyToResume(now, age_s);
        const bool resume_chord_fresh =
            (zmq_resume_sub_ != nullptr)
            && zmq_resume_sub_->LatestFresh(now, /*window_s=*/0.5);
        if (ready_to_resume && resume_chord_fresh) {
          RCLCPP_WARN(node_->get_logger(),
                      "SAFE_IDLE -> CONTROL: resume chord received "
                      "(latest %.3f s ago) AND pose-ref fresh for >= %.3f s "
                      "(current age %.3f s). Re-engaging policy with a "
                      "fresh soft-start ramp.",
                      now - zmq_resume_sub_->LastReceivedMonotonicS(),
                      cli_.pose_ref_min_fresh_s, age_s);
          // Reset the soft-start ramp so the policy target blends back
          // from the CURRENT measured pose over --ramp-seconds (not from
          // default_angles -- that commanded a knee dip). The CONTROL
          // state will pick up from the next tick exactly as if we'd
          // just entered from WAIT_FOR_CONTROL.
          if (fresh) ramp_.Reset(rs.joint_pos_mj); else ramp_.Reset();
          pose_ref_watchdog_.ClearTrip();
          safe_idle_entry_s_ = -1.0;
          control_entry_s_   = now;
          state_.store(State::CONTROL);
        }
        // Logger snapshot so the CSV captures SAFE_IDLE dwell explicitly
        // (operators looking at tick.csv will see ramp_alpha=0 +
        // reason="safe_idle" for the duration).
        logger_.Log(now, rs.joint_pos_mj, rs.joint_vel_mj,
                    rs.base_quat_wxyz, rs.base_ang_vel,
                    last_action_il_, safe_idle_cmd_);
        if (zmq_debug_pub_) {
          PublishDebugFrame(now, rs, last_action_il_, safe_idle_cmd_,
                            /*policy_time=*/now);
        }
        return;
      }
      case State::RAMP_OUT: {
        // Soft-EXIT ramp: linearly interpolate target_pos from the snapshot
        // we took on entering RAMP_OUT (last policy command) toward MC's
        // STAND_DEFAULT pose (or default_angles if --stand-default-pose was
        // not given) over cli_.return_seconds. We hold kp/kd at the
        // deploy-mode values the policy was trained with -- DO NOT lerp
        // kp/kd toward MC's stand-mode gains, even though we know them.
        //
        // Why: MC's STAND_DEFAULT gains have, per joint, very different
        // damping ratios than the deploy gains (e.g. elbow kp triples
        // 14 -> 50 while kd barely moves 0.9 -> 1.0; hip kd halves
        // 6.3 -> 4.0). Lerping kp/kd while the position target is also
        // moving by up to 1.2 rad over 2 s (elbows, when the policy left
        // arms extended) produces a transiently under-damped system
        // chasing a moving setpoint = motor whir + ringing. Verified the
        // hard way on 2026-05-03; reverted to deploy-mode gains here.
        // MC will switch to its own gains in a single message at the
        // takeover boundary, with the position already at the matching
        // pose, so the gain step is benign (low position error).
        // We always keep PD active -- never zero torque -- so the body
        // stays balanced through the whole ramp.
        // When the ramp completes we either:
        //   * --hold-for-mc-timeout-s > 0 -> transition to HOLD_FOR_MC
        //     and wait for MC to take back over the joint command bus;
        //   * otherwise -> request shutdown (legacy).
        const double T = std::max(cli_.return_seconds, 1e-6);
        const double t = now - ramp_out_entry_s_;
        const double alpha = std::clamp(t / T, 0.0, 1.0);  // 0=start, 1=done
        SafeCommand sc;
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          sc.target_pos_mj[i] = (1.0 - alpha) * ramp_out_start_pos_[i]
                                + alpha * stand_pose_target_[i];
          sc.stiffness_mj[i]  = cli_.dry_run ? 0.0 : kps_scaled_[i];
          sc.damping_mj[i]    = cli_.dry_run ? 0.0 : kds_scaled_[i];
        }
        sc.dry_run    = cli_.dry_run;
        sc.tilt_trip  = false;
        sc.ramp_alpha = 1.0 - alpha;  // for tick.csv: 1.0=just started, 0.0=done
        sc.reason     = "ramp_out";
        {
          std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
          latest_cmd_ = sc;
        }
        latest_cmd_ready_.store(true, std::memory_order_release);
        logger_.Log(now, rs.joint_pos_mj, rs.joint_vel_mj,
                    rs.base_quat_wxyz, rs.base_ang_vel,
                    last_action_il_, sc);
        if (alpha >= 1.0) {
          if (cli_.hold_for_mc_timeout_s > 0.0) {
            RCLCPP_WARN(node_->get_logger(),
                        "RAMP_OUT complete (%.2fs) -> HOLD_FOR_MC "
                        "(timeout %.1fs, waiting for MC to take back over the "
                        "joint command bus). Position now at MC STAND_DEFAULT "
                        "pose; HOLD_FOR_MC will step gains up to MC-stand "
                        "stiffness/damping for a firmer hold (legs, ankles, "
                        "waist). Position error at the gain step is ~0 so the "
                        "torque kick is negligible.",
                        cli_.return_seconds, cli_.hold_for_mc_timeout_s);
            EnterHoldForMc(now);
          } else {
            RCLCPP_WARN(node_->get_logger(),
                        "RAMP_OUT complete (%.2fs) -> shutting down. "
                        "Joints commanded back to STAND_DEFAULT pose; safe to "
                        "hand off to MC.", cli_.return_seconds);
            rclcpp::shutdown();
          }
        }
        return;
      }
      case State::HOLD_FOR_MC: {
        // Static-pose hold while MC restarts. We publish MC's STAND_DEFAULT
        // pose with MC-stand kp/kd (loaded from --stand-default-pose YAML;
        // falls back to deploy gains if the YAML wasn't supplied) so the
        // legs / ankles / waist are as stiff as MC-STAND would have them
        // (e.g. waist pitch kp 14 -> 40, knee kp 99 -> 150, ankle kp
        // 21 -> 30-40). This addresses the operator's "legs and waist
        // not holding" feedback during the post-policy hold: deploy-mode
        // gains were tuned for the *active* policy, not for a static
        // hold against gravity. The gain step at HOLD_FOR_MC entry is
        // safe because we just ramped position to stand_pose_target_,
        // so position error is ~0 and the torque kick from the kp step
        // is small. Note: deploy still has NO active balance controller
        // -- this hold is "joints stiff, body free to tilt" -- so a
        // strong perturbation can still tilt the torso. That's the
        // architectural ceiling until MC takes back over.
        // We exit when EITHER:
        //   (a) the MC-takeover detector callback (subscribed to
        //       /aima/hal/joint/leg/command and /aima/hal/joint/waist/command
        //       with ignore_local_publications=true) has seen its first
        //       message from a non-self publisher = MC is publishing again;
        //   (b) hold_for_mc_timeout_s elapses (MC didn't come back; we
        //       give up and let bash decide what to do next).
        SafeCommand sc;
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          sc.target_pos_mj[i] = stand_pose_target_[i];
          sc.stiffness_mj[i]  = cli_.dry_run ? 0.0 : stand_pose_stiffness_[i];
          sc.damping_mj[i]    = cli_.dry_run ? 0.0 : stand_pose_damping_[i];
        }
        sc.dry_run    = cli_.dry_run;
        sc.tilt_trip  = false;
        sc.ramp_alpha = 0.0;
        sc.reason     = "hold_for_mc";
        {
          std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
          latest_cmd_ = sc;
        }
        latest_cmd_ready_.store(true, std::memory_order_release);
        logger_.Log(now, rs.joint_pos_mj, rs.joint_vel_mj,
                    rs.base_quat_wxyz, rs.base_ang_vel,
                    last_action_il_, sc);
        // Exit policy (re-revised 2026-05-03 after operator observed
        // ~1.5 s of zero-torque after deploy released the bus -- the
        // FAST-EXIT-on-first-publish version had this bug because MC
        // publishes commands while still in PASSIVE_DEFAULT during boot):
        //
        //   PRIMARY GATE: --hold-for-mc-exit-sentinel.
        //     Bash touches the file after escalating MC all the way to
        //     STAND_DEFAULT. While the file is absent, deploy keeps
        //     publishing MC's stand pose with MC-stand gains -- so the
        //     legs/ankles/waist stay actively held throughout MC's
        //     PASSIVE -> JOINT -> STAND boot sequence. There IS a brief
        //     dual-publisher window once MC enters JOINT_DEFAULT (~1 s
        //     before STAND_DEFAULT activates), but both deploy and MC
        //     are PD-holding the same stand pose with similar gains, so
        //     the conflict is small and the robot stays under torque.
        //
        //   FALLBACK: first-MC-publish detection (legacy path, no
        //     exit-sentinel configured). Exits the moment the takeover
        //     detector fires. Use this only if you've arranged some
        //     other means of ensuring MC is in a holding mode (not
        //     PASSIVE) before its first publish.
        //
        //   BACKSTOP: hold_for_mc_timeout_s. Hard upper bound on the
        //     hold so deploy doesn't get stuck if bash crashes.
        //
        // First-publish detection is still LOGGED for tracing in both
        // modes -- it's the most useful timestamp for measuring the
        // STAND_DEFAULT settle latency from MC's side.
        const double held       = now - hold_for_mc_entry_s_;
        const bool   takeover   = mc_takeover_detected_.load(std::memory_order_acquire);
        if (takeover && !mc_takeover_logged_) {
          mc_takeover_logged_ = true;
          // ms-precision delta from HOLD_FOR_MC entry to actual DDS
          // callback firing. Captured inside the callback before any
          // locks, so it is independent of OnControl scheduling jitter.
          using ms_d = std::chrono::duration<double, std::milli>;
          double dt_ms_callback = -1.0;
          double dt_ms_now      = -1.0;
          {
            std::lock_guard<std::mutex> lk(mc_takeover_topic_mutex_);
            if (mc_takeover_steady_ts_.time_since_epoch().count() != 0
                && hold_for_mc_entry_steady_.time_since_epoch().count() != 0) {
              dt_ms_callback = ms_d(mc_takeover_steady_ts_
                                    - hold_for_mc_entry_steady_).count();
            }
            dt_ms_now = ms_d(std::chrono::steady_clock::now()
                             - hold_for_mc_entry_steady_).count();
          }
          if (cli_.hold_for_mc_exit_sentinel.empty()) {
            RCLCPP_WARN(node_->get_logger(),
                        "HOLD_FOR_MC: FIRST MC PUBLISH on '%s' at "
                        "callback=+%.3f ms, OnControl=+%.3f ms (entry+%.2fs). "
                        "Exiting -> MC takes the bus alone (no dual-publisher "
                        "fight). Total handoff latency = "
                        "callback->next OnControl tick (<= 20 ms).",
                        mc_takeover_topic_.c_str(),
                        dt_ms_callback, dt_ms_now, held);
          } else {
            RCLCPP_WARN(node_->get_logger(),
                        "HOLD_FOR_MC: FIRST MC PUBLISH on '%s' at "
                        "callback=+%.3f ms, OnControl=+%.3f ms (entry+%.2fs). "
                        "Continuing to publish STAND_DEFAULT pose (MC is "
                        "likely still in PASSIVE_DEFAULT during boot); "
                        "waiting for exit-sentinel to fire from bash after "
                        "MC reaches STAND_DEFAULT.",
                        mc_takeover_topic_.c_str(),
                        dt_ms_callback, dt_ms_now, held);
          }
        }

        // Exit policy:
        //   * If an exit-sentinel is configured, the sentinel is the ONLY
        //     gate. Deploy keeps publishing MC's STAND_DEFAULT pose until
        //     bash explicitly tells us "MC is in STAND_DEFAULT" by
        //     touching the file. This is the intended design from the
        //     original handoff plan: MC boots in PASSIVE_DEFAULT (zero
        //     torque), then escalates PASSIVE -> JOINT -> STAND. Its
        //     FIRST publish lands while it's still in PASSIVE; releasing
        //     the bus then would drop the robot to zero torque for the
        //     ~1-2 s it takes bash to escalate to STAND_DEFAULT (this
        //     was the bug the operator observed on 2026-05-03 20:33:33:
        //     "robot not in control for a couple of seconds before MC
        //     said switching to standing mode").
        //   * If no exit-sentinel was configured (legacy callers, or
        //     a future caller that arbitrates handoff via some other
        //     channel), fall back to first-MC-publish-detection.
        //   * The hold_for_mc_timeout_s cap is the ultimate backstop
        //     in either case.
        bool should_exit = false;
        const char* exit_reason = "";
        if (!cli_.hold_for_mc_exit_sentinel.empty()) {
          // Exit-sentinel mode (the path bash configures by default).
          // First-MC-publish detection is informational only here --
          // we logged it above but do NOT exit on it.
          std::ifstream probe(cli_.hold_for_mc_exit_sentinel);
          if (probe.good()) {
            should_exit = true;
            exit_reason = "exit-sentinel touched";
          }
        } else if (takeover) {
          should_exit = true;
          exit_reason = "first MC publish detected (no exit-sentinel configured)";
        }
        if (should_exit) {
          RCLCPP_WARN(node_->get_logger(),
                      "HOLD_FOR_MC: %s after %.2fs -> shutting down. "
                      "Robot stayed in STAND_DEFAULT pose throughout the "
                      "handoff (no DAMPING / PASSIVE window).",
                      exit_reason, held);
          ClearHoldForMcSentinel();
          rclcpp::shutdown();
          return;
        }
        if (held >= cli_.hold_for_mc_timeout_s) {
          RCLCPP_ERROR(node_->get_logger(),
                       "HOLD_FOR_MC: timed out after %.2fs without bash "
                       "creating the exit-sentinel. Shutting down anyway -- "
                       "the bus will go silent. Operator: confirm MC is "
                       "alive and in STAND_DEFAULT before re-enabling.",
                       held);
          ClearHoldForMcSentinel();
          rclcpp::shutdown();
        }
        return;
      }
      case State::CONTROL: break;
    }

    // Optional soft-shutdown trigger (graceful Ctrl-C path). Polled BEFORE
    // the max-duration check and BEFORE the stale-state guard, for the
    // same reason: a frozen-state robot must still respond to the
    // operator's graceful-stop request. When EITHER the in-process flag
    // (set by SoftShutdownSignalHandler on SIGINT) OR the on-disk sentinel
    // (touched by the bash cleanup trap) trips, we take the exact same
    // RAMP_OUT path that --max-duration would -- see the long-form comment
    // below. Sentinel polling is the belt; in-process flag is the
    // suspenders. We skip the file probe entirely when the flag wasn't
    // provided (legacy callers pay zero cost; SoftShutdownSignalHandler is
    // only installed when the CLI flag is set, so the in-process flag also
    // stays false in legacy mode).
    if (!cli_.soft_shutdown_trigger_sentinel.empty()) {
      bool trigger = g_soft_shutdown_requested.load(std::memory_order_acquire);
      if (!trigger) {
        std::ifstream probe(cli_.soft_shutdown_trigger_sentinel);
        if (probe.good()) trigger = true;
      }
      if (trigger) {
        const bool via_signal =
            g_soft_shutdown_requested.load(std::memory_order_acquire);
        const char* src = via_signal ? "in-process SIGINT/SIGTERM"
                                      : "on-disk sentinel";
        if (cli_.return_seconds > 0.0) {
          {
            std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
            ramp_out_start_pos_ = latest_cmd_.target_pos_mj;
          }
          ramp_out_entry_s_ = now;
          state_.store(State::RAMP_OUT);
          RCLCPP_WARN(node_->get_logger(),
                      "Soft-shutdown requested (source: %s, sentinel='%s') "
                      "-> RAMP_OUT (%.2fs return-to-stand) -> HOLD_FOR_MC. "
                      "Robot stays under torque through MC's "
                      "PASSIVE_DEFAULT boot.",
                      src,
                      cli_.soft_shutdown_trigger_sentinel.c_str(),
                      cli_.return_seconds);
        } else {
          RCLCPP_WARN(node_->get_logger(),
                      "Soft-shutdown requested (source: %s, sentinel='%s') "
                      "-> shutting down immediately (--return-seconds "
                      "disabled). Bus will be silent until MC restarts.",
                      src,
                      cli_.soft_shutdown_trigger_sentinel.c_str());
          rclcpp::shutdown();
        }
        return;
      }
    }

    // Optional bounded-duration auto-shutdown. Triggered N seconds after we
    // entered CONTROL (control_entry_s_ is set in WAIT->CONTROL above). We
    // run this BEFORE the stale-state guard so a frozen robot still hits the
    // deadline -- the whole point of --max-duration is the operator can walk
    // away from a smoke test, and "stuck on stale state forever" defeats
    // that. Instead of shutting down immediately, transition to RAMP_OUT so
    // the joints get linearly returned to default_angles over
    // --return-seconds before MC takes back over. Set --return-seconds 0 to
    // get the legacy "shutdown immediately" behaviour.
    if (cli_.max_duration > 0.0 && (now - control_entry_s_) >= cli_.max_duration) {
      if (cli_.return_seconds > 0.0) {
        // Snapshot the current target as the ramp-out start pose and switch
        // states. Next OnControl tick will be served by the RAMP_OUT case
        // above.
        {
          std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
          ramp_out_start_pos_ = latest_cmd_.target_pos_mj;
        }
        ramp_out_entry_s_ = now;
        state_.store(State::RAMP_OUT);
        RCLCPP_WARN(node_->get_logger(),
                    "Max duration elapsed (%.2fs in CONTROL) -> RAMP_OUT "
                    "(%.2fs return-to-default)%s",
                    cli_.max_duration, cli_.return_seconds,
                    cli_.dry_run ? " (DRY-RUN)" : "");
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "Max duration elapsed (%.2fs in CONTROL) -> shutting down "
                    "(--return-seconds disabled)%s",
                    cli_.max_duration,
                    cli_.dry_run ? " (DRY-RUN)" : "");
        rclcpp::shutdown();
      }
      return;
    }

    // ---- Pose-ref starvation watchdog (split-topology safety) -------------
    // Active only on the ZMQ input path. Trips CONTROL -> SAFE_IDLE the
    // moment the wire goes stale. We measure age as
    //   now - ZmqPoseInputSource::LastReceivedMonotonicS()
    // (both in the same steady_clock seconds frame) and pass that to
    // ``pose_ref_watchdog_.Update`` which is idempotent on repeat calls
    // (latched until ClearTrip() in SAFE_IDLE). Has to run BEFORE the
    // ``!fresh`` IMU guard below because a missing IMU sample is a
    // legitimate "deploy is alive but doing nothing" condition; missing
    // pose-ref frames are not, and SAFE_IDLE entry doesn't need a fresh
    // IMU snapshot to latch its safe-hold target (default_angles is
    // independent of body state).
    if (pose_ref_watchdog_active_ && zmq_pose_source_ != nullptr) {
      const double last_rx = zmq_pose_source_->LastReceivedMonotonicS();
      const bool trip_now  = pose_ref_watchdog_.Update(now, last_rx);
      if (trip_now) {
        RCLCPP_ERROR(node_->get_logger(),
                     "CONTROL -> SAFE_IDLE: %s. Holding default_angles "
                     "with 4x kd; awaiting operator resume chord on "
                     "'%s' AND >= %.3f s of fresh pose frames.",
                     pose_ref_watchdog_.Reason().c_str(),
                     cli_.zmq_resume_topic.c_str(),
                     cli_.pose_ref_min_fresh_s);
        // Build the latched safe-idle command once.
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          safe_idle_cmd_.target_pos_mj[i] = default_angles[i];
          safe_idle_cmd_.stiffness_mj[i]  = cli_.dry_run ? 0.0 : kps_scaled_[i];
          safe_idle_cmd_.damping_mj[i]    = cli_.dry_run ? 0.0 : kds_scaled_[i] * 4.0;
        }
        safe_idle_cmd_.dry_run    = cli_.dry_run;
        safe_idle_cmd_.tilt_trip  = false;
        safe_idle_cmd_.ramp_alpha = 0.0;
        safe_idle_cmd_.reason     = "safe_idle";
        {
          std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
          latest_cmd_ = safe_idle_cmd_;
        }
        latest_cmd_ready_.store(true, std::memory_order_release);
        safe_idle_entry_s_ = now;
        state_.store(State::SAFE_IDLE);
        return;
      }
    }

    if (!fresh) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                           "CONTROL: stale or missing state, skipping tick");
      return;
    }

    // ---- Build observation -------------------------------------------------
    // 1. IL-remap measured joint pos/vel; subtract default for jpos_rel.
    std::array<double, NUM_DOFS> jpos_il{}, jvel_il{}, jpos_rel_il{};
    for (std::size_t il = 0; il < NUM_DOFS; ++il) {
      const std::size_t mj = static_cast<std::size_t>(isaaclab_to_mujoco[il]);
      jpos_il[il]     = rs.joint_pos_mj[mj];
      jvel_il[il]     = rs.joint_vel_mj[mj];
      jpos_rel_il[il] = jpos_il[il] - default_angles[mj];
    }
    // 2. Body-frame gravity from IMU quaternion.
    const auto grav = body_frame_gravity_from_quat_wxyz(rs.base_quat_wxyz);

    prop_buf_.Append(rs.base_ang_vel, jpos_rel_il, jvel_il, last_action_il_, grav);

    // 3. Tokenizer reference window.
    //
    // Bootstrap-safe override: when the ZMQ pose-ref input source has
    // never received a real frame, ``ZmqPoseInputSource::Connect()``
    // pre-fills ``latest_frame_.root_quat_xyzw`` with the identity quat
    // (zmq_pose_input_source.cpp:81-82). Without intervention, the
    // tokenizer's ``rel = inv(measured) * R_identity`` then turns into
    // an actively-driven yaw error and the policy twists the body
    // back to world +X every tick -- the symptom captured in terminal
    // logs as `pose_ref_age=-1.000s mc_mode=0` while the robot
    // visibly rotates back to its spawn heading after every nudge.
    //
    // The fix is local: when we know LastReceivedMonotonicS() < 0
    // (the sentinel for "never received"), pass the CURRENT measured
    // quat in as the reference quat for every future-window slot.
    // The tokenizer then computes ``rel = inv(measured) * measured =
    // identity`` and the policy holds whatever orientation the body
    // is in. Once a real pose-ref frame lands, this branch becomes
    // a no-op and the planner / proxy reference takes over as
    // intended.
    //
    // Joint targets continue to come from the reference (default_angles
    // for the bootstrap default; whatever the planner sends after);
    // only the orientation channel is rebased. Doing this here rather
    // than mutating ZmqPoseInputSource keeps the input-source contract
    // ("Sample() returns latest_frame_") byte-identical for every
    // other consumer (debug, recorder, eval).
    const double policy_time = now - control_entry_s_;
    // New control tick: let the ZMQ sources re-snapshot their latest
    // window on the next Sample(). Without this the wrist/head bypasses
    // sampled a window frozen at engage for the whole whole-body session
    // (2026-09-04; see ZmqPoseInputSource::BeginTick()).
    if (zmq_pose_source_ != nullptr) zmq_pose_source_->BeginTick();
    if (token_source_ != nullptr)    token_source_->BeginTick();
    std::optional<std::array<double, 4>> tok_ref_override;
    if (zmq_pose_source_ != nullptr &&
        zmq_pose_source_->LastReceivedMonotonicS() < 0.0) {
      tok_ref_override = wxyz_to_xyzw(rs.base_quat_wxyz);
    }
    // ---- whole-body overlay: per-tick source select ----------------------
    // Token graph drives ONLY while the laptop stream is fresh AND engaged;
    // otherwise the pad/kplanner pose path drives (gamepad precedence,
    // operator design 2026-08-29). Authority changes re-arm the soft-start
    // ramp so the handover blends instead of stepping.
    bool token_active = false;
    const bool dual_head = (smpl_source_ != nullptr);
    if (dual_head) {
      // ---- NATIVE DUAL-HEAD authority (2026-09-04) ------------------------
      // The laptop's SMPL stream is consumed here. The smpl head drives when
      // the stream is FRESH (< wb_stale_s, not frozen) AND the sender says
      // ENGAGED, after a wb_warmup_s window fill. An explicit disengage
      // (engaged=0 on a fresh stream = operator B) releases in ONE tick;
      // a dead link (stale) HOLDS the smpl head on the latched window while
      // the robot is stable (layer A) until the kplanner's reference moves
      // away from the measured pose (= it started a RETURN on a pad intent)
      // or stability is lost.
      const double snow  = SteadyNow();
      const bool   fresh = smpl_source_->Fresh(snow, cli_.wb_stale_s);
      const bool   eng   = smpl_source_->LatestEngaged() > 0.5;
      bool want = fresh && eng;
      if (want && !wb_want_prev_) {
        if (wb_hold_) {
          // RE-ENGAGE FROM HOLD (operator session 2026-09-04 15:51): the
          // smpl head has been driving on the frozen hold window; a
          // warm-up here made want=false while token_active stayed true,
          // so the hold re-entered EVERY TICK (50 log lines/s). Continue
          // on the live window: no warm-up (the sender streams at 50 Hz
          // whether engaged or not, so the ring is full), action
          // crossfade from the frozen-window output, and the heading
          // RE-ANCHORED like a first engage: keeping the old alignment
          // spun the robot 80 deg when the operator re-engaged after a
          // sender restart 12 min into a hold (2026-09-04 16:55).
          wb_warmup_until_ = snow;
          wb_yaw_align_.reset();
          wb_switch_t0_    = snow;
          wb_switch_from_  = last_action_il_;
          RCLCPP_WARN(node_->get_logger(),
                      "WB HOLD -> RE-ENGAGED: smpl head continues on the live "
                      "window (no warm-up, heading re-anchored, crossfade %.2fs)",
                      cli_.wb_switch_ramp_s);
        } else {
          wb_warmup_until_ = snow + cli_.wb_warmup_s;
          wb_yaw_align_.reset();          // heading re-anchored at every engage
        }
      }
      wb_want_prev_ = want;
      if (want && snow < wb_warmup_until_) want = false;
      double max_qd = 0.0;
      for (std::size_t i = 0; i < 12; ++i) max_qd = std::max(max_qd, std::fabs(rs.joint_vel_mj[i]));
      const bool stable = (grav[2] < -0.95) && (max_qd < 1.0);
      if (!want && cli_.wb_hold) {
        if (wb_hold_) {
          bool ref_moved = false;
          if (zmq_pose_source_ != nullptr && zmq_pose_source_->has_body_reference()) {
            const auto rf = zmq_pose_source_->Sample(policy_time);
            // Hand over on the FIRST tick the kplanner's return moves the
            // reference (1 rad/s x 20 ms = 0.02 rad): waiting for 0.15 rad
            // (8.6 deg) handed the pose head an 8.6 deg tracking error plus
            // a moving target on its first tick (dual2 legs: 5-20 rad/s
            // target swing at the hand-over).
            // Primary: the kplanner's pre-roll flag (wb_return=1 for 0.24 s
            // BEFORE its reference moves) -> hand over onto a static,
            // zero-error reference. Fallback: the reference actually moved
            // vs its value 0.25 s into the hold (the shadow->hold settle).
            if (zmq_pose_source_->WbReturnRequested(snow)) ref_moved = true;
            if (snow - wb_hold_t0_ >= 0.25) {
              if (!wb_hold_ref_.has_value()) wb_hold_ref_ = rf.joint_pos_mj;
              for (std::size_t i = 0; i < 12; ++i) {
                if (std::fabs(rf.joint_pos_mj[i] - (*wb_hold_ref_)[i]) > 0.02) { ref_moved = true; break; }
              }
            }
          }
          // Operator decision 2026-09-04: a B release HOLDS too ("stick to
          // the current pose if stable; the first gamepad command switches
          // to the kplanner"). Only the kplanner's return (pad intent) or
          // instability releases the hold; re-engage resumes shadowing.
          if (stable && !ref_moved) {
            want = true;                  // keep holding
          } else {
            wb_hold_ = false;
            wb_hold_ref_.reset();
            RCLCPP_WARN(node_->get_logger(),
                        "WB HOLD released (%s) -> pose-ref head",
                        !stable ? "unstable" : "kplanner return started");
          }
        } else if (token_active_prev_ && stable) {
          wb_hold_ = true;
          wb_hold_ref_.reset();
          wb_hold_t0_ = snow;
          // Freeze the SMPL window at hold entry: on a B release the
          // sender keeps streaming the operator's live body (engaged=0),
          // and "hold" must mean the pose at B, not "keep following".
          {
            std::array<SmplFrame, SMPL_WINDOW> hw;
            if (smpl_source_->Window(snow, hw)) wb_hold_win_ = hw;
            else wb_hold_win_.reset();
          }
          // Wrists: the sender publishes wrist_pr = 0 once disengaged, so
          // the bypass must keep the LAST ENGAGED wrists for the hold
          // (operator session 2026-09-04 15:16: wrists snapped to zero on
          // every B). Latched here, used by the wrist bypass below.
          wb_hold_wrist_ = smpl_source_->LatestWristPr();
          wb_hold_wrist_yaw_ = smpl_source_->LatestWristYaw();
          want = true;
          RCLCPP_WARN(node_->get_logger(),
                      "WB HOLD: %s while stable -> smpl head stays on the latched window "
                      "until a pad command", fresh ? "operator disengage" : "stream lost");
        }
      }
      if (fresh && eng) wb_hold_ = false;
      if (!wb_hold_) wb_hold_win_.reset();
      token_active = want;
      // Telemetry = "operator stream live and engaged" (NOT the internal
      // hold): on a link drop the kplanner sees 0, holds its legs at the
      // measured pose (== what this hold does) and, on a pad intent, starts
      // its rate-limited RETURN -- the reference moving is what releases
      // the hold above and hands the legs over continuously.
      wb_engaged_dbg_ = (fresh && eng) ? 1.0 : 0.0;
    } else if (cli_.whole_body_teleop && token_source_ != nullptr) {
      const double last_rx = token_source_->LastReceivedMonotonicS();
      const double tok_age = (last_rx < 0.0)
          ? std::numeric_limits<double>::infinity()
          : (SteadyNow() - last_rx);
      token_active = (tok_age < 0.5) && (token_source_->LatestEngaged() > 0.5);
    }
    if (dual_head && token_active != token_active_prev_) {
      // Switch = one tick. Smooth the ACTION (not the target from measured):
      // crossfade from the last action over wb_switch_ramp_s. No soft-start
      // reset: the reference is continuous on both sides (kplanner shadow /
      // measured orientation), so there is nothing to ramp from measured.
      wb_switch_t0_   = SteadyNow();
      wb_switch_from_ = last_action_il_;
      RCLCPP_WARN(node_->get_logger(),
                  token_active ? "WHOLE-BODY ENGAGED — smpl head (crossfade %.2fs)"
                               : "WHOLE-BODY RELEASED — pose-ref head (crossfade %.2fs)",
                  cli_.wb_switch_ramp_s);
      if (!token_active && recorder_.enabled()) recorder_.TriggerReusable("wb_release");
      token_active_prev_ = token_active;
    }
    if (!dual_head && token_active != token_active_prev_) {
      // Engage/disengage authority handoff: blend from the measured pose
      // at the flip tick. A default_angles-anchored re-arm commanded a
      // ~0.66 rad knee target (trained default is a slight crouch) for
      // the first ramp seconds of EVERY engage AND disengage -- the sim
      // twin fell in that dip; hardware rode it out as the engage bump.
      ramp_.Reset(rs.joint_pos_mj);
      RCLCPP_WARN(node_->get_logger(),
                  token_active
                      ? "WHOLE-BODY ENGAGED — token graph has authority "
                        "(blending in over soft-start ramp)"
                      : "WHOLE-BODY RELEASED — pad/kplanner has authority "
                        "(blending back over soft-start ramp)");
      // Sim2real plan phase 0: every engage segment ends with a full ring
      // dump (fast state + slow command chain), robot AND sim — the raw
      // material for the same-input plant-diff analysis. Async thread in
      // Trigger(); no control-loop cost.
      if (!token_active && recorder_.enabled()) {
        recorder_.TriggerReusable("wb_release");
      }
      token_active_prev_ = token_active;
    }
    std::vector<float> tok_obs;
    std::vector<float> smpl_obs;
    if (dual_head) {
      // Both heads hot every tick: pose-ref features from the kplanner
      // wire, SMPL obs from the laptop stream (zeros until a window exists).
      tok_obs = BuildTokenizerObs(
          *ref_motion_, policy_time, rs.base_quat_wxyz, tok_ref_override);
      smpl_obs.assign(SMPL_OBS_DIM, 0.0f);
      std::array<SmplFrame, SMPL_WINDOW> win;
      bool have_win = smpl_source_->Window(SteadyNow(), win);
      if (wb_hold_ && wb_hold_win_) { win = *wb_hold_win_; have_win = true; }   // frozen at hold entry
      if (have_win) {
        LevelOperatorRoot(win, cli_.operator_root_level);   // no-op when unset
        if (!wb_yaw_align_) {
          wb_yaw_align_ = SmplYawAlign(rs.base_quat_wxyz, win[SMPL_WINDOW - 1].quat_wxyz);
        }
        // measured wrists, IL slice order [l_yaw l_pitch l_roll r_yaw r_pitch r_roll]
        std::array<double, 6> wrist6 = {
            rs.joint_pos_mj[19], rs.joint_pos_mj[20], rs.joint_pos_mj[21],
            rs.joint_pos_mj[26], rs.joint_pos_mj[27], rs.joint_pos_mj[28]};
        if (token_active) {
          // The trained obs slot is the REFERENCE wrists (retargeted
          // operator), not the robot's own: feeding measured made the
          // smpl head's wrist_yaw a self-referential loop that drifted to
          // 115 deg (operator session 2026-09-04 16:30, "roll not
          // working" == pronation). Operator targets from the stream
          // (latched through a hold); yaw only once the sender sends it.
          const auto wpr = wb_hold_ ? wb_hold_wrist_ : smpl_source_->LatestWristPr();
          wrist6[1] = wpr[0]; wrist6[2] = wpr[1]; wrist6[4] = wpr[2]; wrist6[5] = wpr[3];
          if (smpl_source_->HasWristYaw()) {
            const auto wy = wb_hold_ ? wb_hold_wrist_yaw_ : smpl_source_->LatestWristYaw();
            wrist6[0] = wy[0]; wrist6[3] = wy[1];
          }
        }
        smpl_obs = BuildSmplObs(win, rs.base_quat_wxyz, wrist6, wb_yaw_align_, smpl_ori_mode_);
        if (smpl_dump_.is_open()) {
          // record layout = gear_sonic/scripts/smpl_obs_parity.py REC_DT
          const double  t   = SteadyNow();
          const float   hd  = token_active ? 1.0f : 0.0f;
          std::array<double, 4> ya{std::nan(""), std::nan(""), std::nan(""), std::nan("")};
          if (wb_yaw_align_) ya = *wb_yaw_align_;
          smpl_dump_.write(reinterpret_cast<const char*>(&t), sizeof(double));
          smpl_dump_.write(reinterpret_cast<const char*>(&hd), sizeof(float));
          smpl_dump_.write(reinterpret_cast<const char*>(rs.base_quat_wxyz.data()), 4 * sizeof(double));
          smpl_dump_.write(reinterpret_cast<const char*>(ya.data()), 4 * sizeof(double));
          smpl_dump_.write(reinterpret_cast<const char*>(wrist6.data()), 6 * sizeof(double));
          for (const auto& fr : win) smpl_dump_.write(reinterpret_cast<const char*>(&fr.t_mono), sizeof(double));
          for (const auto& fr : win) smpl_dump_.write(reinterpret_cast<const char*>(fr.joints.data()), 72 * sizeof(float));
          for (const auto& fr : win) smpl_dump_.write(reinterpret_cast<const char*>(fr.quat_wxyz.data()), 4 * sizeof(double));
          smpl_dump_.write(reinterpret_cast<const char*>(smpl_obs.data()), smpl_obs.size() * sizeof(float));
        }
      }
    } else if (token_active) {
      tok_obs.assign(680, 0.0f);
      const auto tok = token_source_->LatestMotionToken();
      for (int i = 0; i < 64; ++i) tok_obs[i] = static_cast<float>(tok[i]);
    } else {
      tok_obs = BuildTokenizerObs(
          *ref_motion_, policy_time, rs.base_quat_wxyz, tok_ref_override);
    }
    const auto prop    = prop_buf_.GetFlat();

    // ---- Inference ---------------------------------------------------------
    std::array<double, NUM_DOFS> action_il;
    try {
      if (dual_head) {
        std::vector<float> obs;
        obs.reserve(tok_obs.size() + smpl_obs.size() + prop.size() + 1);
        obs.insert(obs.end(), tok_obs.begin(), tok_obs.end());
        obs.insert(obs.end(), smpl_obs.begin(), smpl_obs.end());
        obs.insert(obs.end(), prop.begin(), prop.end());
        obs.push_back(token_active ? 1.0f : 0.0f);
        action_il = onnx_actor_->InferObs(obs);
        if (wb_switch_t0_ >= 0.0) {
          const double ramp = std::max(1e-3, cli_.wb_switch_ramp_s);
          double a = (SteadyNow() - wb_switch_t0_) / ramp;
          if (a >= 1.0) { a = 1.0; wb_switch_t0_ = -1.0; }
          const double w = 0.5 - 0.5 * std::cos(M_PI * a);
          for (std::size_t i = 0; i < NUM_DOFS; ++i) {
            action_il[i] = w * action_il[i] + (1.0 - w) * wb_switch_from_[i];
          }
        }
      } else {
        action_il = (token_active ? token_actor_ : onnx_actor_)
                        ->Infer(tok_obs, prop);
      }
    } catch (const std::exception& e) {
      RCLCPP_FATAL(node_->get_logger(),
                   "ONNX inference failed: %s -> SAFE_HOLD", e.what());
      LatchSafeHold("onnx_failure");
      return;
    }

    // ---- Action clip (matches training-time ManagerEnvWrapper) -------------
    // IsaacLab's training wrapper applies torch.clip(env_actions, -C, C) with
    // C = config.action_clip_value (default 20.0) before stepping the sim.
    // The clipped value is then what env.action_manager.action returns, which
    // is what the ``last_action_wo_hand`` proprioception term sees on the
    // next tick. Deploying without this clip means: (a) target_pos_mj can
    // explode by O(C * action_scale) when the policy saturates, which the
    // ``--max-target-dev`` clamp truncates -- silently breaking the
    // physics<->command relationship -- and (b) ``last_action_il_`` drifts
    // outside the training distribution, accelerating divergence. See the
    // 16k checkpoint smoke test on 2026-04-22 for the gory details.
    // ---- NaN/finite guard (2026-08-04, audit #17: NEITHER deploy had one;
    // std::clamp propagates NaN so a single bad ONNX output could reach the
    // bus as a NaN target). Any non-finite action -> SAFE_HOLD.
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      if (!std::isfinite(action_il[i])) {
        RCLCPP_FATAL(node_->get_logger(),
                     "non-finite policy action at IL dof %zu — SAFE_HOLD", i);
        recorder_.Trigger("safe_hold");
        state_.store(State::SAFE_HOLD);
        return;
      }
    }

    // Snapshot BEFORE the clip: post-clip we can only ever see what was
    // ALLOWED, never what the policy DEMANDED.
    const std::array<double, NUM_DOFS> action_pre_clip_il = action_il;
    std::size_t clipped_joint_count = 0;
    double max_pre_clip = 0.0;
    if (cli_.action_clip > 0.0) {
      const double clip = cli_.action_clip;
      for (std::size_t i = 0; i < NUM_DOFS; ++i) {
        const double a = action_il[i];
        if (std::abs(a) > max_pre_clip) max_pre_clip = std::abs(a);
        if (a >  clip) { action_il[i] =  clip; ++clipped_joint_count; }
        else if (a < -clip) { action_il[i] = -clip; ++clipped_joint_count; }
      }
      if (clipped_joint_count > 0) {
        ++action_clip_tick_count_;
        action_clip_max_pre_clip_ = std::max(action_clip_max_pre_clip_, max_pre_clip);
      }
    }
    last_action_il_ = action_il;

    // ---- Optional one-shot obs dump for offline parity-check ---------------
    // Dumps tokenizer, proprioception, raw policy output, and the robot state
    // that produced them on the first CONTROL tick, then asks the node to
    // shut down. The companion script
    //   gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py
    // diffs this against /tmp/x2_step0_isaaclab_lastpt.pt.
    if (!cli_.obs_dump_path.empty() && !obs_dumped_) {
      DumpObsBlob(cli_.obs_dump_path, tok_obs, prop, action_il, rs, policy_time);
      obs_dumped_ = true;
      RCLCPP_WARN(node_->get_logger(),
                  "--obs-dump fired; wrote %zu bytes to %s; requesting shutdown.",
                  static_cast<std::size_t>(
                    tok_obs.size() * sizeof(float)
                    + prop.size() * sizeof(float)
                    + NUM_DOFS * sizeof(double)),
                  cli_.obs_dump_path.c_str());
      // Returning here means we don't push any SafeCommand for this tick;
      // the writer keeps repeating the previous one (which in --dry-run mode
      // has zero gains, so the robot stays passive). We then ask the node
      // to shut down at the next executor turn.
      rclcpp::shutdown();
      return;
    }

    // ---- Action -> MJ-ordered PD target ------------------------------------
    std::array<double, NUM_DOFS> target_pos_mj{};
    for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
      const std::size_t il = static_cast<std::size_t>(mujoco_to_isaaclab[mj]);
      target_pos_mj[mj] = default_angles[mj] + action_il[il] * x2_action_scale[mj];
    }

    // ---- Wrist bypass: honour IK reference for the 4 broken wrist DOFs -----
    // CLI-gated; preserves sim-to-real fidelity on the motion-file replay
    // path when --wrist-bypass=off (default). Override sits BEFORE the
    // safety stack so soft-start blend, --max-target-dev clamp, and the
    // tilt-trip force-to-default branch all apply uniformly. Loop body
    // lives in include/wrist_bypass.hpp so the unit test can exercise it
    // without a ROS 2 / ONNX runtime in scope.
    if (cli_.wrist_bypass == CliArgs::WristBypass::Ik
        && zmq_pose_source_ != nullptr
        && zmq_pose_source_->has_body_reference()) {
      auto ref_frame = zmq_pose_source_->Sample(policy_time);
      if (dual_head && token_active) {
        // Operator wrists straight from the SMPL stream (sender's SMPL
        // wrist mapping), clipped to the X2 ranges; no kplanner hop.
        const auto w = wb_hold_ ? wb_hold_wrist_ : smpl_source_->LatestWristPr();
        ref_frame.joint_pos_mj[20] = std::clamp(w[0], -0.558, 0.558);
        ref_frame.joint_pos_mj[21] = std::clamp(w[1], -1.571, 0.724);
        ref_frame.joint_pos_mj[27] = std::clamp(w[2], -0.558, 0.558);
        ref_frame.joint_pos_mj[28] = std::clamp(w[3], -0.724, 1.571);
      }
      const double max_delta = ApplyWristBypass(target_pos_mj, ref_frame);
      if (dual_head && token_active && smpl_source_->HasWristYaw()) {
        // Pronation straight from the operator as well (wrist_yaw, MJ
        // 19/26): under the smpl head the operator IS the reference.
        const auto wy = wb_hold_ ? wb_hold_wrist_yaw_ : smpl_source_->LatestWristYaw();
        target_pos_mj[19] = std::clamp(wy[0], -2.556, 2.556);
        target_pos_mj[26] = std::clamp(wy[1], -2.556, 2.556);
      }
      ++wrist_bypass_tick_count_;
      if (max_delta > wrist_bypass_max_delta_) wrist_bypass_max_delta_ = max_delta;
    }

    // Head bypass (mirror of the wrist bypass; see head_bypass.hpp):
    // SONIC saturates head tracking at ~+-10 deg, so the pad head-look
    // reference drives the two head DOFs directly. Sits BEFORE the
    // safety stack for uniform ramp/clamp/tilt-trip treatment.
    if (cli_.head_bypass == CliArgs::HeadBypass::Ref
        && zmq_pose_source_ != nullptr
        && zmq_pose_source_->has_body_reference()) {
      const auto ref_frame = zmq_pose_source_->Sample(policy_time);
      const double max_delta = ApplyHeadBypass(target_pos_mj, ref_frame);
      ++head_bypass_tick_count_;
      if (max_delta > head_bypass_max_delta_) head_bypass_max_delta_ = max_delta;
    }

    // ---- Motor thermal monitor (warn-only; audit #2).
    {
      static thread_local ThermalMonitor thermal_;
      const std::string rep = thermal_.Update(rs.coil_temp_c,
                                              rs.motor_temp_c, now);
      if (!rep.empty()) {
        RCLCPP_WARN(node_->get_logger(), "THERMAL: %s", rep.c_str());
      }
    }

    // ---- Operator e-stop: polled UNCONDITIONALLY before the state switch
    // (see above). Nothing to do here.

    // ---- Joint-velocity watchdog (G1-reference parity; added 2026-08-04).
    // g1_deploy_onnx_ref checks every measured |dq| against 35 rad/s each
    // tick and stops on breach; every G1 stop path ends in the Kp=0/Kd=8
    // damping command. Here a breach force-trips the tilt watchdog with
    // the stage-2 flag, so ApplySafetyStack engages pure damping on THIS
    // tick — a joint at such speed means the fall or a policy divergence
    // is already violent, and fighting it at kp~99 is the damage mode.
    // 35 rad/s is far above any trained motion (fastest corpus joints
    // ~14 rad/s; X2 wrist hardware limit 20.9) so false trips require
    // genuinely impossible motion or sensor garbage.
    if (!watchdog_.Tripped()) {
      for (std::size_t i = 0; i < NUM_DOFS; ++i) {
        const double adq = std::abs(rs.joint_vel_mj[i]);
        if (adq > cli_.joint_vel_trip_rad_s) {
          watchdog_.ForceTripVelocity(static_cast<int>(i), adq);
          break;
        }
      }
    }

    // ---- Safety stack ------------------------------------------------------
    // Use the per-DOF clamp array synthesised in the constructor from
    // --max-target-dev plus the --max-target-dev-{leg,waist,arm,head}
    // overrides, so e.g. arms can take 1.50 rad while legs stay at 0.30
    // rad on the same tick. Also pass the per-DOF effective PD (trained
    // kps/kds * global+family --kp-scale/--kd-scale trims) so the safety
    // stack publishes the deployment-bumped gains rather than the raw
    // training PD. Both arrays are computed once at startup; the hot
    // path is unchanged.
    SafeCommand sc = ApplySafetyStack(target_pos_mj, grav[2],
                                      ramp_, watchdog_, cli_.dry_run, now,
                                      max_target_dev_per_dof_,
                                      kps_scaled_, kds_scaled_);

    // ---- Output-side target LPF (real-deploy only; bypassed by default) ----
    // The EMA runs strictly AFTER the safety stack, so:
    //   * --max-target-dev clamps still bound the PRE-filter target, and the
    //     filter then attenuates further (cannot make the published target
    //     exceed the clamp);
    //   * --obs-dump returned earlier in this tick (line ~672 above) before
    //     `target_pos_mj` was even computed, so the dumped raw policy output
    //     is identical with or without --target-lpf-hz set;
    //   * RAMP_OUT and SAFE_HOLD bypass: those states already produce a
    //     deliberately-shaped trajectory we don't want to attenuate.
    if (target_lpf_any_active_
        && state_.load() == State::CONTROL
        && !sc.tilt_trip) {
      if (!target_lpf_initialized_) {
        // First CONTROL tick: seed the EMA state to the current target so
        // we don't bias toward zero or any prior value.
        target_lpf_state_ = sc.target_pos_mj;
        target_lpf_initialized_ = true;
      } else {
        // Per-DOF EMA. alpha == 0 means "passthrough on this joint" -- we
        // still write the raw target into target_lpf_state_ so that if a
        // per-group LPF is toggled on later (not currently supported, but
        // cheap to keep correct) the state isn't stale.
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          const double a = target_lpf_alpha_per_dof_[i];
          if (a > 0.0) {
            target_lpf_state_[i] =
                a * sc.target_pos_mj[i] + (1.0 - a) * target_lpf_state_[i];
          } else {
            target_lpf_state_[i] = sc.target_pos_mj[i];
          }
        }
        sc.target_pos_mj = target_lpf_state_;
      }
    }

    if (sc.tilt_trip && state_.load() == State::CONTROL) {
      RCLCPP_FATAL(node_->get_logger(), "%s -> SAFE_HOLD", sc.reason.c_str());
      // Latch the safety command so the writer keeps holding it forever.
      {
        std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
        latest_cmd_ = sc;
      }
      latest_cmd_ready_.store(true, std::memory_order_release);
      recorder_.Trigger("safe_hold");
      state_.store(State::SAFE_HOLD);
      return;
    }

    // ---- Publish to writer slot --------------------------------------------
    {
      std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
      latest_cmd_ = sc;
    }
    latest_cmd_ready_.store(true, std::memory_order_release);

    // ---- Log ---------------------------------------------------------------
    logger_.Log(now, rs.joint_pos_mj, rs.joint_vel_mj,
                rs.base_quat_wxyz, rs.base_ang_vel,
                action_il, sc);

      // Black box: the WHOLE chain for this tick -- raw demand, post-clip,
      // pre-clamp target, and the filtered target that actually goes out on
      // ZMQ. Cheap (~160 float stores) and a no-op unless enabled.
      // Fast ring from the snapshot OnControl ALREADY took (line ~2099):
      // no extra lock, no writer-thread work.
      if (recorder_.enabled()) {
        const double ages[5] = {
          now - rs.leg_stamp_s,  now - rs.waist_stamp_s,
          now - rs.arm_stamp_s,  now - rs.head_stamp_s,
          now - rs.imu_stamp_s };
        // PD TORQUE DEMAND, not applied effort. The aimdk state message carries
        // no torque field and PublishGroup sends j.effort = 0.0 -- the firmware
        // closes the PD loop, so nothing on this side ever sees real torque.
        // What we CAN record exactly is what we asked the firmware to produce,
        // using the same scaled gains we published: tau = kp*(target-pos) - kd*vel.
        std::array<double, NUM_DOFS> pd_demand{};
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          pd_demand[i] = sc.stiffness_mj[i] * (sc.target_pos_mj[i] - rs.joint_pos_mj[i])
                       - sc.damping_mj[i]   * rs.joint_vel_mj[i];
        }
        recorder_.RecordFast(now, rs.joint_pos_mj.data(), rs.joint_vel_mj.data(),
                             sc.target_pos_mj.data(), pd_demand.data(),
                             rs.motor_temp_c.data(), rs.base_quat_wxyz.data(),
                             rs.base_ang_vel.data(), ages);
      }
      // The reference the policy is being ASKED to track. This is the
      // discriminator for the waist question: without it we cannot tell an
      // out-of-range reference from a policy generating its own excursion.
      // Sample() is an interpolation over a preloaded trajectory; guarded so a
      // disabled recorder costs nothing.
      std::array<double, NUM_DOFS> ref_pos_mj{};
      if (recorder_.enabled() && ref_motion_ != nullptr) {
        ref_pos_mj = ref_motion_->Sample(policy_time).joint_pos_mj;
      }
      recorder_.RecordSlow(now, action_pre_clip_il.data(), action_il.data(),
                           target_pos_mj.data(), sc.target_pos_mj.data(),
                           ref_pos_mj.data(),
                           static_cast<float>(sc.ramp_alpha),
                           static_cast<float>(clipped_joint_count) / NUM_DOFS,
                           static_cast<int>(state_.load()),
                           sc.tilt_trip,
                           (zmq_pose_source_ != nullptr
                            && zmq_pose_source_->EstopRequested()),
                           (zmq_pose_source_ != nullptr
                            ? static_cast<float>(std::max(0.0,
                                now - zmq_pose_source_->LastReceivedMonotonicS()))
                            : -1.0f));

    // ---- Optional ZMQ telemetry (matches dump_x2_debug.py expectations) ----
    if (zmq_debug_pub_) {
      PublishDebugFrame(now, rs, action_il, sc, policy_time);
    }

    // ---- Periodic status ---------------------------------------------------
    if (++control_tick_ % 50 == 0) {
      const int32_t mode_now = mc_action_mode_.load(std::memory_order_acquire);
      const double  pose_ref_age =
          (pose_ref_watchdog_active_ && zmq_pose_source_ != nullptr)
              ? pose_ref_watchdog_.LatestAgeS()
              : -1.0;
      RCLCPP_INFO(node_->get_logger(),
                  "CONTROL tick=%lu policy_t=%.2fs alpha=%.2f grav_z=%+.2f "
                  "act_clip_ticks=%lu max_pre_clip=%.2f "
                  "wrist_bypass_ticks=%lu wrist_bypass_max_dev_rad=%.3f "
                  "head_bypass_ticks=%lu head_bypass_max_dev_rad=%.3f "
                  "pose_ref_age=%.3fs mc_mode=%d",
                  static_cast<unsigned long>(control_tick_),
                  policy_time, sc.ramp_alpha, grav[2],
                  static_cast<unsigned long>(action_clip_tick_count_),
                  action_clip_max_pre_clip_,
                  static_cast<unsigned long>(wrist_bypass_tick_count_),
                  wrist_bypass_max_delta_,
                  static_cast<unsigned long>(head_bypass_tick_count_),
                  head_bypass_max_delta_,
                  pose_ref_age, static_cast<int>(mode_now));
    }
  }

  // ----- 500 Hz writer loop ------------------------------------------------
  void OnWriter()
  {
    const State cur = state_.load();
    // OBSERVE-ONLY: mute the bus unconditionally, before any state logic.
    // Placed FIRST so no state -- CONTROL, SAFE_HOLD, RAMP_OUT, HOLD_FOR_MC
    // or a future one -- can route around it. MC keeps the robot; we only
    // watch. Everything downstream of the policy (clamps, LPF, CSV logging)
    // still runs, so target_pos.csv records what WOULD have been commanded.
    if (cli_.no_publish) {
      return;
    }
    if (cur == State::STANDBY
        || cur == State::INIT
        || cur == State::WAIT_FOR_CONTROL) {
      // Don't publish anything in pre-control states. The robot's last-good
      // command (from whatever was running before us) keeps the joints held.
      // STANDBY in particular MUST be silent: bash launches us before
      // stop_app, so MC may still be publishing on this bus -- adding our
      // commands would make the firmware see a dual-publisher fight.
      return;
    }
    // SAFE_IDLE, SAFE_HOLD, RAMP_OUT, HOLD_FOR_MC, CONTROL all publish.
    // SAFE_IDLE in particular MUST keep the writer alive at 500 Hz so the
    // body stays under torque while the operator wire is dark.
    // Skip publishing until OnControl (or RAMP_OUT/SAFE_HOLD) has latched
    // a real command. Without this guard, the first ~15 ms after WAIT ->
    // CONTROL would publish a default-zero command (kp=0, kd=0, target=0),
    // which on the sim bridge cancels the pre-handoff freeze prematurely
    // and lets gravity perturb joint state before the policy ever sees it.
    // On hardware the symptom would be different (a no-op zero-gain PD
    // command) but it's still wrong: MC's last-good command is what should
    // hold the joints during this sub-tick window, not our zeroed one.
    if (!latest_cmd_ready_.load(std::memory_order_acquire)) {
      return;
    }
    SafeCommand sc;
    {
      std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
      sc = latest_cmd_;
    }
    // target_vel = zeros (the policy spec is pos-only; firmware integrates).
    static const std::array<double, NUM_DOFS> kZeroVel{};
    // NOTE: the black box does NOT sample here. The 500 Hz writer stays a
    // pure re-publish -- see FlightRecorder::kFastHz for why.
    aimdk_io_->PublishCommand(sc.target_pos_mj, kZeroVel,
                              sc.stiffness_mj, sc.damping_mj);
  }

  void LatchSafeHold(const std::string& reason)
  {
    // Freeze and dump the black box FIRST, so the pre-trigger history is the
    // run-up to the fault rather than the damping that follows it. Serialises
    // on a detached thread; the 500 Hz writer keeps feeding damping meanwhile.
    recorder_.Trigger(reason);
    SafeCommand sc;
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      sc.target_pos_mj[i] = default_angles[i];
      sc.stiffness_mj[i]  = cli_.dry_run ? 0.0 : kps_scaled_[i];
      sc.damping_mj[i]    = cli_.dry_run ? 0.0 : kds_scaled_[i] * 4.0;
    }
    sc.dry_run    = cli_.dry_run;
    sc.tilt_trip  = false;
    sc.ramp_alpha = 0.0;
    sc.reason     = reason;
    {
      std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
      latest_cmd_ = sc;
    }
    latest_cmd_ready_.store(true, std::memory_order_release);
    recorder_.Trigger("safe_hold");
    state_.store(State::SAFE_HOLD);
  }

  // ─── MC mode poller (Phase 5b observability) ────────────────────────
  // Fires every cli_.mc_mode_poll_s seconds. Strictly best-effort: if MC's
  // service isn't up yet (or the call times out), we silently leave
  // mc_action_mode_ at its previous value (-1 if never seen). The deploy
  // never makes control decisions based on this value; it is only
  // surfaced on x2_debug for forensic alignment with deploy CSVs +
  // motor_monitor.jsonl (Phase 5).
  void OnMcModePoll()
  {
    if (!mc_get_action_client_) return;
    if (!mc_get_action_client_->service_is_ready()) {
      // MC service hasn't come up yet (or went away). Keep the previous
      // value; the periodic deploy status line will show -1 until it
      // does. We deliberately do NOT spam-log here -- the operator can
      // tail x2_debug or the motor_monitor JSONL for real-time visibility.
      return;
    }
    // Drain any prior async future. The async path is single-shot per
    // tick; if a request from the previous tick is still in flight we
    // skip this tick (the rclcpp executor will eventually deliver it).
    if (mc_mode_request_in_flight_) {
      if (mc_mode_pending_future_.valid()
          && mc_mode_pending_future_.wait_for(std::chrono::seconds(0))
                 == std::future_status::ready) {
        try {
          auto resp = mc_mode_pending_future_.get();
          if (resp) {
            mc_action_mode_.store(resp->info.current_action.value,
                                  std::memory_order_release);
            mc_action_status_.store(resp->info.status.value,
                                    std::memory_order_release);
          }
        } catch (const std::exception& e) {
          RCLCPP_DEBUG(node_->get_logger(),
                       "OBS: MC mode poll future threw: %s", e.what());
        }
        mc_mode_request_in_flight_ = false;
      } else {
        // Still pending; skip issuing a duplicate.
        return;
      }
    }
    auto req = std::make_shared<aimdk_msgs::srv::GetMcAction::Request>();
    req->request = aimdk_msgs::msg::CommonRequest();
    req->request.header.stamp = node_->now();
    mc_mode_pending_future_ = mc_get_action_client_->async_send_request(req).future.share();
    mc_mode_request_in_flight_ = true;
  }

  // ─── HOLD_FOR_MC support ────────────────────────────────────────────
  // RAMP_OUT calls EnterHoldForMc() once the lerp completes.
  // SAFE_HOLD -> RECOVER: snapshot the measured pose and the LATCHED gains
  // (pure damping, or the stand-pose hold profile) as the ramp start, size
  // phase B by the largest joint travel to the stand pose, and go.
  void EnterRecover(double now, const RobotState& rs) {
    recover_start_pos_ = rs.joint_pos_mj;
    {
      std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
      recover_kp_start_ = latest_cmd_.stiffness_mj;
      recover_kd_start_ = latest_cmd_.damping_mj;
    }
    double max_delta = 0.0;
    for (std::size_t i = 0; i < NUM_DOFS; ++i)
      max_delta = std::max(max_delta, std::abs(stand_pose_target_[i] - recover_start_pos_[i]));
    recover_standup_total_s_ = std::max(cli_.recover_standup_s,
                                        max_delta / std::max(cli_.recover_max_rate, 1e-3));
    recover_entry_s_         = now;
    recover_upright_since_s_ = -1.0;
    ++recover_count_;
    RCLCPP_WARN(node_->get_logger(),
                "SAFE_HOLD -> RECOVER (attempt #%d): upright and quiet for %.1f s. Phase A stiffens in "
                "place over %.1f s, phase B rises to the stand pose over %.1f s (largest joint travel "
                "%.2f rad at <= %.2f rad/s). Tilt past %.2f aborts to pure damping.",
                recover_count_, cli_.recover_dwell_s, cli_.recover_stiffen_s,
                recover_standup_total_s_, max_delta, cli_.recover_max_rate, watchdog_.FallCos());
    recorder_.Trigger("recover_start");
    state_.store(State::RECOVER);
  }

  void EnterHoldForMc(double now)
  {
    hold_for_mc_entry_s_ = now;
    hold_for_mc_entry_steady_ = std::chrono::steady_clock::now();
    mc_takeover_detected_.store(false, std::memory_order_release);
    mc_takeover_logged_ = false;
    {
      std::lock_guard<std::mutex> lk(mc_takeover_topic_mutex_);
      mc_takeover_topic_.clear();
      mc_takeover_steady_ts_ = std::chrono::steady_clock::time_point{};
    }
    if (!cli_.hold_for_mc_sentinel.empty()) {
      // Touch the sentinel so deploy_x2.sh knows the policy phase is
      // done and it can fire start_app + SetMcAction(STAND_DEFAULT).
      // Use std::ofstream so we don't require <unistd.h>; trunc-creates
      // an empty file on every entry.
      std::ofstream sentinel(cli_.hold_for_mc_sentinel, std::ios::trunc);
      if (!sentinel) {
        RCLCPP_ERROR(node_->get_logger(),
                     "HOLD_FOR_MC: failed to touch sentinel '%s' (errno "
                     "= %d). Bash won't sequence start_app; operator "
                     "will need to bring MC back manually.",
                     cli_.hold_for_mc_sentinel.c_str(), errno);
      } else {
        sentinel << "hold_for_mc " << now << "\n";
      }
    }
    state_.store(State::HOLD_FOR_MC);
  }

  void ClearHoldForMcSentinel()
  {
    if (cli_.hold_for_mc_sentinel.empty()) return;
    // Best-effort cleanup; if remove fails there's nothing meaningful
    // we can do (the bash cleanup trap also rm -f's it as a backstop).
    std::remove(cli_.hold_for_mc_sentinel.c_str());
  }

  // Called from the constructor when --hold-for-mc-timeout-s > 0.
  // Subscribes to /aima/hal/joint/leg/command and /aima/hal/joint/waist/command
  // with ignore_local_publications=true, so OUR command writer's
  // traffic never trips the detector. The first non-self message on
  // either topic flips mc_takeover_detected_, which the HOLD_FOR_MC
  // case in OnControl reads on the next tick to exit cleanly.
  void InitMcTakeoverDetectors()
  {
    rclcpp::SubscriptionOptions opts;
    opts.ignore_local_publications = true;
    // Match the QoS of the joint command bus exactly. AimdkIo publishes
    // on these topics with rclcpp::SensorDataQoS() (best-effort, depth=10);
    // MC's HAL on PC1 also publishes best-effort. A default rclcpp::QoS()
    // here is RELIABLE, which is INCOMPATIBLE with MC's best-effort
    // publisher -- DDS refuses to match and our subscriber receives zero
    // messages from MC. That is the root cause of the ~1.7 s dual-
    // publisher whir at the end of the 2026-05-03 run: the takeover
    // detector never fired, deploy fell back to the slower exit-sentinel
    // path, and MC + deploy fought on the bus for the full duration of
    // bash's PASSIVE -> JOINT -> STAND escalation. With matched QoS the
    // detector callback fires sub-ms after MC's first publish.
    // ignore_local_publications still suppresses self-traffic at the
    // GID level, independent of QoS.
    auto qos = rclcpp::SensorDataQoS();
    auto on_takeover =
        [this](const std::string& topic) {
          // Capture the moment MC took the bus at the highest resolution
          // we can, BEFORE acquiring any locks, so the timestamp is as
          // close as possible to "DDS callback fired". steady_clock is
          // monotonic + ns-precision; we publish the delta vs HOLD_FOR_MC
          // entry so the operator can correlate against audible whirring.
          const auto now_steady = std::chrono::steady_clock::now();
          bool first = false;
          {
            std::lock_guard<std::mutex> lk(mc_takeover_topic_mutex_);
            if (mc_takeover_topic_.empty()) {
              mc_takeover_topic_ = topic;
              mc_takeover_steady_ts_ = now_steady;
              first = true;
            }
          }
          mc_takeover_detected_.store(true, std::memory_order_release);
          // Touch the early-signal sentinel ONCE on the first publish so
          // bash can start the JOINT_DEFAULT escalation immediately rather
          // than polling MC's mode service (which lags MC's actual first
          // publish by ~0.5-0.8 s). The std::ofstream truncate is fast (a
          // couple of ms even on EXT4); we do it inside the DDS callback
          // because the latency from this point to bash seeing the file
          // matters: every ms we shave here is a ms less of MC-PASSIVE +
          // deploy dual-publisher whir. Best-effort -- if the open fails
          // (e.g. dir missing), bash falls back to the mc_get_action poll
          // path automatically.
          if (first && !cli_.mc_first_publish_sentinel.empty()) {
            std::ofstream s(cli_.mc_first_publish_sentinel, std::ios::trunc);
            if (s) s << "first_publish " << topic << "\n";
          }
        };
    mc_takeover_leg_sub_ =
        node_->create_subscription<aimdk_msgs::msg::JointCommandArray>(
            "/aima/hal/joint/leg/command", qos,
            [on_takeover](aimdk_msgs::msg::JointCommandArray::ConstSharedPtr) {
              on_takeover("/aima/hal/joint/leg/command");
            },
            opts);
    mc_takeover_waist_sub_ =
        node_->create_subscription<aimdk_msgs::msg::JointCommandArray>(
            "/aima/hal/joint/waist/command", qos,
            [on_takeover](aimdk_msgs::msg::JointCommandArray::ConstSharedPtr) {
              on_takeover("/aima/hal/joint/waist/command");
            },
            opts);
    RCLCPP_INFO(node_->get_logger(),
                "HANDOFF: MC-takeover detectors armed on "
                "/aima/hal/joint/leg/command and "
                "/aima/hal/joint/waist/command "
                "(ignore_local_publications=true).");
  }

  // -------------------------------------------------------------------------
  rclcpp::Node::SharedPtr node_;
  CliArgs                 cli_;

  std::unique_ptr<AimdkIo>          aimdk_io_;
  std::unique_ptr<ReferenceMotion>  ref_motion_;
  std::unique_ptr<OnnxActor>        onnx_actor_;
  ProprioceptionBuffer              prop_buf_;
  std::array<double, NUM_DOFS>      last_action_il_{};

  SoftStartRamp                     ramp_;
  TiltWatchdog                      watchdog_;
  // Split-topology pose-ref starvation watchdog. Active only when
  // ``pose_ref_watchdog_active_`` is true (set in the ctor from CLI).
  // OnControl checks it on every CONTROL tick using the age computed
  // from the ZmqPoseInputSource's monotonic last-recv timestamp; a trip
  // transitions CONTROL -> SAFE_IDLE and latches a default-pose hold
  // with 4x kd. Exit (back to CONTROL) requires both ReadyToResume()
  // and a recent operator chord on ``zmq_resume_sub_``.
  PoseRefStarvationWatchdog         pose_ref_watchdog_;
  bool                              pose_ref_watchdog_active_ = false;
  // Operator-side resume chord SUB. Constructed when the pose-ref
  // watchdog is active. Holds its own background thread + ZMQ socket;
  // tears down cleanly in the destructor via std::unique_ptr.
  std::unique_ptr<ZmqResumeSubscriber> zmq_resume_sub_;
  // Most recent SAFE_IDLE entry time (steady_clock seconds). Used by
  // the periodic status line and the x2_debug ``in_safe_idle`` ms
  // counter. -1 = never been in SAFE_IDLE (or already exited).
  double                            safe_idle_entry_s_ = -1.0;
  // Latched SafeCommand that the writer keeps publishing while in
  // SAFE_IDLE. Built once at entry to avoid recomputing default_angles
  // + 4x kd on every 500 Hz tick.
  SafeCommand                       safe_idle_cmd_;

  // MC mode poller (Phase 5b observability). Background timer fires
  // every cli_.mc_mode_poll_s seconds; mc_action_mode_ holds the latest
  // McAction.value enum (-1 = unknown). Published on x2_debug.
  rclcpp::Client<aimdk_msgs::srv::GetMcAction>::SharedPtr
                                    mc_get_action_client_;
  rclcpp::TimerBase::SharedPtr      mc_mode_poll_timer_;
  std::shared_future<
      aimdk_msgs::srv::GetMcAction::Response::SharedPtr>
                                    mc_mode_pending_future_;
  bool                              mc_mode_request_in_flight_ = false;
  std::atomic<int32_t>              mc_action_mode_{-1};
  std::atomic<int32_t>              mc_action_status_{-1};

  DeployLogger                      logger_;
  agi_x2::FlightRecorder            recorder_;

  // Per-DOF max_target_dev clamp synthesised once from the global +
  // per-group CLI scalars. Indexed in MuJoCo joint order. Entry <= 0 ->
  // no clamp on that joint. See BuildMaxTargetDevPerDof above.
  std::array<double, NUM_DOFS>      max_target_dev_per_dof_{};

  // Effective per-DOF PD synthesised once from kps[] / kds[] (constexpr
  // training values from policy_parameters.hpp) multiplied by the
  // global+per-family --kp-scale / --kd-scale CLI trims. These ARE the
  // gains the safety stack will publish on the bus -- the scaling is
  // applied at startup, not per-tick, so the runtime cost is one
  // multiply per joint at boot. Indexed in MuJoCo joint order.
  std::array<double, NUM_DOFS>      kps_scaled_{};
  std::array<double, NUM_DOFS>      kds_scaled_{};

  rclcpp::TimerBase::SharedPtr      control_timer_;
  rclcpp::TimerBase::SharedPtr      writer_timer_;

  std::mutex                        latest_cmd_mutex_;
  SafeCommand                       latest_cmd_;
  // True once OnControl has populated latest_cmd_ at least once (or
  // RAMP_OUT/SAFE_HOLD has explicitly latched one). Until this flips,
  // OnWriter must NOT publish, otherwise the bridge sees a default-zero
  // JointCommand (kp=0, kd=0, target=0), which (a) silently flips its
  // ``_first_command_received`` flag and unfreezes physics, and (b)
  // perturbs joint state in the ~15 ms between WAIT->CONTROL and the first
  // OnControl tick. That produced an apples-to-oranges first-tick obs
  // (e.g. cpp jvel[5] = -0.0077 vs python jvel[5] = -0.0265 for the same
  // motion frame) and was the source of the parity-profile fall-at-~5 s.
  // See gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py:_sim_step_once
  // for the freeze counterpart on the bridge side.
  std::atomic<bool>                 latest_cmd_ready_{false};

  std::atomic<State>                state_{State::INIT};
  // Latched true the first time OnControl runs in STANDBY -- prevents
  // the ready-sentinel from being touched + logged on every tick.
  bool                              standby_ready_logged_ = false;
  // Latch guard: SAFE_HOLD publishes the pure-damping command ONCE on an
  // operator e-stop, not every tick -- a stable command for the 500 Hz
  // writer and a single FATAL line in the log.
  bool                              safe_hold_damped_ = false;
  double                            safe_hold_down_since_s_ = -1.0;  // SAFE_HOLD down-detect dwell start
  // SAFE_HOLD -> RECOVER assisted stand-up state.
  double                            recover_upright_since_s_ = -1.0;  // gate dwell start
  double                            recover_entry_s_         = -1.0;
  double                            recover_standup_total_s_ = 4.0;   // phase B length for this attempt
  std::array<double, NUM_DOFS>      recover_start_pos_{};
  std::array<double, NUM_DOFS>      recover_kp_start_{};
  std::array<double, NUM_DOFS>      recover_kd_start_{};
  int                               recover_count_ = 0;
  bool                              recover_blocked_logged_ = false;
  double                            control_entry_s_     = -1.0;
  double                            autostart_target_s_  = -1.0;
  std::uint64_t                     control_tick_        = 0;

  // RAMP_OUT bookkeeping: ramp_out_entry_s_ is the steady-clock time we
  // entered RAMP_OUT, and ramp_out_start_pos_ is the target_pos_mj snapshot
  // we took at that moment. The RAMP_OUT case in OnControl lerps from
  // ramp_out_start_pos_ -> stand_pose_target_ over cli_.return_seconds.
  double                            ramp_out_entry_s_    = -1.0;
  std::array<double, NUM_DOFS>      ramp_out_start_pos_{};

  // End-of-run handoff target (MC's STAND_DEFAULT pose). Initialised in
  // the constructor: defaults to default_angles + deploy-mode kp/kd, and
  // overwritten with the captured YAML values when --stand-default-pose
  // is provided. Used by RAMP_OUT (lerp toward) and HOLD_FOR_MC (publish
  // statically).
  std::array<double, NUM_DOFS>      stand_pose_target_{};
  std::array<double, NUM_DOFS>      stand_pose_stiffness_{};
  std::array<double, NUM_DOFS>      stand_pose_damping_{};

  // HOLD_FOR_MC bookkeeping. hold_for_mc_entry_s_ is the steady-clock
  // time we entered the state. mc_takeover_detected_ is flipped by the
  // detection subscribers below the *moment* an external publisher (MC)
  // shows up on /aima/hal/joint/{leg,waist}/command -- we then exit
  // cleanly on the next OnControl tick. mc_takeover_topic_ records which
  // topic saw the first non-self publish, purely for the log line.
  // hold_for_mc_entry_steady_ pairs with mc_takeover_steady_ts_ to
  // produce the ms-precision delta in the takeover log line, independent
  // of the ROS-clock rounding in hold_for_mc_entry_s_.
  double                            hold_for_mc_entry_s_ = -1.0;
  std::chrono::steady_clock::time_point hold_for_mc_entry_steady_{};
  std::atomic<bool>                 mc_takeover_detected_{false};
  // True after we've logged the "first MC publish observed" line once,
  // so we don't spam the log on every tick once MC is back on the bus.
  bool                              mc_takeover_logged_  = false;
  std::string                       mc_takeover_topic_;
  // ns-precision timestamp captured INSIDE the DDS callback (before any
  // locks), so the trace shows the actual moment MC took the bus rather
  // than the next OnControl tick (which can lag up to ~20 ms at 50 Hz).
  std::chrono::steady_clock::time_point mc_takeover_steady_ts_{};
  std::mutex                        mc_takeover_topic_mutex_;
  // Subscribers carry SubscriptionOptions::ignore_local_publications=true
  // so OUR command writer's traffic does not trip the detector. Kept as
  // members so they outlive the constructor.
  rclcpp::Subscription<aimdk_msgs::msg::JointCommandArray>::SharedPtr
      mc_takeover_leg_sub_;
  rclcpp::Subscription<aimdk_msgs::msg::JointCommandArray>::SharedPtr
      mc_takeover_waist_sub_;

  // Output-side target LPF state. target_lpf_alpha_per_dof_ is computed
  // once in Run() from (cli_.target_lpf_hz, cli_.target_lpf_hz_{leg,waist,
  // arm,head}) at the OnControl rate (50 Hz) by BuildTargetLpfAlphaPerDof,
  // which encodes the per-group inherit/disable/explicit semantics. A slot
  // with alpha == 0 is in passthrough mode -- the OnControl hot path
  // branches on that and skips the EMA math entirely, so disabling the LPF
  // on a group truly costs nothing. ``target_lpf_any_active_`` is the
  // logical OR over the array; when it's false the whole filter block is
  // skipped (zero per-DOF cost on the historical default cli config).
  // target_lpf_initialized_ is reset implicitly by being default-false at
  // node startup; we also re-seed on the FIRST CONTROL tick (see OnControl)
  // so post-RAMP_OUT/SAFE_HOLD restarts (if we ever support them) behave.
  std::array<double, NUM_DOFS>      target_lpf_alpha_per_dof_{};
  bool                              target_lpf_any_active_    = false;
  bool                              target_lpf_initialized_   = false;
  std::array<double, NUM_DOFS>      target_lpf_state_{};

  // Cumulative count of ticks where at least one joint hit the
  // ``--action-clip`` symmetric limit, plus the largest |action_il| we've
  // seen pre-clip across the whole run. Reported on the periodic status
  // line so the operator can tell at a glance whether the policy is
  // saturated (large numbers = bad: see action_clip explanation above).
  std::uint64_t                     action_clip_tick_count_   = 0;
  double                            action_clip_max_pre_clip_ = 0.0;

  // Wrist-bypass diagnostics. ``wrist_bypass_tick_count_`` increments once
  // per tick on which the override actually fired (i.e. --wrist-bypass=ik
  // AND a body-bearing ZMQ frame was available). ``wrist_bypass_max_delta_``
  // tracks the largest |policy_target - ik_target| across the bypassed
  // wrist DOFs over the whole run, so the operator can see at a glance how
  // hard SONIC is being overruled (large numbers are normal -- they're
  // exactly why the bypass exists). Both reported on the periodic status
  // line. Stay at 0/0 when --wrist-bypass=off.
  std::uint64_t                     wrist_bypass_tick_count_  = 0;
uint64_t head_bypass_tick_count_ = 0;
  double                            wrist_bypass_max_delta_   = 0.0;
  double head_bypass_max_delta_ = 0.0;

  // Set after --obs-dump fires so we don't accidentally dump a second time
  // if the executor manages to schedule another OnControl before shutdown.
  bool                              obs_dumped_          = false;

  // Optional VLA / ZMQ helpers. ``zmq_pose_source_`` is an observer pointer
  // owned by ``ref_motion_`` -- keeps us from downcasting on every tick
  // when we want hand-joint readback. ``zmq_debug_pub_`` is the per-tick
  // x2_debug PUB sink, bound when --zmq-debug-port > 0. Both default-null;
  // the legacy motion-file path leaves them empty.
  ZmqPoseInputSource*                       zmq_pose_source_ = nullptr;
  // whole-body overlay (hybrid): token graph + its own stream from the laptop
  std::unique_ptr<OnnxActor>                token_actor_;
  // ---- native dual-head whole-body (2026-09-04) --------------------------
  std::unique_ptr<SmplIntentSource>         smpl_source_;
  std::string                               smpl_ori_mode_ = "full";
  std::optional<std::array<double, 4>>      wb_yaw_align_;      // Rz(robot-human) at engage
  bool                                      wb_want_prev_ = false;
  bool                                      wb_hold_ = false;   // layer A: smpl head on the latched window
  std::optional<std::array<double, NUM_DOFS>> wb_hold_ref_;     // pose-ref joints 0.25 s into the hold
  double                                    wb_hold_t0_ = 0.0;
  double                                    wb_warmup_until_ = 0.0;
  double                                    wb_switch_t0_ = -1.0;
  std::array<double, NUM_DOFS>              wb_switch_from_{};
  double                                    wb_engaged_dbg_ = -1.0;   // -1 = not dual-head (kplanner ignores)
  std::array<double, 4>                     wb_hold_wrist_{};          // wrists latched at WB HOLD entry
  std::array<double, 2>                     wb_hold_wrist_yaw_{};      // wrist yaw latched at WB HOLD entry
  std::optional<std::array<SmplFrame, SMPL_WINDOW>> wb_hold_win_;      // SMPL window frozen at WB HOLD entry
  std::ofstream                             smpl_dump_;         // --smpl-obs-dump
  std::unique_ptr<ZmqPoseInputSource>       token_source_owned_;
  ZmqPoseInputSource*                       token_source_ = nullptr;
  bool                                      token_active_prev_ = false;
  // CONTROL staleness debounce (see guard above): consecutive stale ticks.
  int                                       stale_state_ticks_ = 0;
  std::unique_ptr<ZmqDebugPublisher>        zmq_debug_pub_;
  // Cumulative count of x2_debug frames sent (or attempted -- ZMQ HWM may
  // drop). Reported on the periodic status line so the operator can see
  // the wire is alive without tailing a separate dump_x2_debug.py.
  std::uint64_t                             zmq_debug_frames_published_ = 0;

  // Build and send one x2_debug frame for the current control tick.
  // Mirrors the schema documented in
  // ``docs/source/references/x2_zmq_protocol.md`` so dump_x2_debug.py can
  // decode without out-of-band schema knowledge. Hand-joint slots are
  // populated from ``zmq_pose_source_->LatestHandJoints()`` when the
  // ZMQ source is active, else left as zeros.
  void PublishDebugFrame(double                                  now,
                         const RobotState&                       rs,
                         const std::array<double, NUM_DOFS>&     action_il,
                         const SafeCommand&                      sc,
                         double                                  policy_time)
  {
    if (!zmq_debug_pub_) return;
    static_assert(NUM_DOFS == 31, "x2_debug schema assumes 31 body DOFs");

    // Compose stable per-call buffers so PackedField::data pointers stay
    // valid until SendFrame() copies them into the ZMQ message.
    double now_f64        = now;
    double policy_time_f64 = policy_time;
    int64_t tick_i64      = static_cast<int64_t>(control_tick_);

    std::array<double, NUM_DOFS> body_q       = rs.joint_pos_mj;
    std::array<double, NUM_DOFS> body_dq      = rs.joint_vel_mj;
    std::array<double, 4>        base_quat    = rs.base_quat_wxyz;
    std::array<double, 3>        base_ang_vel = rs.base_ang_vel;
    std::array<double, NUM_DOFS> last_action  = sc.target_pos_mj;

    // Hand-joint readback (zeros when no ZMQ source is wired).
    std::array<double, DEFAULT_HAND_DOF_PER_SIDE> left_hand{};
    std::array<double, DEFAULT_HAND_DOF_PER_SIDE> right_hand{};
    int64_t hand_frame_idx = -1;
    if (zmq_pose_source_ != nullptr) {
      const auto snap = zmq_pose_source_->LatestHandJoints();
      if (snap.valid) {
        left_hand  = snap.left;
        right_hand = snap.right;
        hand_frame_idx = snap.frame_index;
      }
    }

    // Booleans are packed as uint8_t so they cross the wire as f32-aligned
    // payload pieces without any host endianness drama.
    std::uint8_t tilt_trip = sc.tilt_trip ? 1 : 0;
    std::uint8_t dry_run   = sc.dry_run   ? 1 : 0;
    double ramp_alpha = sc.ramp_alpha;

    // ─── v5 split-topology / forensic fields ───────────────────────────
    // Computed once per tick so the dump_x2_debug.py consumer (and the
    // x2_motor_monitor sidecar) can correlate the policy clock with
    // operator-side stack health.
    double  pose_ref_age_s = -1.0;  // sentinel: no ZMQ source / no frames
    if (pose_ref_watchdog_active_ && zmq_pose_source_ != nullptr) {
      const double last_rx = zmq_pose_source_->LastReceivedMonotonicS();
      if (last_rx > 0.0) {
        pose_ref_age_s = std::max(0.0, now - last_rx);
      } else {
        // Watchdog active but no frame yet: report a large finite age
        // (NOT +inf, which decodes awkwardly on the Python side). 1e9
        // is "obviously starved" without breaking JSON encoders.
        pose_ref_age_s = 1.0e9;
      }
    }
    std::uint8_t in_safe_idle =
        (state_.load() == State::SAFE_IDLE) ? 1 : 0;
    std::uint8_t in_recover =
        (state_.load() == State::RECOVER) ? 1 : 0;
    std::uint8_t pose_ref_starved =
        (pose_ref_watchdog_active_ && pose_ref_watchdog_.Tripped()) ? 1 : 0;
    // Resume chord telemetry: age in seconds since most recent press
    // (sentinel -1 = no press / no SUB). Operator can correlate visible
    // SAFE_IDLE entries with chord receipt timing.
    double  resume_age_s = -1.0;
    int64_t resume_total = 0;
    if (zmq_resume_sub_) {
      const double last_press = zmq_resume_sub_->LastReceivedMonotonicS();
      if (last_press > 0.0) resume_age_s = std::max(0.0, now - last_press);
      resume_total = static_cast<int64_t>(zmq_resume_sub_->total_received());
    }
    int32_t mc_action_mode   = mc_action_mode_.load(std::memory_order_acquire);
    int32_t mc_action_status = mc_action_status_.load(std::memory_order_acquire);

    std::vector<PackedField> fields;
    fields.reserve(21);
    fields.push_back({"control_tick",   "i64", {1},                     &tick_i64,            sizeof(int64_t)});
    fields.push_back({"ros_timestamp",  "f64", {1},                     &now_f64,             sizeof(double)});
    fields.push_back({"policy_time",    "f64", {1},                     &policy_time_f64,     sizeof(double)});
    fields.push_back({"base_quat",      "f64", {4},                     base_quat.data(),     0});
    fields.push_back({"base_ang_vel",   "f64", {3},                     base_ang_vel.data(),  0});
    fields.push_back({"body_q",         "f64", {NUM_DOFS},              body_q.data(),        0});
    // whole-body engagement (dual-head path): the kplanner reads this to run
    // its shadow/hold/return instead of the token service's wb_state
    fields.push_back({"wb_engaged",     "f64", {1},                     &wb_engaged_dbg_,     0});
    fields.push_back({"body_dq",        "f64", {NUM_DOFS},              body_dq.data(),       0});
    fields.push_back({"last_action",    "f64", {NUM_DOFS},              last_action.data(),   0});
    fields.push_back({"left_hand_q",    "f64", {DEFAULT_HAND_DOF_PER_SIDE}, left_hand.data(),   0});
    fields.push_back({"right_hand_q",   "f64", {DEFAULT_HAND_DOF_PER_SIDE}, right_hand.data(),  0});
    fields.push_back({"hand_frame_idx", "i64", {1},                     &hand_frame_idx,      sizeof(int64_t)});
    fields.push_back({"ramp_alpha",     "f64", {1},                     &ramp_alpha,          sizeof(double)});
    fields.push_back({"tilt_trip",      "u8",  {1},                     &tilt_trip,           sizeof(std::uint8_t)});

    // dry_run was previously LAST in the v4 protocol. v5 keeps it BEFORE
    // the new fields so the protocol number reflects the schema and old
    // dump_x2_debug.py consumers either upgrade or short-read cleanly at
    // the dry_run boundary (the message length tells them which protocol
    // version landed).
    fields.push_back({"dry_run",         "u8",  {1}, &dry_run,         sizeof(std::uint8_t)});
    // v5 forensic fields (Phase 2 + 5b).
    fields.push_back({"pose_ref_age_s",  "f64", {1}, &pose_ref_age_s,  sizeof(double)});
    fields.push_back({"pose_ref_starved","u8",  {1}, &pose_ref_starved,sizeof(std::uint8_t)});
    fields.push_back({"in_safe_idle",    "u8",  {1}, &in_safe_idle,    sizeof(std::uint8_t)});
    fields.push_back({"in_recover",      "u8",  {1}, &in_recover,      sizeof(std::uint8_t)});
    fields.push_back({"resume_age_s",    "f64", {1}, &resume_age_s,    sizeof(double)});
    fields.push_back({"resume_total",    "i64", {1}, &resume_total,    sizeof(int64_t)});
    fields.push_back({"mc_action_mode",  "i32", {1}, &mc_action_mode,  sizeof(int32_t)});
    fields.push_back({"mc_action_status","i32", {1}, &mc_action_status,sizeof(int32_t)});

    if (zmq_debug_pub_->Publish(fields, /*version=*/5, /*count=*/1)) {
      ++zmq_debug_frames_published_;
    }
  }

  // Write the first-tick inference payload to PATH as a binary blob, then
  // request shutdown. Layout (little-endian, no padding):
  //
  //   magic        : char[8]  = "X2OBSV01"
  //   tok_dim      : uint32_t = 680
  //   prop_dim     : uint32_t = 990
  //   action_dim   : uint32_t = 31
  //   policy_time  : float64
  //   tokenizer_obs: float32[tok_dim]
  //   proprioception: float32[prop_dim]
  //   action_il    : float64[action_dim]
  //   joint_pos_mj : float64[31]
  //   joint_vel_mj : float64[31]
  //   base_quat_wxyz: float64[4]
  //   base_ang_vel : float64[3]
  //
  // Total: 8 + 12 + 8 + (680+990)*4 + (31+31+31+4+3)*8 = 7508 bytes.
  // The companion script
  //   gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py
  // reads this and diffs each named slot against
  // /tmp/x2_step0_isaaclab_lastpt.pt.
  void DumpObsBlob(const std::string&                       path,
                   const std::vector<float>&                tok_obs,
                   const std::vector<float>&                prop,
                   const std::array<double, NUM_DOFS>&      action_il,
                   const RobotState&                        rs,
                   double                                   policy_time)
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      RCLCPP_ERROR(node_->get_logger(),
                   "obs-dump: failed to open %s for writing", path.c_str());
      return;
    }
    constexpr char kMagic[8] = {'X','2','O','B','S','V','0','1'};
    const std::uint32_t tok_dim    = static_cast<std::uint32_t>(tok_obs.size());
    const std::uint32_t prop_dim   = static_cast<std::uint32_t>(prop.size());
    const std::uint32_t action_dim = static_cast<std::uint32_t>(NUM_DOFS);

    out.write(kMagic, sizeof(kMagic));
    out.write(reinterpret_cast<const char*>(&tok_dim),    sizeof(tok_dim));
    out.write(reinterpret_cast<const char*>(&prop_dim),   sizeof(prop_dim));
    out.write(reinterpret_cast<const char*>(&action_dim), sizeof(action_dim));
    out.write(reinterpret_cast<const char*>(&policy_time), sizeof(policy_time));

    out.write(reinterpret_cast<const char*>(tok_obs.data()),
              tok_obs.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(prop.data()),
              prop.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(action_il.data()),
              action_il.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(rs.joint_pos_mj.data()),
              rs.joint_pos_mj.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(rs.joint_vel_mj.data()),
              rs.joint_vel_mj.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(rs.base_quat_wxyz.data()),
              rs.base_quat_wxyz.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(rs.base_ang_vel.data()),
              rs.base_ang_vel.size() * sizeof(double));
  }
};

}  // namespace agi_x2

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
  using namespace agi_x2;

  rclcpp::init(argc, argv);

  CliArgs cli;
  try {
    cli = ParseCli(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n\n";
    PrintUsage();
    rclcpp::shutdown();
    return 2;
  }

  // RUNTIME PLANT (2026-09-08). policy_parameters.hpp carries ONE compiled plant
  // (vendor_20260823). The first checkpoint trained on a second plant
  // (pico_v15b_waist36: waist effort 36) showed the trap: its waist action scale is
  // 0.25*effort/kp = 1.5x the compiled value, so this binary applied 2/3 of the
  // trained waist command in sim and would on the robot. plant_loader.hpp existed
  // but was never called. Now: X2_PLANT_YAML=<path> (explicit) or X2_PLANT=<name>
  // (resolved against the repo the binary runs from) loads the yaml and overwrites
  // kps/kds/x2_action_scale/default_angles BEFORE anything reads them. X2_PLANT set
  // but unresolvable = refuse to start (a mismatched plant is the silent failure
  // this guards against). Unset = compiled tables, logged as such.
  {
    const char* yaml_env = std::getenv("X2_PLANT_YAML");
    const char* name_env = std::getenv("X2_PLANT");
    std::string path;
    if (yaml_env && *yaml_env) {
      path = yaml_env;
    } else if (name_env && *name_env) {
      const std::string rel = std::string("gear_sonic/config/robot_plant/x2_ultra_") + name_env + ".yaml";
      const char* root_env = std::getenv("X2_REPO_ROOT");
      const std::vector<std::string> roots = {
          root_env ? std::string(root_env) : std::string(),
          ".", "/workspace/sonic", "/home/run/gear-sonic"};
      for (const auto& r : roots) {
        if (r.empty()) continue;
        const std::string cand = r + "/" + rel;
        if (std::ifstream(cand).good()) { path = cand; break; }
      }
      if (path.empty()) {
        // The plant compiled into policy_parameters.hpp is vendor_20260823. An install that names
        // that plant but carries no yaml (every PC2 before 2026-09-09: the old binary never read one)
        // must keep working with the compiled tables -- refusing here broke the demo ritual on
        // 2026-09-09 01:55 PC2 local. Only a NON-compiled plant with no yaml is a hard error.
        if (std::string(name_env) == "vendor_20260823") {
          std::cout << "[plant] X2_PLANT=vendor_20260823 (the compiled plant) and no yaml found -- "
                       "using the COMPILED tables (identical numbers)\n";
        } else {
          std::cerr << "[plant] X2_PLANT=" << name_env << " set but " << rel
                    << " not found under X2_REPO_ROOT / . / /workspace/sonic / /home/run/gear-sonic"
                    << " -- refusing to run with a plant that may not match the model\n";
          rclcpp::shutdown();
          return 3;
        }
      }
    }
    if (!path.empty()) {
      try {
        const Plant p = LoadPlant(path);
        kps = p.kp; kds = p.kd; x2_action_scale = p.action_scale; default_angles = p.default_angle;
        std::size_t wp = 0;
        for (std::size_t i = 0; i < NUM_DOFS; ++i)
          if (std::string(mujoco_joint_names[i]) == "waist_pitch_joint") wp = i;
        std::cout << "[plant] RUNTIME PLANT LOADED: " << p.name << " from " << path
                  << "  (waist_pitch kp " << kps[wp] << " kd " << kds[wp]
                  << " action_scale " << x2_action_scale[wp] << ")\n";
      } catch (const std::exception& e) {
        std::cerr << "[plant] failed to load plant yaml " << path << ": " << e.what() << "\n";
        rclcpp::shutdown();
        return 3;
      }
    } else {
      std::cout << "[plant] COMPILED plant tables in effect (policy_parameters.hpp; X2_PLANT unset)\n";
    }
  }

  // Install our soft-shutdown SIGINT/SIGTERM handler IFF the operator opted
  // in via --soft-shutdown-trigger-sentinel. Doing so overrides rclcpp's
  // default handler (which calls rclcpp::shutdown() and lets main() return,
  // bypassing RAMP_OUT/HOLD_FOR_MC entirely). When the flag is empty we
  // leave rclcpp's default in place so legacy behaviour is bit-exact.
  // SoftShutdownSignalHandler is async-signal-safe (atomic flag + write(2));
  // it sets g_soft_shutdown_requested, which OnControl polls on the next
  // 50 Hz tick (<= 20 ms latency) and uses to enter RAMP_OUT.
  if (!cli.soft_shutdown_trigger_sentinel.empty()) {
    struct sigaction sa{};
    sa.sa_handler = SoftShutdownSignalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;  // do not interrupt blocking syscalls (e.g. ZMQ recv)
    if (sigaction(SIGINT,  &sa, nullptr) != 0
        || sigaction(SIGTERM, &sa, nullptr) != 0) {
      std::cerr << "[soft-shutdown] sigaction install failed (errno=" << errno
                << "): falling back to rclcpp default SIGINT handler.\n";
    }
  }

  auto node = rclcpp::Node::make_shared("x2_deploy_onnx_ref");
  const std::string banner_dry  = cli.dry_run ? " [DRY-RUN]" : "";
  const std::string banner_auto = cli.autostart_seconds >= 0
      ? (" autostart=" + std::to_string(cli.autostart_seconds) + "s")
      : std::string(" (operator-go required: type 'go' on stdin)");
  const std::string banner_dur  = cli.max_duration > 0
      ? (" max-duration=" + std::to_string(cli.max_duration) + "s")
      : std::string("");
  RCLCPP_INFO(node->get_logger(),
              "x2_deploy_onnx_ref starting%s%s%s",
              banner_dry.c_str(), banner_auto.c_str(), banner_dur.c_str());

  std::unique_ptr<X2Deploy> deploy;
  try {
    deploy = std::make_unique<X2Deploy>(node, cli);
  } catch (const std::exception& e) {
    RCLCPP_FATAL(node->get_logger(), "Initialization failed: %s", e.what());
    rclcpp::shutdown();
    return 3;
  }

  // Operator-go gate: when --autostart-after is not used, read a single line from
  // stdin to advance from WAIT_FOR_CONTROL to CONTROL. We do this in a
  // dedicated thread so it doesn't block the executor.
  std::thread operator_thread;
  if (cli.autostart_seconds < 0.0) {
    operator_thread = std::thread([&deploy]() {
      std::cout << "\n[operator] Type 'go' + Enter to enter CONTROL state.\n"
                   "          (or Ctrl-C to abort)\n"
                << std::flush;
      std::string line;
      while (rclcpp::ok() && std::getline(std::cin, line)) {
        if (line == "go") {
          deploy->RequestGo();
          break;
        }
        std::cout << "[operator] (ignored '" << line
                  << "'; type 'go' to start)\n" << std::flush;
      }
    });
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  if (operator_thread.joinable()) operator_thread.join();
  rclcpp::shutdown();
  return 0;
}
