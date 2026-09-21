/**
 * @file head_bypass.hpp
 * @brief Surgical override of SONIC's head targets from the ZMQ reference.
 *
 * SONIC's head-channel gain is untrained (the corpus head is mostly
 * static -- same pathology the wrist bypass documents for the wrist
 * pitch/roll channels), so the pad's head-look feature (daemon
 * overlays head_yaw on the pose wire) needs the actual joint target
 * to follow the reference deterministically rather than through the
 * policy. (Historical note: the 2026-08-12 "saturation at +-10 deg"
 * probe turned out to be a PHANTOM SIM CONTACT -- head_pitch_link's
 * collision cylinder vs torso_link's convex hull, excluded in
 * x2_ultra.xml 2026-08-13 -- not a policy ceiling.) Identical
 * architecture to ``wrist_bypass.hpp`` -- overwrite the per-tick PD
 * target for the two head DOFs with the reference straight off the
 * ZMQ pose feed, BEFORE the safety stack (soft-start ramp,
 * --max-target-dev-head clamp and the tilt-trip branch still apply).
 * Tokenizer obs unchanged.
 *
 * Safety scale: head_yaw actuator is +-2.6 Nm with a +-0.366 rad joint
 * range; head_pitch +-0.6 Nm / +-0.384 rad. The smallest motors on the
 * robot, zero balance role.
 *
 * CLI-gated ``--head-bypass {off,ref}``, default off (sim-to-real
 * fidelity on the motion-file replay path, same rationale as wrists).
 */

#ifndef AGI_X2_HEAD_BYPASS_HPP
#define AGI_X2_HEAD_BYPASS_HPP

#include "policy_parameters.hpp"
#include "reference_motion.hpp"

#include <array>
#include <cmath>

namespace agi_x2 {

/// MJ-order joint indices the bypass overrides:
///   29 = head_yaw
///   30 = head_pitch
/// Keep in sync with ``policy_parameters.hpp``'s ``mujoco_joint_names``.
inline constexpr std::array<int, 2> kBypassedHeadMjDofs = {29, 30};

/// Override ``target_pos_mj[mj]`` with ``ref.joint_pos_mj[mj]`` for the
/// head DOFs. Returns the largest absolute delta between the original
/// target and the reference across the overridden slots (periodic
/// status-line indicator, mirroring ApplyWristBypass).
inline double ApplyHeadBypass(std::array<double, NUM_DOFS>& target_pos_mj,
                              const ReferenceFrame&         ref)
{
  double max_delta = 0.0;
  for (const int mj : kBypassedHeadMjDofs) {
    const double delta = std::fabs(target_pos_mj[mj] - ref.joint_pos_mj[mj]);
    if (delta > max_delta) max_delta = delta;
    target_pos_mj[mj] = ref.joint_pos_mj[mj];
  }
  return max_delta;
}

}  // namespace agi_x2

#endif  // AGI_X2_HEAD_BYPASS_HPP
