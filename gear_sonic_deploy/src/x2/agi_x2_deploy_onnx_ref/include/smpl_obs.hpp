/**
 * @file smpl_obs.hpp
 * @brief In-deploy SMPL observation builder for the native whole-body head.
 *
 * Faithful port of gear_sonic/utils/teleop/smpl_obs.py::build_smpl_obs
 * (2026-09-04, wb_release_stability.md 4c/"one SONIC on the robot"):
 *
 *   smpl_obs[840] = 10 x [ joints 72 | rel root rot6d 6 | measured wrists 6 ]
 *
 *   rel = inv(cur) * (yaw_align * human_quat[f])
 *     cur       = robot base quat (ori_mode "full") or its yaw only
 *                 (ori_mode "heading", v1.1 release cores)
 *     yaw_align = Rz(heading(robot) - heading(human)) captured at engage
 *
 * The Python builder stays the parity reference (test: same intent tape,
 * same robot quat -> identical 840 floats).
 */
#ifndef AGI_X2_SMPL_OBS_HPP
#define AGI_X2_SMPL_OBS_HPP

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace agi_x2 {

constexpr std::size_t SMPL_WINDOW    = 10;
constexpr double      SMPL_DT        = 0.02;
constexpr double      SMPL_DELAY_S   = (SMPL_WINDOW - 1) * SMPL_DT;   // 0.18 s
constexpr std::size_t SMPL_JOINTS    = 72;                            // 24 x 3
constexpr std::size_t SMPL_OBS_DIM   = SMPL_WINDOW * (SMPL_JOINTS + 6 + 6);  // 840

/// One human frame as the laptop sends it (pico_intent).
struct SmplFrame {
  double                      t_mono = 0.0;    ///< receive time (steady clock, s)
  std::array<float, 72>       joints{};        ///< SMPL joints (root-local, heading-normalised)
  std::array<double, 4>       quat_wxyz{1, 0, 0, 0};  ///< human root orientation
};

/// OPERATOR ROOT LEVELING (2026-09-15): the Pico root carries a steady forward
/// pitch that the SMPL path otherwise commands as a lean. mode = "" | "off"
/// (no-op) | "zero" (keep heading, zero pitch+roll) | "clamp:<deg>" |
/// "offset:<deg>" (subtract a fixed forward pitch). Applied to every window
/// frame's quat_wxyz; joints (root-local) are untouched. Mirrors
/// gear_sonic/scripts/pc2_pico_token_service.py --operator-root-level.
void LevelOperatorRoot(std::array<SmplFrame, SMPL_WINDOW>& win, const std::string& mode);

/// Heading (yaw of the rotated body-X axis) of a wxyz quaternion.
double SmplHeading(const std::array<double, 4>& q_wxyz);

/// yaw_align = Rz(heading(robot) - heading(human)) as an xyzw quaternion.
std::array<double, 4> SmplYawAlign(const std::array<double, 4>& robot_wxyz,
                                   const std::array<double, 4>& human_wxyz);

/**
 * Build the 840-d observation.
 * @param window      exactly SMPL_WINDOW frames, oldest first
 * @param base_wxyz   robot base orientation (IMU, wxyz)
 * @param wrist6      measured wrists in IL slice order
 *                    [l_yaw l_pitch l_roll r_yaw r_pitch r_roll]
 * @param yaw_align   optional Rz alignment (xyzw); nullopt = none
 * @param ori_mode    "full" or "heading"
 */
std::vector<float> BuildSmplObs(
    const std::array<SmplFrame, SMPL_WINDOW>& window,
    const std::array<double, 4>&              base_wxyz,
    const std::array<double, 6>&              wrist6,
    const std::optional<std::array<double, 4>>& yaw_align_xyzw,
    const std::string&                        ori_mode);

}  // namespace agi_x2

#endif  // AGI_X2_SMPL_OBS_HPP
