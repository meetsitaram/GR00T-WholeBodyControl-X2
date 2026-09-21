#include "smpl_obs.hpp"
#include <string>
#include <algorithm>

#include "math_utils.hpp"

#include <cmath>
#include <stdexcept>

namespace agi_x2 {

double SmplHeading(const std::array<double, 4>& q_wxyz)
{
  // scipy: rot.apply([1,0,0]) -> atan2(v.y, v.x) == yaw of the body-X axis
  return yaw_from_quat_wxyz(q_wxyz);
}

std::array<double, 4> SmplYawAlign(const std::array<double, 4>& robot_wxyz,
                                   const std::array<double, 4>& human_wxyz)
{
  return yaw_quat_xyzw(SmplHeading(robot_wxyz) - SmplHeading(human_wxyz));
}

std::vector<float> BuildSmplObs(
    const std::array<SmplFrame, SMPL_WINDOW>& window,
    const std::array<double, 4>&              base_wxyz,
    const std::array<double, 6>&              wrist6,
    const std::optional<std::array<double, 4>>& yaw_align_xyzw,
    const std::string&                        ori_mode)
{
  // cur = robot base orientation; "heading" mode keeps only its yaw
  std::array<double, 4> cur_xyzw = wxyz_to_xyzw(base_wxyz);
  if (ori_mode == "heading") {
    cur_xyzw = yaw_quat_xyzw(yaw_from_quat_xyzw(cur_xyzw));
  } else if (ori_mode != "full") {
    throw std::runtime_error("BuildSmplObs: ori_mode must be 'full' or 'heading', got '" +
                             ori_mode + "'");
  }
  const auto cur_inv = quat_conj_xyzw(cur_xyzw);

  std::vector<float> obs;
  obs.reserve(SMPL_OBS_DIM);
  for (std::size_t f = 0; f < SMPL_WINDOW; ++f) {
    const SmplFrame& fr = window[f];
    std::array<double, 4> o = wxyz_to_xyzw(fr.quat_wxyz);
    if (yaw_align_xyzw) o = quat_mul_xyzw(*yaw_align_xyzw, o);   // yaw_align * o
    const auto rel  = quat_mul_xyzw(cur_inv, o);                  // cur.inv() * o
    const auto rot6 = rot6d_from_quat_xyzw(rel);                  // matrix[:, :2] row-major
    for (float v : fr.joints) obs.push_back(v);
    for (double v : rot6)     obs.push_back(static_cast<float>(v));
    for (double v : wrist6)   obs.push_back(static_cast<float>(v));
  }
  return obs;
}


// ---- OPERATOR ROOT LEVELING ------------------------------------------------
// ZYX (yaw-pitch-roll) extraction / composition on wxyz quaternions, same
// convention as scipy Rotation.from_euler("ZYX", [yaw, pitch, roll]) used by
// the python token service: R = Rz(yaw) * Ry(pitch) * Rx(roll).
namespace {
void quat_wxyz_to_zyx(const std::array<double, 4>& q, double& yaw, double& pitch, double& roll) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double r00 = 1 - 2 * (y * y + z * z), r10 = 2 * (x * y + z * w);
  const double r20 = 2 * (x * z - y * w), r21 = 2 * (y * z + x * w), r22 = 1 - 2 * (x * x + y * y);
  yaw   = std::atan2(r10, r00);
  pitch = std::asin(std::max(-1.0, std::min(1.0, -r20)));
  roll  = std::atan2(r21, r22);
}
std::array<double, 4> zyx_to_quat_wxyz(double yaw, double pitch, double roll) {
  const double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
  const double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
  const double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
  // q = qz(yaw) * qy(pitch) * qx(roll)
  return {cr * cp * cy + sr * sp * sy,
          sr * cp * cy - cr * sp * sy,
          cr * sp * cy + sr * cp * sy,
          cr * cp * sy - sr * sp * cy};
}
}  // namespace

void LevelOperatorRoot(std::array<SmplFrame, SMPL_WINDOW>& win, const std::string& mode) {
  if (mode.empty() || mode == "off") return;
  double lim = -1.0, off = 0.0; bool zero = false;
  if (mode == "zero") zero = true;
  else if (mode.rfind("clamp:", 0) == 0) lim = std::stod(mode.substr(6)) * M_PI / 180.0;
  else if (mode.rfind("offset:", 0) == 0) off = std::stod(mode.substr(7)) * M_PI / 180.0;
  else throw std::runtime_error("LevelOperatorRoot: mode must be off|zero|clamp:<deg>|offset:<deg>, got '" + mode + "'");
  for (auto& fr : win) {
    double yaw, pitch, roll; quat_wxyz_to_zyx(fr.quat_wxyz, yaw, pitch, roll);
    if (zero)          { pitch = 0.0; roll = 0.0; }
    else if (lim >= 0) { pitch = std::max(-lim, std::min(lim, pitch)); roll = std::max(-lim, std::min(lim, roll)); }
    else               { pitch -= off; }
    fr.quat_wxyz = zyx_to_quat_wxyz(yaw, pitch, roll);
  }
}

}  // namespace agi_x2
