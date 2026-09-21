#ifndef AGI_X2_PLANT_LOADER_HPP
#define AGI_X2_PLANT_LOADER_HPP

/**
 * @file plant_loader.hpp
 * @brief Loads the X2 actuator plant (armature / effort / PD / action scale)
 *        from gear_sonic/config/robot_plant/x2_ultra_<name>.yaml at RUNTIME.
 *
 * WHY. Until 2026-08-23 policy_parameters.hpp hardcoded kps[], kds[],
 * x2_action_scale[] and default_angles[]. That made this binary the SEVENTH
 * independent copy of the plant, alongside x2_ultra.py, the MJCF,
 * eval_x2_mujoco.py, the two plant YAMLs and each model's phi sidecar. Nothing
 * tied them together, and they drifted:
 *
 *   waist_pitch/roll action scale   header 0.8420684 (48 N.m era)
 *                                   v12 sidecar 0.6315513 (36 N.m era)
 *   -> every v12 waist command is amplified 1.333x ON THE ROBOT.
 *      v0 is immune: direct policy trained in the 48 era, so it matches.
 *      That is a candidate root cause for the v12 waist failure.
 *
 *   wrist_pitch/roll                header 4.8 N.m, python configs 6.0 N.m
 *   -> here the HEADER is right; the vendor datasheet says 4.8.
 *
 * Parsed with a line regex rather than yaml-cpp, matching stand_pose_loader's
 * reasoning: the files are small, fixed-shape and machine-checked, and this
 * package deliberately carries no yaml dependency.
 *
 * KEYS ARE SUBSTRINGS of the joint name, FIRST MATCH WINS, so ORDER MATTERS --
 * specific before general. ankle and shoulder each span two motor families;
 * collapsing either reintroduces a 2.45x armature error.
 *
 * The derivation reproduces policy_parameters.hpp exactly:
 *   kp[i]           = armature(j) * w^2 * deployment_kp_scale(j)
 *   kd[i]           = 2 * zeta * armature(j) * w * deployment_kd_scale(j)
 *   action_scale[i] = 0.25 * effort(j) / (armature(j) * w^2)   <-- UNSCALED kp,
 *                     so a deployment stiffening shows up as more torque per
 *                     unit error, not as a rescaled command range.
 */

#include "policy_parameters.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace agi_x2 {

struct Plant {
  std::string name;
  std::array<double, NUM_DOFS> kp{};
  std::array<double, NUM_DOFS> kd{};
  std::array<double, NUM_DOFS> action_scale{};
  std::array<double, NUM_DOFS> default_angle{};
  std::size_t parsed_joints = 0;
};

namespace detail {

/// substring -> value, insertion-ordered; first match wins.
using KeyTable = std::vector<std::pair<std::string, double>>;

inline bool lookup(const KeyTable& t, const std::string& joint, double* out) {
  for (const auto& kv : t) {
    if (joint.find(kv.first) != std::string::npos) { *out = kv.second; return true; }
  }
  return false;
}

}  // namespace detail

/**
 * @param path  full path to x2_ultra_<name>.yaml
 * @throws std::runtime_error on a missing file, an unparsable section, or a
 *         joint with no matching armature/effort key -- never silently
 *         defaults, because a silent default is exactly how the 1.333x waist
 *         mismatch survived.
 */
inline Plant LoadPlant(const std::string& path) {
  std::ifstream fh(path);
  if (!fh) throw std::runtime_error("plant YAML not found: " + path);

  static const std::regex sec_re(R"(^([a-z_]+):\s*$)");
  static const std::regex kv_re(
      R"(^\s+([a-z_]+):\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?))");
  static const std::regex name_re(R"(^name:\s*([A-Za-z0-9_]+))");
  static const std::regex freq_re(R"(^natural_freq_hz:\s*([0-9.]+))");
  static const std::regex zeta_re(R"(^damping_ratio:\s*([0-9.]+))");

  detail::KeyTable arm, eff, kps_s, kds_s, dpos;
  std::string name = "unknown", section;
  double freq_hz = 10.0, zeta = 2.0;

  std::string line;
  while (std::getline(fh, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::smatch m;
    if (std::regex_search(line, m, name_re)) { name = m[1].str(); continue; }
    if (std::regex_search(line, m, freq_re)) { freq_hz = std::stod(m[1].str()); continue; }
    if (std::regex_search(line, m, zeta_re)) { zeta = std::stod(m[1].str()); continue; }
    if (std::regex_match(line, m, sec_re))   { section = m[1].str(); continue; }
    if (!section.empty() && std::regex_search(line, m, kv_re)) {
      const std::string k = m[1].str();
      const double v = std::stod(m[2].str());
      if      (section == "armature")            arm.emplace_back(k, v);
      else if (section == "effort")              eff.emplace_back(k, v);
      else if (section == "deployment_kp_scale") kps_s.emplace_back(k, v);
      else if (section == "deployment_kd_scale") kds_s.emplace_back(k, v);
      else if (section == "default_joint_pos")   dpos.emplace_back(k, v);
    }
  }
  if (arm.empty() || eff.empty())
    throw std::runtime_error("plant YAML missing armature/effort: " + path);

  const double w = 2.0 * M_PI * freq_hz;
  Plant p;
  p.name = name;
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    const std::string& j = mujoco_joint_names[i];
    double a = 0.0, e = 0.0, ks = 1.0, ds = 1.0, d0 = 0.0;
    if (!detail::lookup(arm, j, &a))
      throw std::runtime_error("plant " + name + ": no armature for " + j);
    if (!detail::lookup(eff, j, &e))
      throw std::runtime_error("plant " + name + ": no effort for " + j);
    detail::lookup(kps_s, j, &ks);
    detail::lookup(kds_s, j, &ds);
    detail::lookup(dpos, j, &d0);
    const double kp_train = a * w * w;
    p.kp[i]           = kp_train * ks;
    p.kd[i]           = 2.0 * zeta * a * w * ds;
    p.action_scale[i] = 0.25 * e / kp_train;   // UNSCALED kp, deliberately
    p.default_angle[i] = d0;
    ++p.parsed_joints;
  }
  return p;
}

}  // namespace agi_x2

#endif  // AGI_X2_PLANT_LOADER_HPP
