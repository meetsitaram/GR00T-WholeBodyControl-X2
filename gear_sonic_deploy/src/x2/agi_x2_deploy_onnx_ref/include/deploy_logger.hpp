/**
 * @file deploy_logger.hpp
 * @brief Lightweight per-tick CSV logger for X2 deploy.
 *
 * Sized for the X2 obs/action contract (31 DOFs, 990-D proprioception, etc.)
 * and intentionally NOT trying to be the G1 StateLogger -- that one bundles
 * Dex3 hands, encoder mode, motion library state, and other things we don't
 * need on X2. Keeping this small means there's only one file to evolve when
 * we want to add new fields to the rollout audit trail.
 *
 * Files written into the configured output directory:
 *   tick.csv        -- per-control-tick scalars
 *   target_pos.csv  -- 31-D PD target positions, MJ order, post-safety
 *   joint_pos.csv   -- 31-D measured joint positions, MJ order
 *   joint_vel.csv   -- 31-D measured joint velocities, MJ order
 *   action_il.csv   -- 31-D raw policy output, IL order
 *   imu.csv         -- base_quat (wxyz) and base_ang_vel (xyz)
 *
 * All numbers are normal floats; no compression. ~50 Hz * 31 DOF for ~1 hour
 * is ~6 MB of CSV, fine for short bring-up sessions.
 */

#ifndef AGI_X2_DEPLOY_LOGGER_HPP
#define AGI_X2_DEPLOY_LOGGER_HPP

#include "policy_parameters.hpp"
#include "safety.hpp"

#include <array>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace agi_x2 {

class DeployLogger {
 public:
  /// @param output_dir  destination directory; created if missing
  /// @param enabled     master switch (false = no-op for performance tests)
  DeployLogger(const std::string& output_dir, bool enabled = true);
  ~DeployLogger();

  DeployLogger(const DeployLogger&)            = delete;
  DeployLogger& operator=(const DeployLogger&) = delete;

  bool Enabled() const { return enabled_; }

  /// One row per call. Safe to call from the 50 Hz control timer.
  void Log(double                              now_s,
           const std::array<double, NUM_DOFS>& joint_pos_mj,
           const std::array<double, NUM_DOFS>& joint_vel_mj,
           const std::array<double, 4>&        base_quat_wxyz,
           const std::array<double, 3>&        base_ang_vel,
           const std::array<double, NUM_DOFS>& action_il,
           const SafeCommand&                  safe_cmd);

 private:
  bool         enabled_;
  std::string  output_dir_;
  std::mutex   io_mutex_;
  std::ofstream tick_;        // scalars: time, alpha, dry_run, tilt_trip
  std::ofstream target_pos_;  // 31-D
  std::ofstream joint_pos_;
  std::ofstream joint_vel_;
  std::ofstream action_il_;
  std::ofstream imu_;
  double last_flush_s_ = 0.0;  // 1 Hz flush so power cuts keep evidence
};

}  // namespace agi_x2

#endif  // AGI_X2_DEPLOY_LOGGER_HPP
