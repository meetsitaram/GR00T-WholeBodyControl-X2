#include "deploy_logger.hpp"

#include <filesystem>
#include <iomanip>

namespace agi_x2 {

namespace fs = std::filesystem;

namespace {

void WriteHeaderJointVec(std::ofstream& f, const char* prefix)
{
  f << "t";
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    f << "," << prefix << "_" << mujoco_joint_names[i];
  }
  f << "\n";
}

void WriteRow(std::ofstream& f, double t,
              const std::array<double, NUM_DOFS>& v)
{
  f << std::fixed << std::setprecision(6) << t;
  for (double x : v) f << "," << x;
  f << "\n";
}

}  // namespace

DeployLogger::DeployLogger(const std::string& output_dir, bool enabled)
    : enabled_(enabled),
      output_dir_(output_dir)
{
  if (!enabled_) return;

  fs::create_directories(output_dir_);

  tick_.open(output_dir_ + "/tick.csv");
  tick_ << "t,ramp_alpha,dry_run,tilt_trip,reason\n";

  target_pos_.open(output_dir_ + "/target_pos.csv");
  WriteHeaderJointVec(target_pos_, "target");

  joint_pos_.open(output_dir_ + "/joint_pos.csv");
  WriteHeaderJointVec(joint_pos_, "q");

  joint_vel_.open(output_dir_ + "/joint_vel.csv");
  WriteHeaderJointVec(joint_vel_, "dq");

  action_il_.open(output_dir_ + "/action_il.csv");
  // action_il is in IL order, so we don't tag with mj joint names; just
  // index it. Anyone post-processing can re-permute via mujoco_to_isaaclab.
  action_il_ << "t";
  for (std::size_t i = 0; i < NUM_DOFS; ++i) action_il_ << ",a_il_" << i;
  action_il_ << "\n";

  imu_.open(output_dir_ + "/imu.csv");
  imu_ << "t,qw,qx,qy,qz,wx,wy,wz\n";
}

DeployLogger::~DeployLogger() = default;

void DeployLogger::Log(double                              now_s,
                       const std::array<double, NUM_DOFS>& joint_pos_mj,
                       const std::array<double, NUM_DOFS>& joint_vel_mj,
                       const std::array<double, 4>&        base_quat_wxyz,
                       const std::array<double, 3>&        base_ang_vel,
                       const std::array<double, NUM_DOFS>& action_il,
                       const SafeCommand&                  safe_cmd)
{
  if (!enabled_) return;
  std::lock_guard<std::mutex> lk(io_mutex_);

  tick_ << std::fixed << std::setprecision(6) << now_s
        << "," << safe_cmd.ramp_alpha
        << "," << (safe_cmd.dry_run ? 1 : 0)
        << "," << (safe_cmd.tilt_trip ? 1 : 0)
        << ",\"" << safe_cmd.reason << "\"\n";

  WriteRow(target_pos_, now_s, safe_cmd.target_pos_mj);
  WriteRow(joint_pos_,  now_s, joint_pos_mj);
  WriteRow(joint_vel_,  now_s, joint_vel_mj);

  // action_il written manually because it has its own column naming above.
  action_il_ << std::fixed << std::setprecision(6) << now_s;
  for (double x : action_il) action_il_ << "," << x;
  action_il_ << "\n";

  imu_ << std::fixed << std::setprecision(6) << now_s
       << "," << base_quat_wxyz[0]
       << "," << base_quat_wxyz[1]
       << "," << base_quat_wxyz[2]
       << "," << base_quat_wxyz[3]
       << "," << base_ang_vel[0]
       << "," << base_ang_vel[1]
       << "," << base_ang_vel[2]
       << "\n";

  // Flush all streams once a second. 2026-08-04 incident postmortem had
  // ZERO rows — the operator pulled the battery ~14 s into the session
  // and every CSV was still sitting in ofstream buffers. A 1 Hz flush
  // costs nothing at this data rate and guarantees a power-cut leaves
  // at most the last second unrecorded.
  if (now_s - last_flush_s_ >= 1.0) {
    last_flush_s_ = now_s;
    tick_.flush();
    target_pos_.flush();
    joint_pos_.flush();
    joint_vel_.flush();
    action_il_.flush();
    imu_.flush();
  }
}

}  // namespace agi_x2
