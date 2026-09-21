#include <mutex>
#include "aimdk_io.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

namespace agi_x2 {

namespace {

inline double SteadyNowSec()
{
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

AimdkIo::AimdkIo(rclcpp::Node::SharedPtr node,
                 const std::string& leg_topic_prefix,
                 const std::string& waist_topic_prefix,
                 const std::string& arm_topic_prefix,
                 const std::string& head_topic_prefix,
                 const std::string& imu_topic)
    : node_(node)
{
  using std::placeholders::_1;

  const auto qos_state = rclcpp::SensorDataQoS();   // best-effort, depth=10
  const auto qos_cmd   = rclcpp::SensorDataQoS();

  // -------------------------------------------------------------------------
  // Subscribers (one per joint group + IMU). Each callback assembles its
  // slice into latest_state_ under state_mutex_; on the first callback we
  // also validate joint names against mujoco_joint_names[] so a firmware
  // re-ordering trips a fatal error early instead of silently feeding the
  // policy a permuted observation.
  // -------------------------------------------------------------------------
  state_cbg_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions state_opts;
  state_opts.callback_group = state_cbg_;

  sub_leg_ = node_->create_subscription<aimdk_msgs::msg::JointStateArray>(
      leg_topic_prefix + "/state", qos_state,
      [this](const aimdk_msgs::msg::JointStateArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(state_mutex_);
        IngestJointGroup(*msg, kLegStart, kLegLen, "leg", 0,
                         latest_state_.leg_stamp_s, latest_state_.leg_seen);
      },
      state_opts);

  sub_waist_ = node_->create_subscription<aimdk_msgs::msg::JointStateArray>(
      waist_topic_prefix + "/state", qos_state,
      [this](const aimdk_msgs::msg::JointStateArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(state_mutex_);
        IngestJointGroup(*msg, kWaistStart, kWaistLen, "waist", 1,
                         latest_state_.waist_stamp_s, latest_state_.waist_seen);
      },
      state_opts);

  sub_arm_ = node_->create_subscription<aimdk_msgs::msg::JointStateArray>(
      arm_topic_prefix + "/state", qos_state,
      [this](const aimdk_msgs::msg::JointStateArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(state_mutex_);
        IngestJointGroup(*msg, kArmStart, kArmLen, "arm", 2,
                         latest_state_.arm_stamp_s, latest_state_.arm_seen);
      },
      state_opts);

  sub_head_ = node_->create_subscription<aimdk_msgs::msg::JointStateArray>(
      head_topic_prefix + "/state", qos_state,
      [this](const aimdk_msgs::msg::JointStateArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(state_mutex_);
        IngestJointGroup(*msg, kHeadStart, kHeadLen, "head", 3,
                         latest_state_.head_stamp_s, latest_state_.head_seen);
      },
      state_opts);

  sub_imu_ = node_->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, qos_state,
      [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(state_mutex_);
        // sensor_msgs/Imu orientation is xyzw; convert to wxyz to match
        // policy_parameters.hpp / IsaacLab convention.
        latest_state_.base_quat_wxyz = {
            msg->orientation.w,
            msg->orientation.x,
            msg->orientation.y,
            msg->orientation.z,
        };
        latest_state_.base_ang_vel = {
            msg->angular_velocity.x,
            msg->angular_velocity.y,
            msg->angular_velocity.z,
        };
        latest_state_.imu_stamp_s = SteadyNowSec();
        latest_state_.imu_seen    = true;
      },
      state_opts);

  // -------------------------------------------------------------------------
  // Publishers
  // -------------------------------------------------------------------------
  pub_leg_   = node_->create_publisher<aimdk_msgs::msg::JointCommandArray>(
      leg_topic_prefix + "/command",   qos_cmd);
  pub_waist_ = node_->create_publisher<aimdk_msgs::msg::JointCommandArray>(
      waist_topic_prefix + "/command", qos_cmd);
  pub_arm_   = node_->create_publisher<aimdk_msgs::msg::JointCommandArray>(
      arm_topic_prefix + "/command",   qos_cmd);
  pub_head_  = node_->create_publisher<aimdk_msgs::msg::JointCommandArray>(
      head_topic_prefix + "/command",  qos_cmd);
}

void AimdkIo::ValidateJointNames(const aimdk_msgs::msg::JointStateArray& msg,
                                 std::size_t start_mj_idx,
                                 std::size_t group_len,
                                 const char* group_name,
                                 int group_index)
{
  if (group_validated_[group_index]) return;

  if (msg.joints.size() != group_len) {
    RCLCPP_FATAL(node_->get_logger(),
                 "AimdkIo: %s state has %zu joints, expected %zu. Refusing "
                 "to ingest (would corrupt policy obs). Aborting.",
                 group_name, msg.joints.size(), group_len);
    throw std::runtime_error(std::string("AimdkIo: ") + group_name +
                             " joint count mismatch");
  }
  for (std::size_t i = 0; i < group_len; ++i) {
    const std::string  got      = msg.joints[i].name;
    const std::string  expected = mujoco_joint_names[start_mj_idx + i];
    if (got != expected) {
      RCLCPP_FATAL(node_->get_logger(),
                   "AimdkIo: %s state slot %zu is '%s' but policy expects "
                   "'%s' (mujoco_joint_names[%zu]). Refusing to silently "
                   "shuffle joints; check firmware version vs codegen.",
                   group_name, i, got.c_str(), expected.c_str(),
                   start_mj_idx + i);
      throw std::runtime_error(std::string("AimdkIo: ") + group_name +
                               " joint name mismatch at slot " +
                               std::to_string(i) + " (got '" + got +
                               "', expected '" + expected + "')");
    }
  }
  group_validated_[group_index] = true;
  RCLCPP_INFO(node_->get_logger(),
              "AimdkIo: %s joint names validated against mujoco_joint_names "
              "[%zu..%zu).",
              group_name, start_mj_idx, start_mj_idx + group_len);
}

void AimdkIo::IngestJointGroup(const aimdk_msgs::msg::JointStateArray& msg,
                               std::size_t start_mj_idx,
                               std::size_t group_len,
                               const char* group_name,
                               int group_index,
                               double& out_stamp_field,
                               bool&   out_seen_field)
{
  ValidateJointNames(msg, start_mj_idx, group_len, group_name, group_index);
  for (std::size_t i = 0; i < group_len; ++i) {
    latest_state_.joint_pos_mj[start_mj_idx + i] = msg.joints[i].position;
    latest_state_.joint_vel_mj[start_mj_idx + i] = msg.joints[i].velocity;
    latest_state_.coil_temp_c[start_mj_idx + i] =
        static_cast<double>(msg.joints[i].coil_temp);
    latest_state_.motor_temp_c[start_mj_idx + i] =
        static_cast<double>(msg.joints[i].motor_temp);
  }
  out_stamp_field = SteadyNowSec();
  out_seen_field  = true;
}

bool AimdkIo::SnapshotState(RobotState& out) const
{
  std::lock_guard<std::mutex> lk(state_mutex_);
  out = latest_state_;
  return latest_state_.leg_seen && latest_state_.waist_seen &&
         latest_state_.arm_seen && latest_state_.head_seen &&
         latest_state_.imu_seen;
}

bool AimdkIo::TrySnapshotState(RobotState& out) const
{
  std::unique_lock<std::mutex> lk(state_mutex_, std::try_to_lock);
  if (!lk.owns_lock()) return false;   // producers and control win, always
  out = latest_state_;
  return latest_state_.leg_seen && latest_state_.waist_seen &&
         latest_state_.arm_seen && latest_state_.head_seen &&
         latest_state_.imu_seen;
}

bool AimdkIo::AllStateFresh(double max_age_s) const
{
  std::lock_guard<std::mutex> lk(state_mutex_);
  if (!(latest_state_.leg_seen && latest_state_.waist_seen &&
        latest_state_.arm_seen && latest_state_.head_seen &&
        latest_state_.imu_seen)) {
    return false;
  }
  const double now = SteadyNowSec();
  const auto fresh = [&](double stamp) { return (now - stamp) <= max_age_s; };
  return fresh(latest_state_.leg_stamp_s) &&
         fresh(latest_state_.waist_stamp_s) &&
         fresh(latest_state_.arm_stamp_s) &&
         fresh(latest_state_.head_stamp_s) &&
         fresh(latest_state_.imu_stamp_s);
}

std::string AimdkIo::StateAgeReport() const
{
  std::lock_guard<std::mutex> lk(state_mutex_);
  const double now = SteadyNowSec();
  auto one = [&](const char* name, bool seen, double stamp) {
    char buf[48];
    if (!seen) {
      std::snprintf(buf, sizeof(buf), "%s=never", name);
    } else {
      std::snprintf(buf, sizeof(buf), "%s=%.3fs", name, now - stamp);
    }
    return std::string(buf);
  };
  return one("leg",   latest_state_.leg_seen,   latest_state_.leg_stamp_s)
   + " " + one("waist", latest_state_.waist_seen, latest_state_.waist_stamp_s)
   + " " + one("arm",   latest_state_.arm_seen,   latest_state_.arm_stamp_s)
   + " " + one("head",  latest_state_.head_seen,  latest_state_.head_stamp_s)
   + " " + one("imu",   latest_state_.imu_seen,   latest_state_.imu_stamp_s);
}

void AimdkIo::PublishGroup(
    rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr& pub,
    std::size_t start_mj_idx,
    std::size_t group_len,
    const std::array<double, NUM_DOFS>& target_pos_mj,
    const std::array<double, NUM_DOFS>& target_vel_mj,
    const std::array<double, NUM_DOFS>& stiffness_mj,
    const std::array<double, NUM_DOFS>& damping_mj)
{
  aimdk_msgs::msg::JointCommandArray msg;
  msg.joints.resize(group_len);
  for (std::size_t i = 0; i < group_len; ++i) {
    const std::size_t mj = start_mj_idx + i;
    auto& j = msg.joints[i];
    j.name      = mujoco_joint_names[mj];
    j.position  = target_pos_mj[mj];
    j.velocity  = target_vel_mj[mj];
    j.effort    = 0.0;
    j.stiffness = stiffness_mj[mj];
    j.damping   = damping_mj[mj];
  }
  pub->publish(msg);
}

void AimdkIo::PublishCommand(const std::array<double, NUM_DOFS>& target_pos_mj,
                             const std::array<double, NUM_DOFS>& target_vel_mj,
                             const std::array<double, NUM_DOFS>& stiffness_mj,
                             const std::array<double, NUM_DOFS>& damping_mj)
{
  PublishGroup(pub_leg_,   kLegStart,   kLegLen,
               target_pos_mj, target_vel_mj, stiffness_mj, damping_mj);
  PublishGroup(pub_waist_, kWaistStart, kWaistLen,
               target_pos_mj, target_vel_mj, stiffness_mj, damping_mj);
  PublishGroup(pub_arm_,   kArmStart,   kArmLen,
               target_pos_mj, target_vel_mj, stiffness_mj, damping_mj);
  PublishGroup(pub_head_,  kHeadStart,  kHeadLen,
               target_pos_mj, target_vel_mj, stiffness_mj, damping_mj);
  cmd_count_.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace agi_x2
