/**
 * @file aimdk_io.hpp
 * @brief ROS 2 IO wrapper for the X2 Ultra deploy.
 *
 * Topic registry: matches the canonical names in the lx2501_3 SDK
 * `topics_and_services` file and `dev/Interface/control_mod/joint_control.html`
 * / `dev/Interface/hal/sensor.html` documentation.
 *
 * Subscriptions (all aimdk_msgs::msg::JointStateArray):
 *   /aima/hal/joint/leg/state    -> 12 joints (left+right legs)
 *   /aima/hal/joint/waist/state  -> 3 joints  (yaw, pitch, roll)
 *   /aima/hal/joint/arm/state    -> 14 joints (left+right arms)
 *   /aima/hal/joint/head/state   -> 2 joints  (yaw, pitch)
 *
 * Publications (aimdk_msgs::msg::JointCommandArray):
 *   /aima/hal/joint/leg/command
 *   /aima/hal/joint/waist/command
 *   /aima/hal/joint/arm/command
 *   /aima/hal/joint/head/command
 *
 * IMU subscription (sensor_msgs::msg::Imu):
 *   /aima/hal/imu/torso/state    -> torso IMU (orientation + angular velocity)
 *
 * NB on the IMU spelling: the official `topics_and_services` registry, the
 * Sphinx docs (sensor.html), the `agibot-x2-monitor` bridge config, and
 * AgiBot's UI all use `torso`. Some SDK example sources (echo_imu_data.cpp,
 * echo_imu_data.py) contain the typo `torse` in comments. If the firmware on
 * a particular X2 actually publishes to `torse`, override at construction
 * time (or via the deploy node's `--imu-topic` CLI flag).
 *
 * NB on head pitch: per `joint_control.html` the head group accepts 2 joints
 * (yaw, pitch) but only `head_yaw_joint` is physically actuated on current
 * X2 Ultra firmware ("only yaw now, and pitch is unavailable"). The 2-DOF
 * shape is preserved end-to-end so commands published into head_pitch are
 * silently dropped by the firmware.
 *
 * The on-bot publication order on each topic matches the motocontrol.cpp
 * reference example, which is verified to coincide with the IsaacSim/MuJoCo
 * URDF-tree order for the X2 Ultra. This means the assembled 31-D vector
 * is already in MuJoCo joint order and can be IL-remapped via
 * mujoco_to_isaaclab[] from policy_parameters.hpp without any string lookup
 * on the hot path.
 *
 * If a future firmware shuffles the per-topic order, this module will fail
 * loudly at the first state message via JointNameMismatchError -- it always
 * cross-checks the name strings on the inaugural callback.
 */

#ifndef AGI_X2_AIMDK_IO_HPP
#define AGI_X2_AIMDK_IO_HPP

#include "policy_parameters.hpp"

#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>

namespace agi_x2 {

/// Snapshot of robot state at a single instant, all 31 joints assembled in
/// MuJoCo joint order (matches `mujoco_joint_names[]`).
struct RobotState {
  std::array<double, NUM_DOFS> joint_pos_mj{};   ///< rad
  std::array<double, NUM_DOFS> joint_vel_mj{};   ///< rad/s
  // Motor thermals (deg C; 2026-08-04 G1-parity audit #2 — the wire always
  // carried these, IngestJointGroup previously discarded them).
  std::array<double, NUM_DOFS> coil_temp_c{};    ///< winding temperature
  std::array<double, NUM_DOFS> motor_temp_c{};   ///< housing temperature
  std::array<double, 4>        base_quat_wxyz{1.0, 0.0, 0.0, 0.0}; ///< IMU orientation
  std::array<double, 3>        base_ang_vel{0, 0, 0};              ///< rad/s, body frame

  // Per-source freshness timestamps (steady_clock seconds since epoch).
  // Used by AimdkIo::AllStateFresh to gate the INIT->WAIT transition.
  double leg_stamp_s   = 0.0;
  double waist_stamp_s = 0.0;
  double arm_stamp_s   = 0.0;
  double head_stamp_s  = 0.0;
  double imu_stamp_s   = 0.0;

  // Set on the first callback of each kind, never cleared.
  bool   leg_seen   = false;
  bool   waist_seen = false;
  bool   arm_seen   = false;
  bool   head_seen  = false;
  bool   imu_seen   = false;
};

/// Indices into mujoco_joint_names[] for each AimDK group, in the order the
/// firmware publishes them. These ranges add up to 31.
constexpr std::size_t kLegStart   = 0,  kLegLen   = 12;  // [0,  12)
constexpr std::size_t kWaistStart = 12, kWaistLen = 3;   // [12, 15)
constexpr std::size_t kArmStart   = 15, kArmLen   = 14;  // [15, 29)
constexpr std::size_t kHeadStart  = 29, kHeadLen  = 2;   // [29, 31)
static_assert(kHeadStart + kHeadLen == NUM_DOFS,
              "AimDK joint group ranges must cover all 31 X2 DOFs");

class AimdkIo {
 public:
  /// @param node Shared node the IO subs/pubs attach to (must outlive AimdkIo).
  /// @param leg/waist/arm/head_topic_prefix overrides for testing on a
  ///        rosbag with remapped topics. Pass empty strings to use defaults.
  explicit AimdkIo(rclcpp::Node::SharedPtr node,
                   const std::string& leg_topic_prefix   = "/aima/hal/joint/leg",
                   const std::string& waist_topic_prefix = "/aima/hal/joint/waist",
                   const std::string& arm_topic_prefix   = "/aima/hal/joint/arm",
                   const std::string& head_topic_prefix  = "/aima/hal/joint/head",
                   const std::string& imu_topic          = "/aima/hal/imu/torso/state");

  /// Thread-safe copy of the most recent assembled state. Returns true if
  /// every source has produced at least one message; false otherwise (the
  /// snapshot is still valid but some fields are still defaults).
  bool SnapshotState(RobotState& out) const;

  /// NON-BLOCKING SnapshotState, for TELEMETRY ONLY.
  ///
  /// state_mutex_ is the single lock guarding robot state: the five DDS
  /// callbacks take it to WRITE incoming state, and the 50 Hz control loop
  /// takes it to READ the policy's observation. Anything sampling at a higher
  /// rate must never block those two. The 500 Hz flight recorder did exactly
  /// that on 2026-08-25 and starved the callbacks -- every source read stale
  /// within 0.6 s of CONTROL and the run latched SAFE_HOLD.
  ///
  /// Returns false (writing nothing) if the lock is held. A dropped telemetry
  /// sample is free; a delayed state callback is not.
  bool TrySnapshotState(RobotState& out) const;

  /// True iff every source has reported within `max_age_s` seconds of now.
  /// Use this to gate INIT -> WAIT_FOR_CONTROL.
  bool AllStateFresh(double max_age_s = 0.5) const;

  // Human-readable per-stream state ages ("leg=0.002s waist=0.541s ...";
  // "never" for streams not yet seen). For diagnosing WHICH stream
  // tripped the CONTROL staleness guard (2026-08-04 incident: guard
  // tripped on real hardware with no per-stream evidence).
  std::string StateAgeReport() const;

  /// Publish a 31-D MuJoCo-ordered command in one shot. The MJ vector is
  /// sliced according to the kLeg/kWaist/kArm/kHead constants and fanned
  /// out to the four AimDK publishers.
  ///
  /// @param target_pos_mj  31-D PD target positions, MJ order, rad
  /// @param target_vel_mj  31-D feed-forward velocities (zeros are fine), MJ order, rad/s
  /// @param stiffness_mj   31-D per-joint kp; for the trained policy use kps[]
  /// @param damping_mj     31-D per-joint kd; for the trained policy use kds[]
  ///                       (use zeros for dry-run mode)
  void PublishCommand(const std::array<double, NUM_DOFS>& target_pos_mj,
                      const std::array<double, NUM_DOFS>& target_vel_mj,
                      const std::array<double, NUM_DOFS>& stiffness_mj,
                      const std::array<double, NUM_DOFS>& damping_mj);

  /// Counter of commands actually published since startup. Useful for the
  /// 500 Hz watchdog to confirm the writer thread is alive.
  std::uint64_t commands_published() const { return cmd_count_.load(); }

 private:
  rclcpp::Node::SharedPtr node_;

  // Subscriptions
  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr sub_leg_;
  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr sub_waist_;
  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr sub_arm_;
  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr sub_head_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr            sub_imu_;

  // Publishers
  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr  pub_leg_;
  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr  pub_waist_;
  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr  pub_arm_;
  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr  pub_head_;

  // State subscriptions live in their OWN Reentrant callback group.
  //
  // Everything in this package previously used the node's DEFAULT group, which
  // rclcpp makes MutuallyExclusive -- so despite the MultiThreadedExecutor the
  // header comment advertises ("subscribers can deliver messages while the
  // control timer is mid-inference"), they could NOT. The 50 Hz control timer,
  // the 500 Hz writer and all five state subscriptions were serialised through
  // one slot. On 2026-08-25 a control loop running 22-36 ms against a 20 ms
  // budget saturated that slot: the writer fell to 3.4 Hz (from 500) and the
  // subscriptions starved until every source read stale -> SAFE_HOLD.
  //
  // Reentrant + MultiThreadedExecutor lets state delivery proceed during
  // inference, which is what the design always claimed. state_mutex_ is what
  // makes that safe, and is now genuinely contended -- hence TrySnapshotState
  // for the 500 Hz telemetry path.
  rclcpp::CallbackGroup::SharedPtr state_cbg_;
  mutable std::mutex   state_mutex_;
  RobotState           latest_state_;
  std::atomic<uint64_t> cmd_count_{0};

  // First-time joint-name validation (done once per group).
  std::array<bool, 4>  group_validated_{false, false, false, false};
  void ValidateJointNames(const aimdk_msgs::msg::JointStateArray& msg,
                          std::size_t start_mj_idx,
                          std::size_t group_len,
                          const char* group_name,
                          int group_index);

  void IngestJointGroup(const aimdk_msgs::msg::JointStateArray& msg,
                        std::size_t start_mj_idx,
                        std::size_t group_len,
                        const char* group_name,
                        int group_index,
                        double& out_stamp_field,
                        bool&   out_seen_field);

  void PublishGroup(rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr& pub,
                    std::size_t start_mj_idx,
                    std::size_t group_len,
                    const std::array<double, NUM_DOFS>& target_pos_mj,
                    const std::array<double, NUM_DOFS>& target_vel_mj,
                    const std::array<double, NUM_DOFS>& stiffness_mj,
                    const std::array<double, NUM_DOFS>& damping_mj);
};

}  // namespace agi_x2

#endif  // AGI_X2_AIMDK_IO_HPP
