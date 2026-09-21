/**
 * @file reference_motion.hpp
 * @brief Reference-trajectory source for the 680-D tokenizer obs.
 *
 * The X2 g1 tokenizer encoder consumes a 10-frame future-reference window
 * sampled at 0.1 s spacing. Each future frame contributes:
 *   - 31-DOF joint position (IL order)
 *   - 31-DOF joint velocity (IL order)
 *   - 6-D rotation diff (cur_root_rot^-1 * future_root_rot, first two rows
 *     of the matrix)
 *
 * On the real robot we want to swap reference sources without recompiling:
 *   - StandStill  -- always returns DEFAULT_DOF + identity quat (good for
 *                    the first dry-run / standing test on a gantry)
 *   - PklPlayback -- replays a recorded .pkl motion (training data style)
 *   - LiveTopic   -- subscribes to an external command topic (future work)
 *
 * This header defines the abstract interface plus the StandStill default.
 * PklPlayback lives in reference_motion.cpp behind the same interface.
 */

#ifndef AGI_X2_REFERENCE_MOTION_HPP
#define AGI_X2_REFERENCE_MOTION_HPP

#include "policy_parameters.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace agi_x2 {

/// Snapshot of the reference robot at a single point in time.
/// Joint positions / velocities are in **MuJoCo** order; the tokenizer
/// builder is responsible for IL-remapping (that way every reference
/// implementation can stay in the natural URDF order).
struct ReferenceFrame {
  std::array<double, NUM_DOFS> joint_pos_mj;  ///< rad
  std::array<double, NUM_DOFS> joint_vel_mj;  ///< rad/s
  std::array<double, 4>        root_quat_xyzw;///< scipy convention (matches PKL)
};

/// Abstract interface: produce the reference robot state at any time.
/// `time` is monotonically increasing seconds since the deploy harness
/// transitioned into CONTROL state.
class ReferenceMotion {
 public:
  virtual ~ReferenceMotion() = default;
  virtual ReferenceFrame Sample(double time) const = 0;
  virtual std::string    Name()                const = 0;

  /// "Virtual RSI": align the recorded motion's world frame to the robot's
  /// current heading at the moment we hand control to the policy.
  ///
  /// In IsaacLab training every episode begins with Reference State
  /// Initialization: the simulator teleports the robot's root pose to match
  /// the motion's frame 0, so the policy never sees a state where its body
  /// is rotated relative to the motion's heading. On real hardware we can't
  /// teleport; the robot starts with whatever yaw the gantry happens to
  /// leave it at, while the motion file's frame-0 yaw is whatever absolute
  /// world yaw the recording was captured in (often arbitrary -- e.g. ~96
  /// deg for x2_ultra_idle_stand.pkl). The deploy must therefore re-anchor
  /// the motion clip's heading on entry to CONTROL state.
  ///
  /// Concretely, an implementation should:
  ///   1. Read the robot's current root orientation (wxyz, IMU convention).
  ///   2. Read its own frame-0 root orientation (xyzw, scipy/PKL).
  ///   3. Compute Δyaw = yaw(robot) - yaw(motion[0]).
  ///   4. Pre-multiply every subsequently sampled root_quat_xyzw by Rz(Δyaw)
  ///      so motion frame 0 has the same yaw as the robot's current heading
  ///      while preserving the motion's pitch/roll (gravity-grounded).
  ///
  /// We anchor yaw ONLY: pitch and roll are absolute (gravity points down
  /// regardless of world frame), so we leave them as the motion recorded.
  /// Anchoring those would lie about the robot's tilt to the policy.
  ///
  /// Default implementation is a no-op (correct for StandStillReference,
  /// whose motion is identity by construction).
  virtual void Anchor(const std::array<double, 4>& /*robot_quat_wxyz*/) {}

  /// Yaw delta (radians) the most recent Anchor() call applied. 0 by
  /// default; only PklMotionReference currently maintains a meaningful
  /// value. Exposed on the base so the deploy harness can log it without a
  /// downcast.
  virtual double yaw_anchor_delta() const { return 0.0; }
};

/// Trivial reference: always the trained standing pose. Use this for the
/// first real-robot dry run -- if it can hold the standing pose for 5 s on
/// a gantry, we know the entire wiring (obs + ONNX + IO + PD) is sound and
/// can graduate to PKL playback.
class StandStillReference : public ReferenceMotion {
 public:
  StandStillReference();
  ReferenceFrame Sample(double time) const override;
  std::string    Name()              const override { return "stand_still"; }

 private:
  ReferenceFrame frame_;  ///< constructed once at startup, returned forever
};

/// Replay a recorded motion from a flat binary file. The file format is
/// produced by `gear_sonic_deploy/scripts/export_motion_for_deploy.py` (a
/// Python helper that resolves the joblib/pickle motion-lib PKLs into a
/// dependency-free C++ readable layout):
///
///   uint32_t  magic       == 0x58324D32  ("X2M2")
///   uint32_t  num_frames
///   uint32_t  num_dofs    (must equal 31)
///   double    fps
///   For each frame f in [0, num_frames):
///     double  joint_pos_mj[31]
///     double  root_quat_xyzw[4]
///   Joint velocity is reconstructed at runtime via finite difference.
///
/// On Sample(t), the file is treated as a looped trajectory: the frame
/// nearest `(t * fps) mod num_frames` is returned.
class PklMotionReference : public ReferenceMotion {
 public:
  /// Throws std::runtime_error on malformed file or DOF mismatch.
  static std::unique_ptr<PklMotionReference> Load(const std::string& path);

  ReferenceFrame Sample(double time) const override;
  std::string    Name()              const override { return name_; }

  /// Compute Δyaw between the robot's current heading and the motion's
  /// frame-0 heading, store it as a pre-multiplied yaw quaternion, and apply
  /// it to every subsequent Sample() call. See base-class docstring for the
  /// full rationale (essentially: virtual RSI on a real robot).
  void Anchor(const std::array<double, 4>& robot_quat_wxyz) override;

  std::size_t num_frames() const { return num_frames_; }
  double      fps()        const { return fps_; }

  /// Yaw delta (radians) the most recent Anchor() call applied. 0.0 if
  /// Anchor() was never called. Useful for logging from the deploy harness.
  double yaw_anchor_delta() const override { return yaw_anchor_delta_; }

 private:
  PklMotionReference() = default;

  std::string name_;
  std::size_t num_frames_ = 0;
  double      fps_        = 30.0;
  std::vector<std::array<double, NUM_DOFS>> jpos_mj_;
  std::vector<std::array<double, 4>>        root_quat_xyzw_;

  // Yaw-anchor state. Identity quaternion means "no anchor applied yet".
  // Sample() unconditionally pre-multiplies root_quat_xyzw_[f] by this; the
  // initial identity makes that a no-op pre-Anchor().
  std::array<double, 4> yaw_anchor_xyzw_  = { 0.0, 0.0, 0.0, 1.0 };
  double                yaw_anchor_delta_ = 0.0;
};

}  // namespace agi_x2

#endif  // AGI_X2_REFERENCE_MOTION_HPP
