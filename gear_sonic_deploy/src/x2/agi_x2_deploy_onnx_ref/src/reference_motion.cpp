#include "reference_motion.hpp"

#include "math_utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace agi_x2 {

// ---------------------------------------------------------------------------
// StandStillReference
// ---------------------------------------------------------------------------
StandStillReference::StandStillReference()
{
  // Joint targets = trained default standing pose (the same constants the
  // 2 s soft-start ramp converges to in safety.cpp).
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    frame_.joint_pos_mj[i] = default_angles[i];
    frame_.joint_vel_mj[i] = 0.0;
  }
  // Identity quaternion (xyzw): no rotation -> 6D-rot diff is identity, so
  // the tokenizer's ori_per_frame block is all-zeros except for the two
  // diagonal entries. Matches what the trainer sees on a perfectly upright
  // standing reference.
  frame_.root_quat_xyzw = {0.0, 0.0, 0.0, 1.0};
}

ReferenceFrame StandStillReference::Sample(double /*time*/) const
{
  return frame_;
}

// ---------------------------------------------------------------------------
// PklMotionReference loader (X2M2 binary format -- see header docstring)
// ---------------------------------------------------------------------------
namespace {

constexpr std::uint32_t kX2M2Magic = 0x58324D32u;  // "X2M2" little-endian

template <typename T>
T ReadPod(std::istream& s)
{
  T value{};
  s.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!s) {
    throw std::runtime_error("PklMotionReference: short read on binary motion file");
  }
  return value;
}

}  // namespace

std::unique_ptr<PklMotionReference> PklMotionReference::Load(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("PklMotionReference: cannot open " + path);
  }

  const auto magic      = ReadPod<std::uint32_t>(f);
  const auto num_frames = ReadPod<std::uint32_t>(f);
  const auto num_dofs   = ReadPod<std::uint32_t>(f);
  const auto fps        = ReadPod<double>(f);

  if (magic != kX2M2Magic) {
    throw std::runtime_error("PklMotionReference: bad magic in " + path +
                             " (expected X2M2 = 0x58324D32, got " +
                             std::to_string(magic) +
                             "). Did you regenerate via "
                             "export_motion_for_deploy.py?");
  }
  if (num_dofs != NUM_DOFS) {
    throw std::runtime_error(
        "PklMotionReference: file has " + std::to_string(num_dofs) +
        " DOFs but X2 deploy expects " + std::to_string(NUM_DOFS));
  }
  if (num_frames == 0) {
    throw std::runtime_error("PklMotionReference: zero-frame motion file");
  }

  std::unique_ptr<PklMotionReference> ref(new PklMotionReference);
  ref->name_       = path;
  ref->num_frames_ = num_frames;
  ref->fps_        = fps;
  ref->jpos_mj_.resize(num_frames);
  ref->root_quat_xyzw_.resize(num_frames);

  for (std::size_t i = 0; i < num_frames; ++i) {
    f.read(reinterpret_cast<char*>(ref->jpos_mj_[i].data()),
           sizeof(double) * NUM_DOFS);
    f.read(reinterpret_cast<char*>(ref->root_quat_xyzw_[i].data()),
           sizeof(double) * 4);
    if (!f) {
      throw std::runtime_error("PklMotionReference: truncated motion file at frame " +
                               std::to_string(i) + "/" + std::to_string(num_frames));
    }
  }

  return ref;
}

ReferenceFrame PklMotionReference::Sample(double time) const
{
  // Looped playback. We compute joint velocity by finite differencing the
  // file (rather than storing it) so the binary format stays compact and
  // matches eval_x2_mujoco's compute_motion_state semantics.
  //
  // CRITICAL: use BACKWARD difference (current - previous), not forward,
  // to match `gear_sonic/scripts/eval_x2_mujoco.py:596-598`:
  //   prev_fi = max(0, fi - 1)
  //   jvel_mj = (m["dof"][fi] - m["dof"][prev_fi]) * motion_fps
  //
  // Forward difference (next - current) gives the same magnitude on
  // monotonic / smooth motions but lags by one frame, which shows up as
  // ~0.1 rad/s discrepancies on first-tick obs diffs against the Python
  // ground truth and accumulates into a few-rad offset by the time the
  // policy needs to make a balance correction. Mismatched jvel was the
  // residual after fixing the tokenizer layout (see tokenizer_obs.cpp
  // bug history #4).
  const double frame_f = time * fps_;
  const auto   N       = static_cast<long long>(num_frames_);
  long long    f_idx   = static_cast<long long>(frame_f) % N;
  if (f_idx < 0) f_idx += N;          // C++ % for negative inputs
  // Backward neighbour, clamped at 0 to mirror `max(0, fi - 1)` in Python.
  const long long f_prev = (f_idx > 0) ? (f_idx - 1) : 0;
  const double    dt     = 1.0 / fps_;

  ReferenceFrame out;
  out.joint_pos_mj = jpos_mj_[f_idx];
  // Yaw-anchor: rotate the recorded root quat into the robot's heading
  // frame. yaw_anchor_xyzw_ defaults to identity, so this is a no-op until
  // Anchor() is called on CONTROL entry. See header for rationale.
  out.root_quat_xyzw = quat_mul_xyzw(yaw_anchor_xyzw_, root_quat_xyzw_[f_idx]);
  for (std::size_t d = 0; d < NUM_DOFS; ++d) {
    out.joint_vel_mj[d] = (jpos_mj_[f_idx][d] - jpos_mj_[f_prev][d]) / dt;
  }
  return out;
}

void PklMotionReference::Anchor(const std::array<double, 4>& robot_quat_wxyz)
{
  if (root_quat_xyzw_.empty()) {
    // Defensive: should be impossible because Load() rejects zero-frame
    // files, but keep Anchor() total just in case.
    yaw_anchor_xyzw_  = { 0.0, 0.0, 0.0, 1.0 };
    yaw_anchor_delta_ = 0.0;
    return;
  }
  const double yaw_robot  = yaw_from_quat_wxyz(robot_quat_wxyz);
  const double yaw_motion = yaw_from_quat_xyzw(root_quat_xyzw_[0]);
  // Δyaw = robot - motion: pre-multiplying motion[0] by Rz(Δyaw) makes its
  // yaw equal to the robot's current yaw, leaving the motion's pitch/roll
  // (which are gravity-grounded) untouched.
  yaw_anchor_delta_ = yaw_robot - yaw_motion;
  yaw_anchor_xyzw_  = yaw_quat_xyzw(yaw_anchor_delta_);
}

}  // namespace agi_x2
