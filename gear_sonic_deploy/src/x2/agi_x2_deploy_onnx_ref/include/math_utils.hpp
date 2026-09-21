/**
 * @file math_utils.hpp
 * @brief Minimal quaternion / vector helpers used by the X2 obs builder.
 *
 * We only need the operations the Python obs builder uses:
 *   - quat_rotate_inverse(q_wxyz, v)        -> body-frame projection
 *   - quat_relative(q_a_xyzw, q_b_xyzw)     -> q_a^-1 * q_b
 *   - rot6d_from_quat(q_xyzw)               -> first two rows of the matrix
 *
 * All quaternion conventions are explicitly tagged in the function names
 * (`_wxyz` vs `_xyzw`) because the X2 PKLs are scipy-style xyzw while
 * MuJoCo / IMU messages are wxyz, and silently mixing them is the most
 * likely source of "policy looks fine but robot tilts" bugs.
 */

#ifndef AGI_X2_MATH_UTILS_HPP
#define AGI_X2_MATH_UTILS_HPP

#include <array>
#include <cmath>

namespace agi_x2 {

/// Rotate vector v by INVERSE of quaternion q (wxyz convention).
/// Mirrors gear_sonic/scripts/eval_x2_mujoco.py::quat_rotate_inverse, which
/// in turn matches IsaacLab's quat_apply_inverse: v - w*t + cross(u, t)
/// with t = 2 * cross(u, v) and u = (qx, qy, qz).
inline std::array<double, 3> quat_rotate_inverse_wxyz(
    const std::array<double, 4>& q_wxyz,
    const std::array<double, 3>& v)
{
  const double w = q_wxyz[0];
  const double ux = q_wxyz[1], uy = q_wxyz[2], uz = q_wxyz[3];

  // t = 2 * cross(u, v)
  const double tx = 2.0 * (uy * v[2] - uz * v[1]);
  const double ty = 2.0 * (uz * v[0] - ux * v[2]);
  const double tz = 2.0 * (ux * v[1] - uy * v[0]);

  // out = v - w*t + cross(u, t)
  return {
    v[0] - w * tx + (uy * tz - uz * ty),
    v[1] - w * ty + (uz * tx - ux * tz),
    v[2] - w * tz + (ux * ty - uy * tx),
  };
}

/// Project gravity direction [0,0,-1] world into the body frame.
/// This is what ProprioceptionBuffer.gravity_hist stores.
inline std::array<double, 3> body_frame_gravity_from_quat_wxyz(
    const std::array<double, 4>& q_wxyz)
{
  return quat_rotate_inverse_wxyz(q_wxyz, {0.0, 0.0, -1.0});
}

/// Quaternion multiply (xyzw): out = a * b. Used to compute relative rotations.
inline std::array<double, 4> quat_mul_xyzw(
    const std::array<double, 4>& a,
    const std::array<double, 4>& b)
{
  const double ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const double bx = b[0], by = b[1], bz = b[2], bw = b[3];
  return {
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  };
}

/// Conjugate of a unit quaternion (xyzw): just negate the vector part.
inline std::array<double, 4> quat_conj_xyzw(const std::array<double, 4>& q)
{
  return { -q[0], -q[1], -q[2], q[3] };
}

/// Convert wxyz quaternion to xyzw (scipy / PKL convention).
inline std::array<double, 4> wxyz_to_xyzw(const std::array<double, 4>& q_wxyz)
{
  return { q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0] };
}

/// Convert xyzw quaternion to wxyz (MuJoCo / IMU convention).
inline std::array<double, 4> xyzw_to_wxyz(const std::array<double, 4>& q_xyzw)
{
  return { q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2] };
}

/// Build a 3x3 rotation matrix from a unit quaternion (xyzw, scipy ordering).
/// Returns row-major: out[row*3 + col]. Matches scipy.spatial.transform.Rotation.
inline std::array<double, 9> rotation_matrix_xyzw(
    const std::array<double, 4>& q_xyzw)
{
  const double x = q_xyzw[0], y = q_xyzw[1], z = q_xyzw[2], w = q_xyzw[3];
  const double xx = x * x, yy = y * y, zz = z * z;
  const double xy = x * y, xz = x * z, yz = y * z;
  const double wx = w * x, wy = w * y, wz = w * z;
  return {
    1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
    2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
    2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy),
  };
}

/// 6D rotation representation: first two COLUMNS of the rotation matrix,
/// flattened row-major. Matches IsaacLab's ``motion_anchor_ori_b_mf`` term
/// (gear_sonic/envs/manager_env/mdp/observations.py::motion_anchor_ori_b_mf
/// -> commands.py::root_rot_dif_l_multi_future), which is:
///
///     mat = matrix_from_quat(root_rot_dif).reshape(B, F, 3, 3)
///     rot6 = mat[..., :2].reshape(B, -1)               # first 2 COLUMNS
///
/// Because PyTorch ``mat[..., :2]`` slices the LAST dim of (3, 3), the
/// resulting (3, 2) sub-tensor is the first two COLUMNS of the rotmat
/// (row-index varies fastest under the implicit C-order flatten that
/// follows). Flattened row-major it is:
///
///     [m[0,0], m[0,1], m[1,0], m[1,1], m[2,0], m[2,1]]
///
/// rotation_matrix_xyzw above stores out[row*3 + col] (row-major), so:
///
///     m[0,0] = m[0],  m[0,1] = m[1]
///     m[1,0] = m[3],  m[1,1] = m[4]
///     m[2,0] = m[6],  m[2,1] = m[7]
///
/// Bug history: the previous implementation returned the first two ROWS
/// (m[0..5] = m[0,0], m[0,1], m[0,2], m[1,0], m[1,1], m[1,2]). For the
/// identity rotation that happens to equal ``[1, 0, 0, 0, 1, 0]`` which is
/// also what columns-flat returns for identity, so the unit test in
/// test_obs_builder.cpp passed. Off-axis the two reps disagree: e.g. for a
/// 90-degree yaw the rows-flat rep is [0,-1,0, 1,0,0] while the columns-
/// flat rep is [0,1,0, -1,0,0]. The trained policy expected columns-flat,
/// so the rows-flat C++ output left ``motion_anchor_ori_b_mf`` 6D garbage
/// whenever the reference motion's frame deviated from the robot's by more
/// than a few degrees. This was caught by the slot diff in
/// gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py.
inline std::array<double, 6> rot6d_from_quat_xyzw(
    const std::array<double, 4>& q_xyzw)
{
  const auto m = rotation_matrix_xyzw(q_xyzw);
  return { m[0], m[1], m[3], m[4], m[6], m[7] };
}

/// Extract the world-frame yaw (heading) angle from a unit quaternion (xyzw).
///
/// We compute the world-frame forward vector that the robot's body-X axis
/// gets rotated into by the quaternion (= first column of the rotmat), then
/// take atan2(forward.y, forward.x). This is more robust to non-zero
/// pitch/roll than the naive 2*atan2(qz, qw) shortcut: for a robot tilted
/// 10 deg in pitch the shortcut starts losing yaw resolution; this version
/// stays well-behaved as long as the body-X axis isn't pointing nearly
/// vertical (i.e. the robot isn't lying on its face).
inline double yaw_from_quat_xyzw(const std::array<double, 4>& q_xyzw)
{
  const double x = q_xyzw[0], y = q_xyzw[1], z = q_xyzw[2], w = q_xyzw[3];
  const double fx = 1.0 - 2.0 * (y * y + z * z);
  const double fy = 2.0 * (x * y + w * z);
  return std::atan2(fy, fx);
}

/// Extract the world-frame yaw from a unit quaternion (wxyz, IMU convention).
inline double yaw_from_quat_wxyz(const std::array<double, 4>& q_wxyz)
{
  return yaw_from_quat_xyzw(wxyz_to_xyzw(q_wxyz));
}

/// Build a pure-yaw rotation quaternion (xyzw) Rz(yaw_rad). For yaw=0 this
/// returns the identity quaternion (0,0,0,1). Pre-multiplying any quat by
/// this rotates it about the world-Z axis by ``yaw_rad`` without touching
/// pitch or roll.
inline std::array<double, 4> yaw_quat_xyzw(double yaw_rad)
{
  const double half = 0.5 * yaw_rad;
  return { 0.0, 0.0, std::sin(half), std::cos(half) };
}

/// Strip pitch/roll from a quaternion, keeping only its world-Z heading.
/// Equivalent to Rz(yaw(q)). Used by the "heading" anchor-ori convention:
/// normalizing the reference by THIS instead of the robot's full orientation
/// preserves the reference motion's pitch/roll relative to gravity, which is
/// what every v1.1-lineage policy was trained on. See tokenizer_obs.cpp.
/// Heading (yaw-only) quaternion, MATCHING TRAINING EXACTLY.
///
/// Training builds ``motion_anchor_ori_heading_mf`` from
/// ``commands.py::anchor_heading_quat`` -> ``torch_transform.get_heading_q``,
/// which is a TWIST extraction: zero the x,y components of the (w,x,y,z)
/// quaternion and renormalize. That is NOT Euler yaw.
///
///   get_heading_q : q_new = [w, 0, 0, z] / ||[w, 0, 0, z]||
///                   (equivalently a rotation of 2*atan2(z, w) about Z)
///   Euler yaw     : atan2(2(wz + xy), 1 - 2(y^2 + z^2))
///
/// These agree exactly at zero tilt and diverge as the robot tilts --
/// measured on real robot IMU: mean 0.007 deg, max 0.703 deg over 5 min of
/// walking (tilt max 15.8 deg), but 1.3-4.0 deg at combined pitch+roll of
/// 15-30 deg, i.e. during a fall or a recovery, which is precisely when a
/// wrong orientation reference is least affordable.
///
/// THIS FUNCTION ORIGINALLY USED EULER YAW (2026-08-22) because it was
/// matched to torch_transform.py::yaw_quat -- a real function in the
/// codebase, but NOT the one this observation is built from. The error was
/// too small to explain any observed fault, yet cost significant time during
/// the 2026-08-23 investigation precisely because it could not be dismissed
/// by inspection. Keeping a known-wrong-but-small divergence makes every
/// future debugging session harder; hence this is now exact.
inline std::array<double, 4> yaw_only_quat_xyzw(
    const std::array<double, 4>& q_xyzw)
{
  const double z = q_xyzw[2], w = q_xyzw[3];
  const double n = std::sqrt(w * w + z * z);
  // Degenerate only if the body is rotated exactly 180 deg about a
  // horizontal axis (w == z == 0) -- upside down. get_heading_q's own
  // docstring flags this as ill-defined and notes it "does not often happen
  // for humanoid robots". Fall back to identity rather than emit NaN.
  if (n < 1e-12) {
    return { 0.0, 0.0, 0.0, 1.0 };
  }
  return { 0.0, 0.0, z / n, w / n };
}

}  // namespace agi_x2

#endif  // AGI_X2_MATH_UTILS_HPP
