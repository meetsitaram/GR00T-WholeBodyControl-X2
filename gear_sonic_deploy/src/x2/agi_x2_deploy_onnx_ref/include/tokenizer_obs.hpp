/**
 * @file tokenizer_obs.hpp
 * @brief 680-D tokenizer observation builder. Layout deliberately mirrors a
 *        documented bug in IsaacLab's `command_multi_future_nonflat` so the
 *        deploy fed an observation that's bit-identical to what the trained
 *        policy saw during PPO.
 *
 * IsaacLab builds `command_multi_future_nonflat` via
 *   gear_sonic/envs/manager_env/mdp/observations.py:584-585
 *
 *     return command.command_multi_future.reshape(
 *         command.num_envs, command.num_future_frames, -1)   # (N, 10, 62)
 *
 * but `command.command_multi_future` is
 *   `cat([joint_pos_multi_future, joint_vel_multi_future], dim=1)`
 * with shape `(N, 2 * num_future * num_dof) == (N, 620)`, structured as
 *   [jpos_f0(31) | jpos_f1(31) | ... | jpos_f9(31)
 *    | jvel_f0(31) | jvel_f1(31) | ... | jvel_f9(31)]
 *
 * `reshape(N, 10, 62)` therefore does NOT produce per-frame
 * [jpos_fk, jvel_fk] rows. It produces the "buggy" pair-then-pair pattern:
 *   row 0: [jpos_f0(31) | jpos_f1(31)]
 *   row 1: [jpos_f2(31) | jpos_f3(31)]    ...    row 4: [jpos_f8 | jpos_f9]
 *   row 5: [jvel_f0(31) | jvel_f1(31)]    ...    row 9: [jvel_f8 | jvel_f9]
 *
 * The encoder then `cat([..._nonflat, motion_anchor_ori_b_mf_nonflat], -1)`
 * (the ori block IS per-frame), giving (N, 10, 68), and flattens row-major
 * to 680 ints into the actor obs vector.
 *
 * `gear_sonic/trl/losses/token_losses.py:72-86` documents this bug and
 * un-scrambles it on the decoder side. PPO trained the policy END-TO-END
 * against this scrambled input, so the deploy MUST reproduce the exact
 * scramble. A "correct" per-frame interleaved layout makes every frame
 * look like the wrong DOFs to the policy and the robot drifts off-pose
 * after a few seconds (caught by parity sim profile holding ~5 s before
 * collapse).
 *
 * Output layout (10 rows of 68 floats, flattened row-major to 680):
 *
 *   row k < 5:   [jpos_f(2k)(31)     | jpos_f(2k+1)(31)     | ori_fk(6)]
 *   row k >= 5:  [jvel_f(2(k-5))(31) | jvel_f(2(k-5)+1)(31) | ori_fk(6)]
 *
 * Where each future frame is sampled from the current ReferenceMotion at
 *   t_future_k = current_time + k * DT_FUTURE_REF          (k = 0..9)
 * and the 6-D rotation diff per frame is `mat[:, :2].reshape(-1)` (row-major
 * flatten of the first two columns of the rotmat) of
 *   M_k = (cur_root_quat^-1 * future_root_quat).as_matrix()
 * matching IsaacLab's `motion_anchor_ori_b_mf` builder, see
 * math_utils::rot6d_from_quat_xyzw.
 *
 * Reference (Python eval that holds 30 s on iter-4000):
 *   gear_sonic/scripts/eval_x2_mujoco.py:620-629
 */

#ifndef AGI_X2_TOKENIZER_OBS_HPP
#define AGI_X2_TOKENIZER_OBS_HPP

#include "policy_parameters.hpp"
#include "reference_motion.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace agi_x2 {

constexpr std::size_t TOK_DIM           = 680;
constexpr std::size_t TOK_CMD_FLAT_DIM  = NUM_FUTURE_FRAMES * (NUM_DOFS + NUM_DOFS);  // 620
constexpr std::size_t TOK_ORI_FLAT_DIM  = NUM_FUTURE_FRAMES * 6;                       // 60
static_assert(TOK_CMD_FLAT_DIM + TOK_ORI_FLAT_DIM == TOK_DIM,
              "tokenizer obs widths do not sum to 680");

/**
 * Build the 680-D tokenizer observation in IsaacLab's buggy
 * pair-of-jpos-then-pair-of-jvel layout (see file-level docstring above).
 *
 * @param ref          reference-motion source (StandStill or PklMotion)
 * @param current_time seconds since CONTROL state entry (matches Sample(t)
 *                     contract used elsewhere in the deploy)
 * @param base_quat_wxyz current robot torso orientation, MuJoCo / IMU convention
 * @param override_ref_quat_xyzw OPTIONAL bootstrap-safe substitute for the
 *                     reference frame's root quat. When provided, every
 *                     future frame sampled from ``ref`` has its
 *                     ``root_quat_xyzw`` replaced by this value BEFORE the
 *                     ``rel = inv(cur) * ref`` rotation diff is computed.
 *                     Passing ``base_quat_xyzw`` (i.e. the current measured
 *                     orientation) here yields ``rel = identity`` for all
 *                     10 future slots, which collapses the 6-D ori block
 *                     to a constant. The deploy hands this in when the
 *                     ZMQ pose-ref input source has never received a
 *                     real frame (LastReceivedMonotonicS() < 0): without
 *                     it, the ``ZmqPoseInputSource::Connect()`` bootstrap
 *                     pre-fill of ``root_quat_xyzw = (0,0,0,1)`` becomes
 *                     a fixed world-+X reference and the policy actively
 *                     twists the body back to spawn heading on every
 *                     tick (the "robot recovers to a fixed world
 *                     orientation" symptom). Default ``std::nullopt``
 *                     preserves pre-fix behaviour for the unit tests
 *                     and any caller that already has a fresh reference.
 * @return float vector of length TOK_DIM, ready to feed the ONNX session
 *         alongside the 990-D proprioception.
 */
std::vector<float> BuildTokenizerObs(
    const ReferenceMotion& ref,
    double current_time,
    const std::array<double, 4>& base_quat_wxyz,
    const std::optional<std::array<double, 4>>& override_ref_quat_xyzw
        = std::nullopt);

}  // namespace agi_x2

#endif  // AGI_X2_TOKENIZER_OBS_HPP
