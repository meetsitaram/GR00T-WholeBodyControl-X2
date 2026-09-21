#include "tokenizer_obs.hpp"

#include "math_utils.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace agi_x2 {

std::vector<float> BuildTokenizerObs(
    const ReferenceMotion& ref,
    double current_time,
    const std::array<double, 4>& base_quat_wxyz,
    const std::optional<std::array<double, 4>>& override_ref_quat_xyzw)
{
  std::vector<float> out;
  out.reserve(TOK_DIM);

  // Pre-compute the normalizing rotation's conjugate (xyzw), used for the
  // relative rotation norm^-1 * future across all 10 future frames.
  //
  // ANCHOR-ORI CONVENTION -- TWO VARIANTS, IDENTICAL SHAPE, SILENT IF WRONG.
  //   "b"       normalize by the robot's FULL orientation (incl. pitch/roll)
  //             -> motion_anchor_ori_b_mf_nonflat. What PRE-v1.1 models
  //             (e.g. cc392257) were trained on. Historic default.
  //   "heading" normalize by the robot's YAW ONLY, preserving the reference's
  //             pitch/roll relative to gravity
  //             -> motion_anchor_ori_heading_mf_nonflat. What every
  //             V1.1-LINEAGE model (v11/v12 and later) expects.
  //
  // Both emit 10x6 = 60 floats, so feeding the wrong one raises NOTHING. The
  // orientation error handed to the policy equals the robot's own tilt, 1:1:
  // 0 deg upright, 10 deg at 10 deg tilt, 45 deg at 45 deg. It is therefore
  // invisible during static/idle bring-up and appears only once the robot
  // leans -- exactly when a wrong reference is most costly.
  //
  // Select with X2_ANCHOR_ORI_MODE=heading (default "b" keeps every existing
  // deployment bit-identical). The model itself declares which it needs:
  //   python gear_sonic/scripts/onnx_obs_contract.py MODEL.onnx --require b
  // ACCEPTED VALUES -- and anything else is FATAL, deliberately.
  //
  //   "body" / "body_frame" / "b"   normalise by the robot's FULL orientation,
  //                                 pitch and roll included. "b" is the legacy
  //                                 spelling: it is just the letter in the
  //                                 training observation name
  //                                 motion_anchor_ori_B_mf_nonflat, i.e. BODY
  //                                 FRAME. Kept as an alias so existing
  //                                 profiles keep working.
  //   "heading"                     normalise by YAW ONLY.
  //
  // The old test was `e == "heading"`, so EVERY other string -- a typo, a
  // trailing space, "Heading" -- silently selected body frame, which is the
  // opposite mode. Feeding the wrong one raises nothing (both emit 10x6 = 60
  // floats) and the error grows 1:1 with the robot's tilt, so it survives
  // static bring-up and appears only once the robot leans. That is exactly the
  // 2026-08-24 hardware incident. An unrecognised value must therefore stop
  // the run, not guess.
  static const bool kHeadingMode = [] {
    const char* e = std::getenv("X2_ANCHOR_ORI_MODE");
    if (e == nullptr) return false;                 // unset -> body frame
    const std::string v(e);
    if (v == "heading") return true;
    if (v == "body" || v == "body_frame" || v == "b") return false;
    throw std::runtime_error(
        "X2_ANCHOR_ORI_MODE=\"" + v + "\" is not recognised. Use \"body\" "
        "(aliases: body_frame, b) or \"heading\". Refusing to guess: the two "
        "differ only once the robot tilts, so a wrong guess passes bring-up "
        "and fails under lean.");
  }();

  const auto cur_quat_xyzw = wxyz_to_xyzw(base_quat_wxyz);
  const auto norm_quat_xyzw = kHeadingMode
      ? yaw_only_quat_xyzw(cur_quat_xyzw)
      : cur_quat_xyzw;
  const auto cur_quat_inv_xyzw = quat_conj_xyzw(norm_quat_xyzw);

  // -------------------------------------------------------------------------
  // ONNX 680-D tokenizer layout = "IsaacLab buggy temporal-axis reshape":
  //
  //   For each row k in 0..9 (68 floats per row):
  //     k < 5:   [jpos_f(2k)(31) | jpos_f(2k+1)(31) | ori_fk(6)]
  //     k >= 5:  [jvel_f(2(k-5))(31) | jvel_f(2(k-5)+1)(31) | ori_fk(6)]
  //   flattened row-major to 680.
  //
  // Where frame k of the FUTURE WINDOW is sampled at
  //   t_k = current_time + k * DT_FUTURE_REF                  for k = 0..9
  // i.e. frame 0 IS THE CURRENT MOTION FRAME (t + 0.0), matching IsaacLab's
  // `TrackingCommand.future_time_steps_init = arange(N) * frame_skips`.
  //
  // Why this odd layout? It mirrors a documented bug in IsaacLab itself.
  // `command_multi_future` returns cat([joint_pos_multi_future,
  // joint_vel_multi_future], dim=1) with shape (N, 2 * num_future * num_dof).
  // The `_nonflat` form is built by
  // `gear_sonic/envs/manager_env/mdp/observations.py:584-585`:
  //
  //     return command.command_multi_future.reshape(
  //         command.num_envs, command.num_future_frames, -1)
  //
  // i.e. reshape((N, 620)) -> (N, 10, 62). But the underlying buffer is
  // [jpos_f0..f9 (310) | jvel_f0..f9 (310)], so this reshape DOES NOT produce
  // per-frame [jpos_fk, jvel_fk] rows. It produces:
  //   row 0: jpos_f0 + jpos_f1
  //   row 1: jpos_f2 + jpos_f3   ...   row 4: jpos_f8 + jpos_f9
  //   row 5: jvel_f0 + jvel_f1   ...   row 9: jvel_f8 + jvel_f9
  // The encoder then `cat([..._nonflat, motion_anchor_ori_b_mf_nonflat],
  // dim=-1)` (shape (N, 10, 68)) and flattens row-major to 680. The trained
  // policy LEARNS this scrambled mapping during PPO, so deploy MUST
  // reproduce it bit-for-bit -- a "correct" per-frame layout makes every
  // frame look like the wrong DOFs to the policy.
  //
  // gear_sonic/trl/losses/token_losses.py:72-86 explicitly documents this
  // ("there's a bug in command_multi_future_nonflat, where the temporal axis
  // is incorrectly flattened") and the reverse-mapping decoder code there
  // confirms the slicing convention used at training time.
  //
  // gear_sonic/scripts/eval_x2_mujoco.py:620-629 reproduces the same buggy
  // reshape on the Python eval side, which is why Python eval holds 30 s
  // cleanly on the iter-4000 checkpoint while the C++ deploy (previously
  // using "true" per-frame interleaved) was sending the policy a scrambled
  // observation and falling at ~5 s.
  //
  // -------------------------------------------------------------------------
  // Bug history #4 (THIS revision): switched from per-frame interleaved
  // [jpos_fk | jvel_fk | ori_fk] (revision #3) to IsaacLab's buggy
  // pair-of-jpos-then-pair-of-jvel layout above. Symptom: parity-profile
  // sim held ~4 s before slow-drift collapse; first-tick obs diff vs
  // Python eval showed identical numerical values at structurally
  // mismatched indices (slot 21 / row 5..9 in C++ vs slot 52 / row 0..4 in
  // Python -- same jpos data, different positions in the flat 680).
  //
  // Bug history #3: per-frame interleaved
  // [cmd_f0(62)|ori_f0(6)|cmd_f1(62)|ori_f1(6)|...]. Adopted based on
  // misreading the re-exported ONNX graph structure -- the graph's
  // Reshape(-1, 10, 68) merely matched IL's `view(-1, 10, 68)` shape, but
  // the input flat order coming into that view was the "buggy" one above,
  // not per-frame interleaved.
  //
  // Bug history #2: an earlier port from `eval_x2_mujoco.build_tokenizer_obs`
  // used `(f + 1) * DT_FUTURE_REF` (== [t+0.1, ..., t+1.0]). That window is
  // shifted by exactly one frame from training, so the policy was being
  // asked to track a 100ms-ahead reference instead of the current one and
  // produced large catch-up actions on every step.
  //
  // Bug history #1: a previous "fix" reshuffled this to
  // [jpos_f0..f9 | jvel_f0..f9 | ori_f0..f9] (full block layout, no ori
  // interleaving with rows). Close to IsaacLab's actual layout for the
  // command part, but missing the per-row ori concatenation, so the 60 ori
  // floats landed where the policy expected jpos/jvel of late frames.
  //
  // -------------------------------------------------------------------------
  // The reference returns joint pos/vel in MuJoCo order; we IL-remap here
  // (matches eval_x2_mujoco.py's `m["dof"][fi][IL_TO_MJ_DOF]` -- using the
  // IsaacLab->MuJoCo permutation INDEXED BY IL gives an IL-ordered output).
  // -------------------------------------------------------------------------

  // Pre-collect per-future-frame data so we can lay it out in the buggy
  // pair-then-pair pattern below.
  std::array<std::array<float, NUM_DOFS>, NUM_FUTURE_FRAMES> jpos_frames{};
  std::array<std::array<float, NUM_DOFS>, NUM_FUTURE_FRAMES> jvel_frames{};
  std::array<std::array<float, 6>,        NUM_FUTURE_FRAMES> ori_frames{};

  for (std::size_t k = 0; k < NUM_FUTURE_FRAMES; ++k) {
    const double t_k   = current_time + static_cast<double>(k) * DT_FUTURE_REF;
    const auto   frame = ref.Sample(t_k);

    for (std::size_t il = 0; il < NUM_DOFS; ++il) {
      const std::size_t mj = static_cast<std::size_t>(isaaclab_to_mujoco[il]);
      jpos_frames[k][il] = static_cast<float>(frame.joint_pos_mj[mj]);
      jvel_frames[k][il] = static_cast<float>(frame.joint_vel_mj[mj]);
    }

    // Bootstrap-safe reference quat. The caller passes
    // ``override_ref_quat_xyzw = base_quat_xyzw`` when the input source
    // has never received a real pose-ref frame, so the 10 ori rows
    // collapse to identity (rel = inv(cur) * cur) and the policy
    // stops trying to twist the body back to the C++ bootstrap's
    // ``root_quat_xyzw = (0, 0, 0, 1)`` (world +X) pre-fill that
    // would otherwise leak through here. See header docstring for
    // the full failure-mode story.
    const auto& ref_quat_xyzw =
        override_ref_quat_xyzw.has_value()
            ? *override_ref_quat_xyzw
            : frame.root_quat_xyzw;
    const auto rel_xyzw = quat_mul_xyzw(cur_quat_inv_xyzw, ref_quat_xyzw);
    const auto rot6     = rot6d_from_quat_xyzw(rel_xyzw);
    for (std::size_t i = 0; i < 6; ++i) {
      ori_frames[k][i] = static_cast<float>(rot6[i]);
    }
  }

  // Emit 10 rows of 68 floats, matching IsaacLab's buggy reshape.
  for (std::size_t k = 0; k < NUM_FUTURE_FRAMES; ++k) {
    const bool        is_jpos_row = (k < 5);
    const std::size_t pair_lo     = is_jpos_row ? (2 * k) : (2 * (k - 5));
    const std::size_t pair_hi     = pair_lo + 1;
    const auto& a = is_jpos_row ? jpos_frames[pair_lo] : jvel_frames[pair_lo];
    const auto& b = is_jpos_row ? jpos_frames[pair_hi] : jvel_frames[pair_hi];
    for (std::size_t i = 0; i < NUM_DOFS; ++i) out.push_back(a[i]);
    for (std::size_t i = 0; i < NUM_DOFS; ++i) out.push_back(b[i]);
    for (std::size_t i = 0; i < 6; ++i)        out.push_back(ori_frames[k][i]);
  }

  return out;
}

}  // namespace agi_x2
