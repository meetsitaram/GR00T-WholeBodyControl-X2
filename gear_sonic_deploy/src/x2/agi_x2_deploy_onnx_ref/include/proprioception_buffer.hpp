/**
 * @file proprioception_buffer.hpp
 * @brief 990-D ring-buffered proprioception, IsaacLab CircularBuffer semantics.
 *
 * Exact port of gear_sonic/scripts/eval_x2_mujoco.py::ProprioceptionBuffer.
 * The dimensionality and term order are spec'd in
 *   gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/config/obs_config_x2_ultra.yaml
 * and verified end-to-end by /tmp/x2_step0_isaaclab.pt during Phase 0.
 *
 * Layout produced by GetFlat():
 *   base_ang_vel  (3 ) x 10 = 30
 *   joint_pos_rel (31) x 10 = 310     (joint_pos_il - DEFAULT_DOF[il_idx])
 *   joint_vel     (31) x 10 = 310     (il-ordered)
 *   last_action   (31) x 10 = 310     (il-ordered, the previous policy output)
 *   gravity_dir   (3 ) x 10 = 30
 *   ----------------------------------
 *   TOTAL                     990
 *
 * Within each term frames are oldest-first (frame 0 = oldest, frame 9 = newest).
 *
 * Reset / first-sample priming:
 *   On reset, the buffer is empty. The next Append() broadcast-fills all 10
 *   slots of every term with the first observation. Without this, frames
 *   0..8 would be zero on the first inference step, which is OOD for any
 *   policy trained with history.
 */

#ifndef AGI_X2_PROPRIOCEPTION_BUFFER_HPP
#define AGI_X2_PROPRIOCEPTION_BUFFER_HPP

#include "policy_parameters.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace agi_x2 {

constexpr std::size_t PROP_DIM = 990;

/// Compile-time check that the per-frame layout adds up to 78 (= 3+31+31+31+3)
/// and that 10 frames * 78 = 780... wait that's wrong. Let me redo: actually
/// it's 30+310+310+310+30 = 990. OK static_assert that.
static_assert(NUM_DOFS == 31,
              "ProprioceptionBuffer assumes 31-DOF X2; regenerate header "
              "via codegen_x2_policy_parameters.py if NUM_DOFS changed.");
static_assert(HISTORY_LEN == 10,
              "ProprioceptionBuffer assumes HISTORY_LEN=10; matches IsaacLab "
              "CircularBuffer depth used at training time.");
static_assert(HISTORY_LEN * (3 + NUM_DOFS + NUM_DOFS + NUM_DOFS + 3) == PROP_DIM,
              "Proprioception term widths do not sum to 990 -- did term order "
              "or HISTORY_LEN change?");

class ProprioceptionBuffer {
 public:
  ProprioceptionBuffer();

  /// Drop all history. The next Append() will broadcast-fill (priming).
  void Reset();

  /**
   * Append one observation tick.
   *
   * @param base_ang_vel       (3,) IMU torso angular velocity, body frame
   * @param joint_pos_rel_il   (31,) joint_pos - DEFAULT_DOF, IL order
   * @param joint_vel_il       (31,) joint velocity, IL order
   * @param last_action_il     (31,) previous policy action, IL order
   * @param gravity_body       (3,) gravity in body frame
   *                           (== quat_rotate_inverse_wxyz(base_quat, {0,0,-1}))
   *
   * If the buffer is currently empty, this single call broadcast-fills all
   * HISTORY_LEN slots with the supplied values (matches IsaacLab
   * CircularBuffer priming on reset).
   */
  void Append(const std::array<double, 3>& base_ang_vel,
              const std::array<double, NUM_DOFS>& joint_pos_rel_il,
              const std::array<double, NUM_DOFS>& joint_vel_il,
              const std::array<double, NUM_DOFS>& last_action_il,
              const std::array<double, 3>& gravity_body);

  /// Flatten the entire buffer into the 990-D float32 layout the ONNX
  /// encoder expects. Term order: ang_vel | jpos_rel | jvel | action | grav,
  /// each oldest-first across the 10 frames.
  std::vector<float> GetFlat() const;

  bool IsPrimed() const { return primed_; }

 private:
  // Each ring stores (HISTORY_LEN, dim) doubles, written in chronological
  // order. We keep an integer write_idx_ that advances mod HISTORY_LEN; on
  // GetFlat() we read out starting at write_idx_ so the result is always
  // oldest-first regardless of where the cursor is.
  std::array<std::array<double, 3>, HISTORY_LEN>          ang_vel_ring_;
  std::array<std::array<double, NUM_DOFS>, HISTORY_LEN>   jpos_rel_ring_;
  std::array<std::array<double, NUM_DOFS>, HISTORY_LEN>   jvel_ring_;
  std::array<std::array<double, NUM_DOFS>, HISTORY_LEN>   action_ring_;
  std::array<std::array<double, 3>, HISTORY_LEN>          grav_ring_;

  std::size_t write_idx_ = 0;  ///< next slot to overwrite
  bool        primed_    = false;
};

}  // namespace agi_x2

#endif  // AGI_X2_PROPRIOCEPTION_BUFFER_HPP
