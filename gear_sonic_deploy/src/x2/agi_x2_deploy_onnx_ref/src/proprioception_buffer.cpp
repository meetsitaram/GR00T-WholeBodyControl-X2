#include "proprioception_buffer.hpp"

#include <cstring>

namespace agi_x2 {

ProprioceptionBuffer::ProprioceptionBuffer() { Reset(); }

void ProprioceptionBuffer::Reset()
{
  // Zero-fill rings; primed_=false is the real "empty" signal -- the zeros
  // are never read out before being overwritten because GetFlat() is only
  // meaningful after the first Append().
  for (auto& f : ang_vel_ring_)  f.fill(0.0);
  for (auto& f : jpos_rel_ring_) f.fill(0.0);
  for (auto& f : jvel_ring_)     f.fill(0.0);
  for (auto& f : action_ring_)   f.fill(0.0);
  for (auto& f : grav_ring_)     f.fill(0.0);
  write_idx_ = 0;
  primed_    = false;
}

void ProprioceptionBuffer::Append(
    const std::array<double, 3>& base_ang_vel,
    const std::array<double, NUM_DOFS>& joint_pos_rel_il,
    const std::array<double, NUM_DOFS>& joint_vel_il,
    const std::array<double, NUM_DOFS>& last_action_il,
    const std::array<double, 3>& gravity_body)
{
  if (!primed_) {
    // Broadcast-fill every slot with the first sample (IsaacLab semantics).
    for (std::size_t i = 0; i < HISTORY_LEN; ++i) {
      ang_vel_ring_[i]  = base_ang_vel;
      jpos_rel_ring_[i] = joint_pos_rel_il;
      jvel_ring_[i]     = joint_vel_il;
      action_ring_[i]   = last_action_il;
      grav_ring_[i]     = gravity_body;
    }
    write_idx_ = 0;  // newest written sample is at slot HISTORY_LEN-1
    primed_    = true;
    return;
  }

  // Normal append: overwrite the oldest slot. write_idx_ points at the slot
  // that will be overwritten next (i.e. the current oldest), so writing
  // there and bumping the cursor keeps the "oldest at write_idx_" invariant
  // GetFlat() relies on.
  ang_vel_ring_[write_idx_]  = base_ang_vel;
  jpos_rel_ring_[write_idx_] = joint_pos_rel_il;
  jvel_ring_[write_idx_]     = joint_vel_il;
  action_ring_[write_idx_]   = last_action_il;
  grav_ring_[write_idx_]     = gravity_body;
  write_idx_ = (write_idx_ + 1) % HISTORY_LEN;
}

std::vector<float> ProprioceptionBuffer::GetFlat() const
{
  std::vector<float> out;
  out.reserve(PROP_DIM);

  // After Append(), the oldest sample lives at write_idx_ and the newest at
  // (write_idx_ + HISTORY_LEN - 1) % HISTORY_LEN. Walk forwards from
  // write_idx_ for HISTORY_LEN steps -> oldest-first iteration.
  auto emit_term_3 = [&](const std::array<std::array<double, 3>, HISTORY_LEN>& ring) {
    for (std::size_t k = 0; k < HISTORY_LEN; ++k) {
      const auto& f = ring[(write_idx_ + k) % HISTORY_LEN];
      out.push_back(static_cast<float>(f[0]));
      out.push_back(static_cast<float>(f[1]));
      out.push_back(static_cast<float>(f[2]));
    }
  };
  auto emit_term_n = [&](const std::array<std::array<double, NUM_DOFS>, HISTORY_LEN>& ring) {
    for (std::size_t k = 0; k < HISTORY_LEN; ++k) {
      const auto& f = ring[(write_idx_ + k) % HISTORY_LEN];
      for (std::size_t d = 0; d < NUM_DOFS; ++d) {
        out.push_back(static_cast<float>(f[d]));
      }
    }
  };

  // Order MUST match the PolicyCfg dataclass attribute order (see
  // observations.py lines 107-128 in the training tree). Don't sort.
  emit_term_3(ang_vel_ring_);    // base_ang_vel
  emit_term_n(jpos_rel_ring_);   // joint_pos_rel
  emit_term_n(jvel_ring_);       // joint_vel
  emit_term_n(action_ring_);     // last_action
  emit_term_3(grav_ring_);       // gravity_dir

  return out;
}

}  // namespace agi_x2
