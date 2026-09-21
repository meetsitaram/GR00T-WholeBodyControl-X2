/**
 * @file onnx_actor.hpp
 * @brief Thin C++ wrapper around the fused X2 g1+g1_dyn ONNX session.
 *
 * Mirrors gear_sonic/scripts/eval_x2_mujoco_onnx.py::OnnxActor:
 *   - Single ONNX input  : float32 [1, 1670] = [tokenizer(680) | proprio(990)]
 *   - Single ONNX output : float32 [1, 31]   = action in IsaacLab DOF order
 *
 * The ONNX consumes the 680-D tokenizer slice in IsaacLab's "buggy
 * temporal-axis reshape" layout (see tokenizer_obs.hpp for the exact
 * pair-of-jpos-then-pair-of-jvel pattern). `BuildTokenizerObs` produces
 * that layout directly so this wrapper does no rearrangement.
 *
 * The session is created once at startup (slow), and `Infer()` may be called
 * from the realtime control timer (fast: ~0.5 ms on x86_64 CPU).
 *
 * Phase-0 acceptance: fused ONNX vs .pt actions agree to ~2.4e-7 over 30 s
 * rollout; threshold is 1e-4.
 */

#ifndef AGI_X2_ONNX_ACTOR_HPP
#define AGI_X2_ONNX_ACTOR_HPP

#include "policy_parameters.hpp"
#include "proprioception_buffer.hpp"
#include "tokenizer_obs.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

// Forward-declare ONNX Runtime types so callers don't need to pull the heavy
// ORT C++ API into headers that include this one. The full include lives in
// onnx_actor.cpp.
namespace Ort {
class Env;
class Session;
class AllocatorWithDefaultOptions;
}  // namespace Ort

namespace agi_x2 {

class OnnxActor {
 public:
  /// Loads the ONNX model from disk and validates its input/output shapes
  /// against the expected fused-g1 contract. Throws std::runtime_error on
  /// any mismatch (wrong width, multi-input, etc.).
  explicit OnnxActor(const std::string& model_path,
                     int intra_op_threads = 1);

  ~OnnxActor();

  // Non-copyable, non-movable: holds a unique ORT session.
  OnnxActor(const OnnxActor&)            = delete;
  OnnxActor& operator=(const OnnxActor&) = delete;

  /**
   * Run one inference tick.
   *
   * @param tokenizer_obs  680-D tokenizer obs in IsaacLab buggy-reshape
   *                       layout (see BuildTokenizerObs)
   * @param proprioception 990-D proprioception in PolicyCfg term order
   * @return 31-D action in IsaacLab DOF order. Caller is responsible for
   *         applying x2_action_scale and the il->mj remap before sending
   *         to the joint controller.
   */
  std::array<double, NUM_DOFS> Infer(const std::vector<float>& tokenizer_obs,
                                     const std::vector<float>& proprioception);

  /**
   * Run one inference tick on an already-assembled observation vector of
   * exactly expected_obs_dim() floats. Used by the native DUAL-HEAD graph
   * (2026-09-04, native_dual_head_onnx_export.py): obs =
   * [g1 pose-ref features 680 | smpl obs 840 | proprio 990 | head 1].
   */
  std::array<double, NUM_DOFS> InferObs(const std::vector<float>& obs);

  /**
   * Read a custom-metadata value stamped into the ONNX at export time
   * (e.g. ``graph_kind``: "any2any_token_deploy" for the whole-body-teleop
   * token graph). Returns "" when the key is absent — a graph exported
   * before metadata stamping is indistinguishable from a pose-encoder
   * graph, which is exactly the safe default for the mode guard.
   */
  std::string GetCustomMetadata(const std::string& key) const;

  const std::string& model_path()       const { return model_path_; }
  const std::string& input_name()       const { return input_name_; }
  const std::string& output_name()      const { return output_name_; }
  std::int64_t       expected_obs_dim() const { return expected_obs_dim_; }

 private:
  std::string model_path_;
  std::int64_t expected_obs_dim_ = 0;

  // Held by unique_ptr to keep ORT types out of the header.
  std::unique_ptr<Ort::Env>                       env_;
  std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
  std::unique_ptr<Ort::Session>                   session_;

  // Strings owned here; pointers are passed to ORT's C interface at Run().
  std::string              input_name_;
  std::string              output_name_;
  std::vector<const char*> input_name_ptrs_;
  std::vector<const char*> output_name_ptrs_;

  // Reusable IO buffers to avoid heap churn at 50 Hz.
  std::vector<float> input_buffer_;   ///< concatenated tok || prop, flat
  std::vector<float> output_buffer_;  ///< 31 floats
};

}  // namespace agi_x2

#endif  // AGI_X2_ONNX_ACTOR_HPP
