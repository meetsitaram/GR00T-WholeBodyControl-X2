#include "onnx_actor.hpp"

#include <onnxruntime_cxx_api.h>

#include <stdexcept>
#include <string>

namespace agi_x2 {

namespace {
// Expected input width for the X2 g1+g1_dyn fused ONNX:
//   tokenizer(680) || proprioception(990) = 1670
constexpr std::int64_t kExpectedObsDim = static_cast<std::int64_t>(TOK_DIM + PROP_DIM);
constexpr std::int64_t kExpectedActDim = static_cast<std::int64_t>(NUM_DOFS);  // 31
// Native dual-head graph (2026-09-04): [g1 680 | smpl 840 | proprio 990 | head 1]
constexpr std::int64_t kDualHeadObsDim = 680 + 840 + PROP_DIM + 1;             // 2511
}  // namespace

OnnxActor::OnnxActor(const std::string& model_path, int intra_op_threads)
    : model_path_(model_path),
      expected_obs_dim_(kExpectedObsDim)
{
  env_       = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                           "agi_x2_deploy_onnx_ref");
  allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();

  Ort::SessionOptions session_opts;
  session_opts.SetIntraOpNumThreads(intra_op_threads);
  session_opts.SetInterOpNumThreads(1);
  session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  try {
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_opts);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error("OnnxActor: failed to load " + model_path +
                             " (" + e.what() + ")");
  }

  if (session_->GetInputCount() != 1) {
    throw std::runtime_error("OnnxActor: expected 1 input on fused X2 ONNX, got " +
                             std::to_string(session_->GetInputCount()) +
                             ". This loader only supports the fused " +
                             "encoder+decoder graph (model_step_*_g1.onnx).");
  }
  if (session_->GetOutputCount() != 1) {
    throw std::runtime_error("OnnxActor: expected 1 output, got " +
                             std::to_string(session_->GetOutputCount()));
  }

  // Cache input/output names + pointer aliases.
  {
    auto in  = session_->GetInputNameAllocated(0, *allocator_);
    auto out = session_->GetOutputNameAllocated(0, *allocator_);
    input_name_  = std::string(in.get());
    output_name_ = std::string(out.get());
  }
  input_name_ptrs_  = { input_name_.c_str() };
  output_name_ptrs_ = { output_name_.c_str() };

  // Validate input shape: [1, 1670] (or [-1, 1670] for dynamic batch).
  // NB: GetTensorTypeAndShapeInfo() returns a non-owning ConstTensorType view
  // into the parent TypeInfo. We MUST bind the TypeInfo to a local first;
  // chaining `GetInputTypeInfo(0).GetTensorTypeAndShapeInfo()` lets the
  // temporary TypeInfo (and its underlying OrtTypeInfo*) be released at
  // end-of-statement, and the subsequent GetShape() call reads through a
  // dangling pointer (manifests as `std::length_error: cannot create
  // std::vector larger than max_size()` because the corrupted dim count is
  // interpreted as a huge size_t).
  {
    const Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
    const auto info  = type_info.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (shape.size() != 2 ||
        (shape[1] != kExpectedObsDim && shape[1] != kDualHeadObsDim &&
         shape[1] != -1)) {
      throw std::runtime_error("OnnxActor: input shape mismatch. Expected "
                               "[*, " + std::to_string(kExpectedObsDim) +
                               "] (fused g1) or [*, " +
                               std::to_string(kDualHeadObsDim) +
                               "] (native dual-head), got [*, " +
                               std::to_string(shape[1]) + "].");
    }
    if (shape[1] == kDualHeadObsDim) expected_obs_dim_ = kDualHeadObsDim;
  }
  // Validate output shape: [1, 31].
  {
    const Ort::TypeInfo type_info = session_->GetOutputTypeInfo(0);
    const auto info  = type_info.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (shape.size() != 2 || (shape[1] != kExpectedActDim && shape[1] != -1)) {
      throw std::runtime_error("OnnxActor: output shape mismatch. Expected "
                               "[*, " + std::to_string(kExpectedActDim) +
                               "], policy may be from a different robot.");
    }
  }

  input_buffer_.assign(static_cast<std::size_t>(expected_obs_dim_), 0.0f);
  output_buffer_.assign(static_cast<std::size_t>(kExpectedActDim), 0.0f);
}

OnnxActor::~OnnxActor() = default;

std::array<double, NUM_DOFS> OnnxActor::InferObs(const std::vector<float>& obs)
{
  if (static_cast<std::int64_t>(obs.size()) != expected_obs_dim_) {
    throw std::runtime_error("OnnxActor::InferObs: obs size " +
                             std::to_string(obs.size()) + " != expected " +
                             std::to_string(expected_obs_dim_));
  }
  std::copy(obs.begin(), obs.end(), input_buffer_.begin());
  const std::array<std::int64_t, 2> input_shape  = {1, expected_obs_dim_};
  static const std::array<std::int64_t, 2> output_shape = {1, kExpectedActDim};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                        OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      mem_info, input_buffer_.data(), input_buffer_.size(),
      input_shape.data(), input_shape.size()));
  std::vector<Ort::Value> outputs;
  outputs.emplace_back(Ort::Value::CreateTensor<float>(
      mem_info, output_buffer_.data(), output_buffer_.size(),
      output_shape.data(), output_shape.size()));
  session_->Run(Ort::RunOptions{nullptr},
                input_name_ptrs_.data(), inputs.data(), inputs.size(),
                output_name_ptrs_.data(), outputs.data(), outputs.size());
  std::array<double, NUM_DOFS> action_il{};
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    action_il[i] = static_cast<double>(output_buffer_[i]);
  }
  return action_il;
}

std::array<double, NUM_DOFS> OnnxActor::Infer(
    const std::vector<float>& tokenizer_obs,
    const std::vector<float>& proprioception)
{
  if (expected_obs_dim_ != kExpectedObsDim) {
    throw std::runtime_error("OnnxActor::Infer: this session is a dual-head "
                             "graph; use InferObs().");
  }
  if (tokenizer_obs.size() != TOK_DIM) {
    throw std::runtime_error("OnnxActor::Infer: tokenizer obs size " +
                             std::to_string(tokenizer_obs.size()) +
                             " != expected " + std::to_string(TOK_DIM));
  }
  if (proprioception.size() != PROP_DIM) {
    throw std::runtime_error("OnnxActor::Infer: proprioception size " +
                             std::to_string(proprioception.size()) +
                             " != expected " + std::to_string(PROP_DIM));
  }

  // Concat into the persistent input buffer, no allocations.
  std::copy(tokenizer_obs.begin(), tokenizer_obs.end(), input_buffer_.begin());
  std::copy(proprioception.begin(), proprioception.end(),
            input_buffer_.begin() + static_cast<std::ptrdiff_t>(TOK_DIM));

  static const std::array<std::int64_t, 2> input_shape  = {1, kExpectedObsDim};
  static const std::array<std::int64_t, 2> output_shape = {1, kExpectedActDim};

  // Construct an Ort::Value view over our buffer; ORT does not own the data.
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                        OrtMemTypeDefault);
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      mem_info, input_buffer_.data(), input_buffer_.size(),
      input_shape.data(), input_shape.size()));

  std::vector<Ort::Value> outputs;
  outputs.emplace_back(Ort::Value::CreateTensor<float>(
      mem_info, output_buffer_.data(), output_buffer_.size(),
      output_shape.data(), output_shape.size()));

  session_->Run(Ort::RunOptions{nullptr},
                input_name_ptrs_.data(), inputs.data(), inputs.size(),
                output_name_ptrs_.data(), outputs.data(), outputs.size());

  std::array<double, NUM_DOFS> action_il{};
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    action_il[i] = static_cast<double>(output_buffer_[i]);
  }
  return action_il;
}

std::string OnnxActor::GetCustomMetadata(const std::string& key) const {
  Ort::AllocatorWithDefaultOptions alloc;
  const Ort::ModelMetadata meta = session_->GetModelMetadata();
  Ort::AllocatedStringPtr v =
      meta.LookupCustomMetadataMapAllocated(key.c_str(), alloc);
  return v ? std::string(v.get()) : std::string();
}

}  // namespace agi_x2
