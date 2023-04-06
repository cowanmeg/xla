#include "torch_xla/csrc/ops/repeat_interleave.h"

#include "absl/strings/str_join.h"
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerRepeatInterleave(xla::XlaOp input, xla::XlaOp repeat,
                            c10::optional<int64_t> dim,
                            c10::optional<int64_t> output_size) {
  int64_t dim_val;
  if (dim.has_value()) {
    dim_val = dim.value();
  } else {
    dim_val = 0;
    xla::Shape shape;
    input = XlaHelpers::Flatten(input, &shape);
  }
  return BuildRepeatInterleave(input, repeat, dim_val);
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& repeat,
                           c10::optional<int64_t> dim,
                           c10::optional<int64_t> output_size) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerRepeatInterleave(operands[0], operands[1], dim, output_size);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

RepeatInterleave::RepeatInterleave(const torch::lazy::Value& input, 
                                   const torch::lazy::Value& repeats,
                                   c10::optional<int64_t> dim, 
                                   c10::optional<int64_t> output_size)
    : XlaNode(torch::lazy::OpKind(at::aten::repeat_interleave), {input, repeats},
              [&]() {
                return NodeOutputShape(input, repeats, dim, output_size);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(torch::lazy::OptionalOr<int64_t>(dim, 0),
                                 torch::lazy::OptionalOr<int64_t>(output_size, -1))),
      dim_(dim),
      output_size_(output_size) {}

torch::lazy::NodePtr RepeatInterleave::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<RepeatInterleave>(operands.at(0), operands.at(1),
                                                 dim_, output_size_);
}

XlaOpVector RepeatInterleave::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp repeats = loctx->GetOutputOp(operand(1));
  return ReturnOp(
      LowerRepeatInterleave(input, repeats, dim_, output_size_), loctx);
}

std::string RepeatInterleave::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString();
  if (dim_.has_value()) {
    ss << ", dim=" << dim_.value();
  } else {
    ss << ", dim=null";
  }
  if (output_size_.has_value()) {
    ss << ", output_size=" << output_size_.value();
  } else {
    ss << ", output_size=null";
  }
  return ss.str();
}

}  // namespace torch_xla
