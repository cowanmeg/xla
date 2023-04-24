#pragma once

#include <c10/util/Optional.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class RepeatInterleave : public XlaNode {
 public:
  RepeatInterleave(const torch::lazy::Value& input, int64_t repeats,
                   c10::optional<int64_t> dim,
                   c10::optional<int64_t> output_size);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t repeats_;
  c10::optional<int64_t> dim_;
  c10::optional<int64_t> output_size_;
};

}  // namespace torch_xla
