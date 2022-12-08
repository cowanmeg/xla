#include "torch_xla/csrc/tensor_impl.h"

#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/lazy/backend/backend_interface.h"
#include "torch/csrc/lazy/core/tensor.h"
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir_builder.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

struct XLAGuardImpl : public c10::impl::DeviceGuardImplInterface {
  at::DeviceType type() const override { return at::DeviceType::XLA; }

  c10::Device exchangeDevice(c10::Device device) const override {
    return bridge::SetCurrentDevice(device);
  }

  c10::Device getDevice() const override {
    return bridge::GetCurrentAtenDevice();
  }

  void setDevice(c10::Device device) const override {
    bridge::SetCurrentDevice(device);
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    bridge::SetCurrentDevice(device);
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, bridge::GetCurrentAtenDevice());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return xla::ComputationClient::Get()->GetNumDevices();
  }
};

C10_REGISTER_GUARD_IMPL(XLA, XLAGuardImpl);

}  // namespace

XLATensorImpl::XLATensorImpl(XLATensorPtr tensor)
    : torch::lazy::LTCTensorImpl(
          tensor, c10::DispatchKey::XLA, c10::DispatchKey::AutogradXLA,
          GetTypeMeta(*tensor),
          bridge::XlaDeviceToAtenDevice(tensor->GetDevice())) {
  tensor_ = c10::make_intrusive<XLATensor>(std::move(*tensor));
  is_non_overlapping_and_dense_ = false;
  set_custom_sizes_strides(SizesStridesPolicy::CustomSizes);
}

void XLATensorImpl::set_tensor(XLATensorPtr xla_tensor) {
  tensor_ = c10::make_intrusive<XLATensor>(std::move(*xla_tensor));
  generation_ = 0;
}

at::IntArrayRef XLATensorImpl::sizes_custom() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return sizes_default();
}

c10::SymIntArrayRef XLATensorImpl::sym_sizes_custom() const {
  // N.B. SetupSizeProperties also updates sym_sizes_
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return c10::SymIntArrayRef(sym_sizes_.data(), sym_sizes_.size());
}

c10::SymInt XLATensorImpl::sym_numel_custom() const {
  auto sym_sizes = sym_sizes_custom();
  c10::SymInt prod{1};
  for (auto s : sym_sizes) {
    prod *= s;
  }
  return prod;
}

at::IntArrayRef XLATensorImpl::strides_custom() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return strides_default();
}

int64_t XLATensorImpl::dim_custom() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return dim_default();
}

int64_t XLATensorImpl::numel_custom() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return numel_default();
}

void XLATensorImpl::SetupSizeProperties() {
  size_t generation = tensor_->generation();
  if (generation != generation_) {
    // Fill up the basic dimension data members which the base class
    // implementation uses in its APIs.
    auto shape = tensor_->shape();
    c10::SmallVector<int64_t, 5> updated_sizes;
    numel_ = 1;
    for (auto dim : shape.get().dimensions()) {
      updated_sizes.push_back(dim);
      numel_ *= dim;
    }
    sizes_and_strides_.set_sizes(updated_sizes);
    auto updated_strides = torch::lazy::ComputeArrayStrides(
        torch::lazy::ToVector<int64_t>(shape.get().dimensions()));
    for (int i = 0; i < updated_strides.size(); i++) {
      sizes_and_strides_.stride_at_unchecked(i) = updated_strides[i];
    }
    SetupSymSizeProperties();
    generation_ = generation;
  }
}

void XLATensorImpl::SetupSymSizeProperties() {
  auto shape = tensor_->shape();
  auto rank = shape.get().rank();
  std::vector<c10::SymInt> sym_sizes;
  sym_sizes.reserve(rank);

  XLAIrBuilder a = XLAIrBuilder();
  for (auto i : c10::irange(rank)) {
    if (shape.get().is_dynamic_dimension(i)) {
      auto dim_node = a.MakeSizeNode(tensor_->GetIrValue(), i);
      auto symint_node = c10::make_intrusive<XLASymNodeImpl>(dim_node);
      sym_sizes.push_back(c10::SymInt(
          static_cast<c10::intrusive_ptr<c10::SymNodeImpl>>(symint_node)));
    } else {
      sym_sizes.push_back(c10::SymInt(shape.get().dimensions(i)));
    }
  }
  sym_sizes_ = sym_sizes;
}

caffe2::TypeMeta XLATensorImpl::GetTypeMeta(const XLATensor& tensor) {
  return c10::scalarTypeToTypeMeta(tensor.dtype());
}

void XLATensorImpl::AtenInitialize() {
  // ATEN specific initialization calls placed below.
}

}  // namespace torch_xla
