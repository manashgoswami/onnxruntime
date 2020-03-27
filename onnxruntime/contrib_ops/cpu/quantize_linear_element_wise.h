// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {
namespace quantize {

template <typename T>
class QLinearElementWiseBase {
 public:
  QLinearElementWiseBase() = default;
  onnxruntime::common::Status CheckInputs(OpKernelContext* context);

  T zeroA, zeroB, zeroC;
  float
};

template <typename T>
class QLinearAdd final : public OpKernel {
 public:
  QLinearAdd(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

template <typename T>
class QLinearMul final : public OpKernel {
 public:
  QLinearMul(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace quantize
}  // namespace contrib
}  // namespace onnxruntime
