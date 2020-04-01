// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear_element_wise.h"

#include "core/providers/cpu/math/element_wise_ops.h"
//#include <unsupported/Eigen/SpecialFunctions>
//#include "core/util/math.h"
//#include <cmath>

namespace onnxruntime {
namespace contrib {

// Broadcast loop for when using gsl::span<T>, functions are in this form:
// Input0Scalar: [](gsl::span<TOutput> output, TInput0 input0, gsl::span<const TInput1> input1, a_scale, b_scale, c_scale, a_zero, b_zero, c_zero)
// Input1Scalar: [](gsl::span<TOutput> output, gsl::span<const TInput0> input0, TInput1 input1, a_scale, b_scale, c_scale, a_zero, b_zero, c_zero)
// General     : [](gsl::span<TOutput> output, gsl::span<const TInput0> input0, gsl::span<const TInput1> input1, a_scale, b_scale, c_scale, a_zero, b_zero, c_zero)
// Scalar parameters can also be of type const TX&.
template <typename T, typename TBroadcaster, typename Output, typename Input0Scalar, typename Input1Scalar, typename General>
void QLinearBroadcastLoop(TBroadcaster& bc, Output& output, Input0Scalar input0scalar, Input1Scalar input1scalar, General general,
                          float a_scale, float b_scale, float c_scale, T a_zero, T b_zero, T c_zero) {
  if (bc.IsInput0Scalar()) {
    while (output)
      input0scalar(output.NextEigenOutput(), bc.NextScalar0(), bc.NextEigen1(), a_scale, b_scale, c_scale, a_zero, b_zero, c_zero);
  } else if (bc.IsInput1Scalar()) {
    while (output)
      input1scalar(output.NextEigenOutput(), bc.NextEigen0(), bc.NextScalar1(), a_scale, b_scale, c_scale, a_zero, b_zero, c_zero);
  } else {
    while (output)
      general(output.NextEigenOutput(), bc.NextEigen0(), bc.NextEigen1(), a_scale, b_scale, c_scale, a_zero, b_zero, c_zero);
  }
}

template <typename T, typename Input0Scalar, typename Input1Scalar, typename General>
Status QLinearBroadcastTwo(OpKernelContext& context, Input0Scalar input0scalar, Input1Scalar input1scalar, General general) {
  const float a_scale = *(context.Input<Tensor>(1)->Data<float>());
  const T a_zero = (nullptr == context.Input<Tensor>(2)) ? static_cast<T>(0) : *(context.Input<Tensor>(2)->template Data<T>());
  const float b_scale = *(context.Input<Tensor>(4)->Data<float>());
  const T b_zero = (nullptr == context.Input<Tensor>(5)) ? static_cast<T>(0) : *(context.Input<Tensor>(5)->template Data<T>());
  const float c_scale = *(context.Input<Tensor>(6)->Data<float>());
  const T c_zero = (nullptr == context.Input<Tensor>(7)) ? static_cast<T>(0) : *(context.Input<Tensor>(7)->template Data<T>());

  TBroadcaster<T, T> bc(*context.Input<Tensor>(0), *context.Input<Tensor>(3));
  TBroadcastOutput<T> output(bc.GetSpanSize(), *context.Output(0, bc.GetOutputShape()));
  QLinearBroadcastLoop(bc, output, input0scalar, input1scalar, general, a_scale, b_scale, c_scale, a_zero, b_zero, c_zero);
  return Status::OK();
}

template <typename T>
Status QLinearAdd<T>::Compute(OpKernelContext* context) const {
  return QLinearBroadcastTwo<T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1, float a_scale, float b_scale, float c_scale, T a_zero, T b_zero, T c_zero) {
        float a_value = a_scale * (input0 - a_zero);
        output = ((((input1.array() - b_zero).template cast<float>() * b_scale) + a_value) / c_scale - c_zero).template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1, float a_scale, float b_scale, float c_scale, T a_zero, T b_zero, T c_zero) {
        float b_value = b_scale * (input1 - b_zero);
        output = ((((input0.array() - a_zero).template cast<float>() * a_scale) + b_value) / c_scale - c_zero).template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1, float a_scale, float b_scale, float c_scale, T a_zero, T b_zero, T c_zero) {
        output = ((((input0.array() - a_zero).template cast<float>() * a_scale) + ((input1.array() - b_zero).template cast<float>() * b_scale)) / c_scale - c_zero).template cast<T>();
      });
}

template <typename T>
Status QLinearMul<T>::Compute(OpKernelContext* context) const {
  return QLinearBroadcastTwo<T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1, float a_scale, float b_scale, float c_scale, T a_zero, T b_zero, T c_zero) {
        float a_value_scaled_b_c = a_scale * static_cast<float>(input0 - a_zero) * b_scale / c_scale;
        output = ((input1.array() - b_zero).template cast<float>() * a_value_scaled_b_c - c_zero).template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1, float a_scale, float b_scale, float c_scale, T a_zero, T b_zero, T c_zero) {
        float b_value_scaled_a_c = b_scale * static_cast<float>(input1 - b_zero) * a_scale / c_scale;
        output = ((input0.array() - a_zero).template cast<float>() * b_value_scaled_a_c - c_zero).template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1, float a_scale, float b_scale, float c_scale, T a_zero, T b_zero, T c_zero) {
        output = (((input0.array() - a_zero).template cast<float>() * a_scale) * ((input1.array() - b_zero).template cast<float>() * b_scale) / c_scale - c_zero).template cast<T>();
      });
}

template class QLinearAdd<int8_t>;
template class QLinearAdd<uint8_t>;


#define REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(op_name, version, data_type, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL( \
      op_name, version,  data_type,  \
      KernelDefBuilder() \
        .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      KERNEL_CLASS<data_type>);

REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, int8_t, QLinearAdd);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, uint8_t, QLinearAdd);
//REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearMul, 1, int8_t, QLinearMul)
//REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearMul, 1, uint8_t, QLinearMul)


}  // namespace contrib
}  // namespace onnxruntime
