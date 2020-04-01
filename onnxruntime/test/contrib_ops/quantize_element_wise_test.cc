// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace test {

static int64_t calc_strides(const std::vector<int64_t>& dims, std::vector<int64_t>& strides) 
{
  strides.clear();
  strides.resize(dims.size(),  1);
  for (int i = (int)dims.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides[0] * dims[0];
}


// correct shape must be provided: 
//   c_shape is the shape after broadcast. 
//   a_shape and b_shape should be same length, prefix with 1 if needed.
template <typename T, typename CalcFunction> void
RunQLinearMathTestFromFloat(const char* op_name, CalcFunction calc,
    const std::vector<float>& a, const std::vector<int64_t>& a_shape, float a_scale, T a_zero,
    const std::vector<float>& b, const std::vector<int64_t>& b_shape, float b_scale, T b_zero,
    const std::vector<int64_t>& c_shape,  float c_scale, T c_zero)
{
  OpTester test(op_name, 1, onnxruntime::kMSDomain);
  std::vector<int64_t> a_strides, b_strides, c_strides;

  auto c_size = calc_strides(c_shape, c_strides);
  calc_strides(a_shape, a_strides);
  calc_strides(b_shape, b_strides);

  const float qmax = std::numeric_limits<T>::max();
  const float qmin_default = std::numeric_limits<T>::min();
  // adjust qmin for int8 inputs. This is required to keep zero point as zero
  const float qmin = qmin_default == -128 ? -127 : qmin_default;

  int a_zero_int = static_cast<int>(a_zero);
  int b_zero_int = static_cast<int>(b_zero);
  int c_zero_int = static_cast<int>(c_zero);

  std::vector<T> a_quantized(a.size());
  for (size_t i = 0, sz = a.size(); i < sz; ++i) {
    a_quantized[i] = static_cast<T>(clamp(a[i] / a_scale + a_zero_int, qmin, qmax));
  }
  test.template AddInput<T>("A", a_shape, a_quantized);
  test.AddInput<float>("A_scale", {},  {a_scale});
  test.template AddInput<T>("A_zero_point", {}, {a_zero});

  std::vector<T> b_quantized(b.size());
  for (size_t i = 0, sz = b.size(); i < sz; ++i) {
    b_quantized[i] = static_cast<T>(clamp(b[i] / b_scale + b_zero_int, qmin, qmax));
  }
  test.template AddInput<T>("B", b_shape, b_quantized);
  test.AddInput<float>("B_scale", {}, {b_scale});
  test.template AddInput<T>("B_zero_point", {}, {b_zero});

  std::vector<T> c(c_size);
  for (int64_t offset = 0; offset < c_size; ++offset) {
    int64_t a_remain = offset, b_remain = offset;
    int64_t a_offset = 0, b_offset = 0;
    for (int axis = 0, n = c_shape.size(); axis < n; ++axis) {
      a_offset += ((a_remain / a_strides[axis]) % a_shape[axis]) * a_strides[axis];
      b_offset += ((b_remain / b_strides[axis]) % b_shape[axis]) * b_strides[axis];
      a_remain = a_remain % a_strides[axis];
      b_remain = b_remain % b_strides[axis];
    }
    
    float a_dequantized = a_scale * (static_cast<int>(a_quantized[a_offset]) - a_zero_int);
    float b_dequantized = b_scale * (static_cast<int>(b_quantized[b_offset]) - b_zero_int);
    c[offset] = static_cast<T>(std::round(calc(a_dequantized, b_dequantized) / c_scale) + c_zero_int);
  }

  test.AddInput<float>("C_scale", {}, {c_scale});
  test.template AddInput<T>("C_zero_point", {}, {c_zero});

  test.template AddOutput<T>("C", c_shape, c);
  test.Run();
}

TEST(QuantizeLinearContribMathOpTest, AddUInt8) {
  std::vector<float> A = {0.8f, 0.3f, 0.1f, -0.5f, -0.2f, -0.6f, -0.9f, 0.0f, -1.0f, 1.0f};
  std::vector<int64_t> A_shape = {2,  5};
  std::vector<int64_t> C_shape = A_shape;
  float A_scale = 2.0f / 255.0f;
  uint8_t A_zero = 128;
  std::vector<float> B = {-2.0f, -1.0f, 2.0f, 0.3f, 0.9f};
  std::vector<int64_t> B_shape = {1, 5};
  float B_scale = 4.0f / 255.0f;
  uint8_t B_zero = 128;

  float C_scale = 6.0f / 255.0f;
  uint8_t C_zero = 128;

  RunQLinearMathTestFromFloat(
    "QLinearAdd", 
    [](float a_dequantized, float b_dequantized) { return a_dequantized + b_dequantized; }, 
    A, A_shape, A_scale, A_zero, B, B_shape, B_scale, B_zero, C_shape, C_scale, C_zero);
}

TEST(QuantizeLinearContribMathOpTest, AddInt8) {
  std::vector<float> A = {0.8f, 0.3f, 0.1f, -0.5f, -0.2f, -0.6f, -0.9f, 0.0f, -1.0f, 1.0f};
  std::vector<int64_t> A_shape = {2, 5};
  std::vector<int64_t> C_shape = A_shape;
  float A_scale = 2.0f / 255.0f;
  int8_t A_zero = 0;
  std::vector<float> B = {-2.0f, -1.0f, 2.0f, 0.3f, 0.9f};
  std::vector<int64_t> B_shape = {1, 5};
  float B_scale = 4.0f / 255.0f;
  int8_t B_zero = 0;

  float C_scale = 6.0f / 255.0f;
  int8_t C_zero = 0;

  RunQLinearMathTestFromFloat(
    "QLinearAdd", 
    [](float a_dequantized, float b_dequantized) { return a_dequantized + b_dequantized; }, 
    A, A_shape, A_scale, A_zero, B, B_shape, B_scale, B_zero, C_shape, C_scale, C_zero);
}

}  // namespace test
}  // namespace onnxruntime

