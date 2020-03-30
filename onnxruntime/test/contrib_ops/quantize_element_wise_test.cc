// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace test {

template <typename T, typename CalcFunction>
RunQLinearMathTestFromFloat(const char* op_name, CalcFunction calc,
    std::vector<float>& a, std::vector<int64_t>& a_shape, float a_scope, T a_zero,
    std::vector<float>& b, std::vector<int64_t>& b_shape, float b_scope, T b_zero,
    std::vector<T>& c, float c_scope, T c_zero)
{
  OpTester test(op_name, 1, onnxruntime::kMSDomain);

  Broadcaster broadcaster(a_shape, b_shape);
  const auto& result_shape = broadcaster.output_shape_;
  auto result_size = TensorShape(result_shape).Size();
  auto span_size = broadcaster.GetSpanSize();
  if (c.size() == 0) {
    c.resize(result_size);
    const float* a_span = a.data();
    const float* b_span = b.data(); 
    for (int64_t offset = 0; offset < result_size; offset += span_size) {
        const float* pa = a_span, *pb = b_span, *pc = c.data() + offset;
        for (int64_t k = 0; k < span_size; ++k) {
            T a_quantized = static_cast<T>(*pa / a_scope + a_zero);
            float a_dequantized = a_scope * (static_cast<int>(a_quantized) - a_zero);
            T b_quantized = static_cast<T>(*pb / b_scope + b_zero);
            float b_dequantized = b_scope * (static_cast<int>(b_quantized) - b_zero);
            float c_value = calc(a_dequantized, b_dequantized);
            *pc++ = static_cast<T>(c_value / c_scope + c_zero);
            ++pa;
            ++pb;
        }
        a_span = a.data() + broadcaster.iterator1_.AdvanceBy(span_size);
        b_span = b.data() + broadcaster.iterator2_.AdvanceBy(span_size);
    }
  }

  std::vector<T> a_quantized(a.size());
  for (size_t i = 0, sz = a.size(); i < sz; ++i) {
      a_quantized[i] = static_cast<T>(a[i] / a_scope + a_zero);
  }
  test.AddInput("A", a_shape, a_quantized);
  std::vector<float> a_scope_vec(1, a_scope);
  test.AddInput("A_scale", {1}, a_scope_vec);
  std::vector<T> a_zero_vec(1, a_zero);
  test.AddInput("A_zero_point", {1}, a_zero_vec);

  std::vector<T> b_quantized(b.size());
  for (size_t i = 0, sz = b.size(); i < sz; ++i) {
      b_quantized[i] = static_cast<T>(b[i] / b_scope + b_zero);
  }
  test.AddInput("B", b_shape, b_quantized);
  std::vector<float> b_scope_vec(1, b_scope);
  test.AddInput("B_scale", {1}, b_scope_vec);
  std::vector<T> b_zero_vec(1, b_zero);
  test.AddInput("B_zero_point", {1}, b_zero_vec);

  std::vector<float> c_scope_vec(1, c_scope);
  test.AddInput("C_scale", {1}, c_scope_vec);
  std::vector<T> c_zero_vec(1, c_zero);
  test.AddInput("C_zero_point", {1}, c_zero_vec);

  test.AddOutput("C", result_shape, c);
  test.Run();
}

TEST(QuantizeLinearContribMathOpTest, AddUInt8) {
  OpTester test("QLinearAdd", 1, onnxruntime::kMSDomain);
  std::vector<float> A = {0.8f, 0.3f, 0.1f, -0.5f, -0.2f, -0.6f, -0.9f, 0.0f, -1.0f, 1.0f};
  std::vector<int64_t> A_shape = {A.size()};
  float A_scale = 2.0f;
  uint8_t A_zero = 128;
  std::vector<float> B = {-2.0f, -1.0f, 2.0f, 0.3f, 0.9f};
  std::vector<int64_t> B_shape = {B.size()};
  float B_scale = 4.0f;
  uint8_t B_zero = 128;

  foat C_scale = 6.0f;
  uint8_t C_zero = 128;

  RunQLinearMathTestFromFloat<uint8_t>(
    "QLinearAdd", [](float a_dequantized, float b_dequantized) { return a_dequantized + b_dequantized; }
    A, A_shape, A_scale, A_zero, B, B_shape, B_scale, B_zero, C_scale, C_zero
  );
}

TEST(QuantizeLinearContribMathOpTest, AddInt8) {
  OpTester test("QLinearAdd", 1, onnxruntime::kMSDomain);
  std::vector<float> A = {0.8f, 0.3f, 0.1f, -0.5f, -0.2f, -0.6f, -0.9f, 0.0f, -1.0f, 1.0f};
  std::vector<int64_t> A_shape = {A.size()};
  float A_scale = 2.0f;
  int8_t A_zero = 0;
  std::vector<float> B = {-2.0f, -1.0f, 2.0f, 0.3f, 0.9f};
  std::vector<int64_t> B_shape = {B.size()};
  float B_scale = 4.0f;
  int8_t B_zero = 0;

  foat C_scale = 6.0f;
  int8_t C_zero = 0;

  RunQLinearMathTestFromFloat<int8_t>(
    "QLinearAdd", 
    [](float a_dequantized, float b_dequantized) { return a_dequantized + b_dequantized; }
    A, A_shape, A_scale, A_zero, B, B_shape, B_scale, B_zero, C_scale, C_zero
  );
}

}  // namespace test
}  // namespace onnxruntime
