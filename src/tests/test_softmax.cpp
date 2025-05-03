//
// Created by ksharma on 4/27/25.
//
#include "imports/catch_amalgamated.hpp"
#include <torch/torch.h>
#include "../softmax/basic_softmax.cuh"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define SOFTMAX_TEST_CASE(BATCH, SEQ_LEN)                                  \
TEST_CASE("Softmax_" TOSTRING(BATCH) "x" TOSTRING(SEQ_LEN), "[basic_softmax]") { \
torch::manual_seed(42);                                                \
const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA); \
const auto input_tensor = torch::randn({BATCH, SEQ_LEN}, options);     \
const auto output = basic_softmax(input_tensor);                       \
const auto torch_output = torch::softmax(input_tensor, -1);           \
REQUIRE(output.allclose(torch_output, 1e-5));                          \
}

SOFTMAX_TEST_CASE(1024, 3076)
SOFTMAX_TEST_CASE(512, 1024)
SOFTMAX_TEST_CASE(256, 768)
SOFTMAX_TEST_CASE(128, 512)
SOFTMAX_TEST_CASE(512, 128)
SOFTMAX_TEST_CASE(32, 16)
SOFTMAX_TEST_CASE(16, 8)
SOFTMAX_TEST_CASE(8, 4)
SOFTMAX_TEST_CASE(4, 2)



