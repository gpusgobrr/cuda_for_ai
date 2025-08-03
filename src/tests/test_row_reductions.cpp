//
// Created by ksharma on 4/27/25.
//
#include "imports/catch_amalgamated.hpp"
#include <torch/torch.h>
#include "../softmax/row_reductions.cuh"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define ROW_MAX_TEST_CASE(BATCH, SEQ_LEN)                                                                          \
    TEST_CASE("basic_row_max_" TOSTRING(BATCH) "x" TOSTRING(SEQ_LEN), "[basic_softmax]")                           \
    {                                                                                                              \
        torch::manual_seed(42);                                                                                    \
        const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);                   \
        const auto input_tensor = torch::randn({BATCH, SEQ_LEN}, options);                                         \
        const auto output = row_max(input_tensor);                                                                 \
        const auto torch_output = std::get<0>(torch::max(input_tensor, -1));                                       \
        auto mask = torch::isclose(torch_output, output, 1e-5);                                                    \
        auto not_allclose = ~mask;                                                                                 \
        auto indices = torch::nonzero(not_allclose);                                                               \
        std::cout << "Indices: " << indices << std::endl;                                                          \
        for (int64_t idx = 0; idx < indices.numel(); ++idx)                                                        \
        {                                                                                                          \
            auto i = indices[idx].item<int64_t>();                                                                 \
            std::cout << "Mismatch at index: " << output[i].item() << ", " << torch_output[i].item() << std::endl; \
        }                                                                                                          \
        REQUIRE(output.allclose(torch_output, 1e-4));                                                              \
    }

ROW_MAX_TEST_CASE(4096, 4096)
ROW_MAX_TEST_CASE(2048, 3076)
ROW_MAX_TEST_CASE(1024, 3076)
ROW_MAX_TEST_CASE(512, 1024)
ROW_MAX_TEST_CASE(256, 768)
ROW_MAX_TEST_CASE(128, 512)
ROW_MAX_TEST_CASE(512, 128)
ROW_MAX_TEST_CASE(32, 16)
ROW_MAX_TEST_CASE(16, 8)
ROW_MAX_TEST_CASE(8, 4)
ROW_MAX_TEST_CASE(4, 2)
