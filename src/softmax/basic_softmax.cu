/**
 * Basic CUDA kernel for softmax operation with PyTorch integration
 * This kernel computes softmax for each row in a 2D matrix
 * It can be used as a PyTorch C++/CUDA extension
 */

#include <cuda_runtime.h>
#include <torch/torch.h>
#include "basic_softmax.cuh"

// CUDA kernel for softmax operation
__global__ void basic_softmax_kernel(float* input, float* output, const int batch_size, const int num_elements) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes one row (one sample in the batch)
    if (tid < batch_size) {
        // Find max value in the row (for numerical stability)
        float max_val = -INFINITY;
        for (int i = 0; i < num_elements; ++i) {
            int idx = tid * num_elements + i;
            if (input[idx] > max_val) {
                max_val = input[idx];
            }
        }

        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < num_elements; ++i) {
            int idx = tid * num_elements + i;
            // Subtract max for numerical stability before exp
            output[idx] = expf(input[idx] - max_val);
            sum += output[idx];
        }

        // Normalize to get softmax values
        for (int i = 0; i < num_elements; ++i) {
            int idx = tid * num_elements + i;
            output[idx] /= sum;
        }
    }
}

// Host function to launch the kernel with PyTorch tensors
torch::Tensor basic_softmax(const torch::Tensor& input) {
    auto output = torch::empty_like(input);
    // Make sure tensors are on CUDA
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output tensor must be float32");

    // Extract dimensions
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (batch_size × num_elements)");
    const int batch_size = static_cast<int>(input.size(0));
    const int num_elements = static_cast<int>(input.size(1));

    // Get raw pointers to the tensor data
    const auto d_input = input.data_ptr<float>();
    const auto d_output = output.data_ptr<float>();

    // Calculate grid and block dimensions
    size_t threads = (num_elements >= 1024) ? 1024 : (1 << static_cast<int>(log2f64(num_elements)));
    size_t blocks_per_grid = (batch_size + threads - 1) / threads;

    // Launch the kernel
    basic_softmax_kernel<<<blocks_per_grid, threads>>>(d_input, d_output, batch_size, num_elements);

    return output;
}
