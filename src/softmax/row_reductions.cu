#include <cuda_runtime.h>
#include <torch/torch.h>
#include "row_reductions.cuh"


// CUDA kernel for softmax operation
__global__ void row_max_kernel(float *input, float *output, const int batch_size, const int num_elements) {
    extern __shared__ float shared_data[]; // Shared memory for intermediate results
    int row = blockIdx.x; // Each block processes one row
    int tid = threadIdx.x;

    if (row < batch_size) {
        // Each thread finds max of its assigned elements
        float max_val = -INFINITY;
        for (int i = tid; i < num_elements; i += blockDim.x) {
            float val = input[row * num_elements + i];
            max_val = fmaxf(max_val, val);
        }

        // Store local max in shared memory
        shared_data[tid] = max_val;
        __syncthreads();

        // Parallel reduction to find global max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
            }
            __syncthreads();
        }

        // Thread 0 writes the final result
        if (tid == 0) {
            output[row] = shared_data[0];
        }
    }
}

// Host function to launch the kernel with PyTorch tensors
torch::Tensor row_max(const torch::Tensor &input)
{
    const int batch_size = static_cast<int>(input.size(0));
    auto output = torch::empty({batch_size}, input.options());
    // Make sure tensors are on CUDA
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output tensor must be float32");

    // Extract dimensions
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (batch_size × num_elements)");
    const int sequence_length = static_cast<int>(input.size(1));
    TORCH_CHECK(output.dim() == 1, "Output tensor must be 1D (batch_size)");


    // Get raw pointers to the tensor data
    const auto d_input = input.data_ptr<float>();
    const auto d_output = output.data_ptr<float>();

    size_t threads = std::min<size_t>(1024, 1 << static_cast<int>(std::floor(std::log2(sequence_length))));
    size_t blocks_per_grid = batch_size;
    std::cout << "Threads: " << threads  << " | Blocks per grid: " << blocks_per_grid << std::endl;

    row_max_kernel<<<blocks_per_grid, threads, sequence_length * sizeof(float)>>>(
        d_input, d_output, batch_size, sequence_length
    );
    return output;
}
