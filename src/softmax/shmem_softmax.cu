#include <cuda_runtime.h>
#include <torch/torch.h>
#include "shmem_softmax.cuh"


// CUDA kernel for softmax operation
__global__ void shmem_softmax_kernel(float *input, float *output, const int batch_size, const int num_elements) {
    extern __shared__ float shared_data[]; // Shared memory for intermediate results
    int row = blockIdx.x; // Each block processes one row
    int tid = threadIdx.x;
    // printf("-------- Row: %d, Thread: %d --------- \n", row, tid);
    if (row < batch_size) {
        // Load row into shared memory
        float max_val = -INFINITY;
        for (int i = tid; i < num_elements; i += 1) {
            shared_data[i] = input[row * num_elements + i];
            max_val = fmaxf(max_val, shared_data[i]);
            // printf("Row: %d, Thread: %d, Value: %f\n", row, tid, shared_data[i]);
        }
        __syncthreads();

        // printf("Row: %d, Thread: %d, Max Value: %f\n", row, tid, max_val);

        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = tid; i < num_elements; i += 1) {
            shared_data[i] = expf(shared_data[i] - max_val);
            sum += shared_data[i];
            // printf("Row: %d, Thread: %d, Max Value: %f, Exp Value: %f\n, Sum: %f\n", row, tid, max_val, shared_data[i], sum);
        }

        // printf("Row: %d, Thread: %d, Sum before sync: %f\n", row, tid, sum);
        __syncthreads();

        // Normalize and write back
        for (int i = tid; i < num_elements; i += 1) {
            output[row * num_elements + i] = shared_data[i] / sum;
            // printf("Row: %d, Thread: %d, Output Value: %f\n", row, tid, output[row * num_elements + i]);
        }
    }
}

// Host function to launch the kernel with PyTorch tensors
torch::Tensor shmem_softmax(const torch::Tensor &input)
{
    auto output = torch::empty_like(input);
    // Make sure tensors are on CUDA
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output tensor must be float32");

    // Extract dimensions
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (batch_size × num_elements)");
    const int batch_size = static_cast<int>(input.size(0));
    const int sequence_length = static_cast<int>(input.size(1));

    // Get raw pointers to the tensor data
    const auto d_input = input.data_ptr<float>();
    const auto d_output = output.data_ptr<float>();

    size_t threads = std::min<size_t>(1024, 1 << static_cast<int>(std::floor(std::log2(sequence_length))));
    size_t blocks_per_grid = batch_size;
    // std::cout << "Threads: " << threads  << " | Blocks per grid: " << blocks_per_grid << std::endl;

    shmem_softmax_kernel<<<blocks_per_grid, threads, sequence_length * sizeof(float)>>>(
        d_input, d_output, batch_size, sequence_length
    );
    return output;
}
