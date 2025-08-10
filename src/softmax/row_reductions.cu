#include <cuda_runtime.h>
#include <torch/torch.h>
#include "row_reductions.cuh"

__global__ void row_max_kernel(const float *input, float *output, const int batch_size, const int num_elements)
{
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    unsigned int warps_per_block = (blockDim.x + 31) / 32;

    if (row < static_cast<unsigned int>(batch_size))
    {
        extern __shared__ float shared_data[];
        float max_val = -INFINITY;
        for (unsigned int i = tid; i < static_cast<unsigned int>(num_elements); i += blockDim.x)
        {
            float val = input[row * num_elements + i];
            max_val = fmaxf(max_val, val);
        }

        for (unsigned int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }

        if (lane_id == 0) {
            shared_data[warp_id] = max_val;
        }
        __syncthreads();

        if (warp_id == 0) {
            max_val = (tid < warps_per_block) ? shared_data[tid] : -INFINITY;

            for (unsigned int offset = 16; offset > 0; offset /= 2) {
                max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
            }

            if (tid == 0) {
                output[row] = max_val;
            }
        }
    }
}

int get_optimal_block_size(int num_elements) {
    int block_size = 256;
    if (num_elements < 128) {
        block_size = 128;
    } else if (num_elements < 256) {
        block_size = 256;
    } else {
        block_size = 512;
    }
    return ((block_size + 31) / 32) * 32;
}

torch::Tensor row_max(const torch::Tensor &input)
{
    const int batch_size = static_cast<int>(input.size(0));
    auto output = torch::empty({batch_size}, input.options());
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(output.device().is_cuda(), "Output tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output tensor must be float32");
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D (batch_size × num_elements)");
    const int sequence_length = static_cast<int>(input.size(1));
    TORCH_CHECK(output.dim() == 1, "Output tensor must be 1D (batch_size)");
    const auto d_input = input.data_ptr<float>();
    const auto d_output = output.data_ptr<float>();
    auto threads = static_cast<unsigned int>(get_optimal_block_size(sequence_length));
    auto blocks_per_grid = static_cast<unsigned int>(batch_size);
    unsigned int warps_per_block = (threads + 31) / 32;
    size_t shared_mem_size = warps_per_block * sizeof(float);
    row_max_kernel<<<blocks_per_grid, threads, shared_mem_size>>>(d_input, d_output, batch_size, sequence_length);
    return output;
}
