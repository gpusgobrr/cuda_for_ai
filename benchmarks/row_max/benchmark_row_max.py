import os
from typing import Optional, Tuple
import torch
from torch.utils.cpp_extension import load_inline
from loguru import logger
import triton
import triton.testing
import triton.language as tl

FILE_DIR = os.path.dirname(__file__)
print(f"File path: {FILE_DIR}")


def get_cuda_code(cuda_file: str, header_file: str) -> Tuple[str, str]:
    with open(cuda_file) as f:
        cuda_code = "".join([f for f in f.readlines() if not f.startswith("#include")])
        print(cuda_code)

    with open(header_file) as f:
        header_code = "".join(
            [f for f in f.readlines() if not f.startswith("#include")]
        )
        print(header_code)

    return cuda_code, header_code


def create_extension(kernel_name: str, file_name: Optional[str] = None):
    file_name = file_name or kernel_name
    file_dir = os.path.dirname(__file__)
    # Load the CUDA code
    source_file = f"{file_dir}/../../src/softmax/{file_name}.cu"
    header_file = f"{file_dir}/../../src/softmax/{file_name}.cuh"

    logger.info(f"Source file: {source_file}")
    logger.info(f"Header file: {header_file}")

    cuda_code, header_code = get_cuda_code(source_file, header_file)
    build_dir = os.path.join(file_dir, "build")
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    build_dir = os.path.join(build_dir, kernel_name)
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    logger.info(f"Build directory: {build_dir}")
    # Load the extension
    extension = load_inline(
        name=f"softmax_{kernel_name}_extension",
        cpp_sources=header_code,
        cuda_sources=cuda_code,
        functions=[kernel_name],
        with_cuda=True,
        verbose=True,
        extra_cuda_cflags=["-O2"],
        build_directory=build_dir,
        # extra_cuda_cflags=['--expt-relaxed-constexpr']
    )

    logger.info(f"Extension loaded: {extension}")

    return extension


# @triton.jit
# def row_max_kernel_optimized(
#     input_ptr,
#     output_ptr,
#     batch_size,
#     sequence_length,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     """
#     Optimized version that processes multiple elements per thread
#     for better performance on very long sequences.
#     """
#     row_idx = tl.program_id(0)

#     if row_idx >= batch_size:
#         return

#     row_start_ptr = input_ptr + row_idx * sequence_length
#     max_val = float("-inf")

#     # Process larger chunks - each thread handles multiple BLOCK_SIZE chunks
#     num_blocks = tl.cdiv(sequence_length, BLOCK_SIZE)

#     for block_idx in range(num_blocks):
#         block_start = block_idx * BLOCK_SIZE
#         offsets = block_start + tl.arange(0, BLOCK_SIZE)
#         mask = offsets < sequence_length

#         # Load and process block
#         block_data = tl.load(row_start_ptr + offsets, mask=mask, other=float("-inf"))
#         block_max = tl.max(block_data)
#         max_val = tl.maximum(max_val, block_max)

#     tl.store(output_ptr + row_idx, max_val)


# def row_max_triton_optimized(input_tensor: torch.Tensor) -> torch.Tensor:
#     """
#     Optimized version for very long sequences.
#     """
#     assert input_tensor.dim() == 2, "Input must be 2D tensor"
#     assert input_tensor.is_cuda, "Input tensor must be on CUDA device"
#     assert input_tensor.dtype == torch.float32, "Input tensor must be float32"
#     assert input_tensor.is_contiguous(), "Input tensor must be contiguous"

#     batch_size, sequence_length = input_tensor.shape
#     output = torch.empty(
#         (batch_size,), device=input_tensor.device, dtype=input_tensor.dtype
#     )

#     # Use larger block size for better vectorization
#     BLOCK_SIZE = min(1024, triton.next_power_of_2(sequence_length))

#     grid = (batch_size,)

#     row_max_kernel_optimized[grid](
#         input_tensor,
#         output,
#         batch_size,
#         sequence_length,
#         BLOCK_SIZE=BLOCK_SIZE,
#     )

#     return output


def get_inputs(batch_size: int, sequence_length: int):
    dense_size = (batch_size, sequence_length)
    return torch.randn(dense_size, device="cuda", dtype=torch.float32)


rowmax_cuda = create_extension("row_max", "row_reductions")


def cuda_row_max(input_tensor):
    # Call the CUDA kernel
    return rowmax_cuda.row_max(input_tensor)


def torch_row_max(input_tensor):
    # Call the PyTorch softmax
    return torch.max(input_tensor, dim=-1)


torch_row_max_scripted = torch.jit.script(
    torch_row_max, example_inputs=[get_inputs(2, 2)]
)


def memory_throughput_gbps(batch_size, sequence_length, time_ms):
    input_bytes = batch_size * sequence_length * 4  # 4 bytes per float32
    output_bytes = batch_size * 4  # 4 bytes per float32 output
    total_bytes = input_bytes + output_bytes
    return total_bytes / (time_ms * 1e-3) / 1e9  # GB/s


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "batch_size",
            "sequence_length",
        ],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            (2**i, 2 ** (j // 2)) for i, j in zip(range(0, 22, 1), range(0, 22, 1))
        ],
        # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "cuda_row_max",
            "torch",
            "torch_scripted",
            # "triton_row_max",
        ],  # Possible values for `line_arg`.
        line_names=[
            "cuda_row_max",
            "torch",
            "torch_scripted",
            # "triton_row_max",
        ],  # Label name for the lines.
        styles=[
            ("blue", "-"),
            ("green", "-."),
            ("red", "--"),
            # ("orange", ":"),
        ],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(batch_size, sequence_length, provider):
    input_tensor = get_inputs(batch_size, sequence_length)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cuda_row_max":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: cuda_row_max(input_tensor),
            quantiles=quantiles,
        )
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_row_max(input_tensor),
            quantiles=quantiles,
        )

    if provider == "torch_scripted":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_row_max_scripted(input_tensor),
            quantiles=quantiles,
        )
    # if provider == "triton_row_max":
    #     ms, min_ms, max_ms = triton.testing.do_bench(
    #         lambda: row_max_triton_optimized(input_tensor),
    #         quantiles=quantiles,
    #     )
    gbps = lambda ms: memory_throughput_gbps(
        batch_size=batch_size, sequence_length=sequence_length, time_ms=ms
    )
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    #
    # # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    benchmark.run(print_data=True, show_plots=True)
