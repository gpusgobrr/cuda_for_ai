import os
from typing import Tuple
import torch
from torch.utils.cpp_extension import load_inline
from loguru import logger
import triton
import triton.testing


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


def create_extension(kernel_name):
    file_dir = os.path.dirname(__file__)
    # Load the CUDA code
    source_file = f"{file_dir}/../../src/softmax/{kernel_name}.cu"
    header_file = f"{file_dir}/../../src/softmax/{kernel_name}.cuh"

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


def get_inputs(batch_size: int, sequence_length: int):
    dense_size = (batch_size, sequence_length)
    return torch.randn(dense_size, device="cuda", dtype=torch.float32)


basic_softmax_cuda = create_extension("basic_softmax")
shmem_softmax_cuda = create_extension("shmem_softmax")


def basic_softmax(input_tensor):
    # Call the CUDA kernel
    return basic_softmax_cuda.basic_softmax(input_tensor)


def shmem_softmax(input_tensor):
    print(f"Input tensor: {input_tensor.shape}")
    # Call the CUDA kernel
    return shmem_softmax_cuda.shmem_softmax(input_tensor)


def torch_softmax(input_tensor):
    # Call the PyTorch softmax
    return torch.softmax(input_tensor, dim=-1)


torch_softmax_scripted = torch.jit.script(
    torch_softmax, example_inputs=[get_inputs(2, 2)]
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "batch_size",
            "sequence_length",
        ],  # Argument names to use as an x-axis for the plot.
        x_vals=[(2**i, 2**j) for i, j in zip(range(2, 12, 1), range(2, 26, 2))],
        # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "basic_softmax_cuda",
            "shmem_softmax_cuda",
            "torch",
            "torch_scripted",
        ],  # Possible values for `line_arg`.
        line_names=[
            "basic_softmax_cuda",
            "shmem_softmax_cuda",
            "torch",
            "torch_scripted",
        ],  # Label name for the lines.
        styles=[
            ("blue", "-"),
            ("orange", "-"),
            ("green", "-."),
            ("red", "--"),
        ],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(batch_size, sequence_length, provider):
    input_tensor = get_inputs(batch_size, sequence_length)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "basic_softmax_cuda":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: basic_softmax_cuda.basic_softmax(input_tensor),
            quantiles=quantiles,
        )
    if provider == "shmem_softmax_cuda":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: shmem_softmax_cuda.shmem_softmax(input_tensor),
            quantiles=quantiles,
        )
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_softmax(input_tensor),
            quantiles=quantiles,
        )

    if provider == "torch_scripted":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_softmax_scripted(input_tensor),
            quantiles=quantiles,
        )
    gbps = lambda ms: 12 * (batch_size * sequence_length) / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    #
    # # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    benchmark.run(print_data=True, show_plots=True)
