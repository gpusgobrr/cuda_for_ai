
#include "imports/catch_amalgamated.hpp"
#include <cuda_runtime.h>
#include <iostream>

TEST_CASE("CudaCapabilities", "[Details]")
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << "\n\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes\n";
        std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
        std::cout << "  Warp size: " << prop.warpSize << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Number of SMs: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << "\n";
        std::cout << "  Max grid dimensions: ["
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << "]\n";
        std::cout << "  Max threads dim (block): ["
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << "]\n";
        std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz\n";
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz\n";
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
        std::cout << std::endl;
    }
}