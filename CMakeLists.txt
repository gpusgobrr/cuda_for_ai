cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cuda_for_ai)

file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/src/*/*.cuh")
file(GLOB test_import_files "${CMAKE_SOURCE_DIR}/src/tests/imports/*")


# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu121.zip
list(APPEND CMAKE_PREFIX_PATH /home/ksharma/dev/git/cuda-learn/third-party/libtorch)
list(APPEND CMAKE_PREFIX_PATH /home/ksharma/local/cuda)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# tests
add_executable(test_softmax ${CMAKE_SOURCE_DIR}/src/tests/test_softmax.cpp ${test_import_files} ${gpu_source_files})
target_link_libraries(test_softmax "${TORCH_LIBRARIES}")

add_executable(test_cuda_capabilities ${CMAKE_SOURCE_DIR}/src/tests/test_cuda_capabilities.cpp ${test_import_files} ${gpu_source_files})
target_link_libraries(test_cuda_capabilities "${TORCH_LIBRARIES}")

enable_testing()

add_test(NAME test_softmax COMMAND test_softmax)
add_test(NAME test_cuda_capabilities COMMAND test_cuda_capabilities)

# NOTES
# needed to pass -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=-std=c++17 -DCMAKE_CUDA_COMPILER=/home/ksharma/local/cuda/bin/nvcc
