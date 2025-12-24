cmake_minimum_required(VERSION 3.18)
project(sgdma_test CUDA CXX)

# Find CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Build test executable
add_executable(sgdma_test sgdma_test.cpp)
target_link_libraries(sgdma_test cudart)

# Set output directory
set_target_properties(sgdma_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
