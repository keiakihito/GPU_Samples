cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 50 CACHE STRING "CUDA architectures" FORCE)
project(Device_query LANGUAGES CXX CUDA)

# Attempt to force CMake to recognize CMAKE_CUDA_ARCHITECTURES
mark_as_advanced(CLEAR CMAKE_CUDA_ARCHITECTURES)


add_executable(Device_query device-query.cu)

set_target_properties(Device_query PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES)
