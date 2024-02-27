cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 50 CACHE STRING "CUDA architectures" FORCE)
project(main LANGUAGES CXX CUDA)

# Set OpenCV_DIR to the directory containing OpenCVConfig.cmake
# Example: set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
# Update this path according to your OpenCV installation
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# Now use the variable provided by find_package
include_directories(${OpenCV_INCLUDE_DIRS})

# Project configuration
# Attempt to force CMake to recognize CMAKE_CUDA_ARCHITECTURES
mark_as_advanced(CLEAR CMAKE_CUDA_ARCHITECTURES)

add_executable(main main.cu)

set_target_properties(main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED YES)

# Link against OpenCV libraries
#target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
