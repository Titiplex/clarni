cmake_minimum_required(VERSION 4.2)
project(ML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

add_library(ML STATIC
        src/layer.cpp
        src/matrix.cpp
        src/network.cpp
        src/dataset.cpp
        src/compute_device.cpp
)

target_include_directories(ML PUBLIC headers)

find_package(Threads REQUIRED)

target_link_libraries(ML PRIVATE Threads::Threads)

find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

target_sources(ML PRIVATE src/gpu/matrix.cu src/gpu/backend_stub.cpp)

target_link_libraries(ML PRIVATE CUDA::cudart)

target_compile_definitions(ML PRIVATE
        HAS_CUDA=1
        HAS_CUDA_KERNELS=1
)

set_target_properties(ML PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES 89
)

message(STATUS "CUDA Toolkit : ${CUDAToolkit_VERSION}")
message(STATUS "Compilateur CUDA : ${CMAKE_CUDA_COMPILER}")