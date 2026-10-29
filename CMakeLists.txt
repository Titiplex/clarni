cmake_minimum_required(VERSION 4.2)
project(clarni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ML_ENABLE_CUDA "Compiler le backend CUDA (necessite nvcc/MSVC sous Windows)" OFF)
option(ML_BUILD_EXAMPLES "Compiler les exemples" ON)
option(ML_BUILD_BENCHMARKS "Compiler les mesures CPU/CUDA" OFF)
option(ML_USE_CUBLAS "Utiliser cuBLAS pour les produits Tensor CUDA" OFF)
if (ML_USE_CUBLAS AND NOT ML_ENABLE_CUDA)
    message(FATAL_ERROR "ML_USE_CUBLAS exige ML_ENABLE_CUDA=ON")
endif ()
set(ML_CUDA_BUILT OFF)

if (ML_ENABLE_CUDA)
    # Valeur reprise du projet ; adapter aux GPU cibles.
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "Architectures CUDA visees")
    include(CheckLanguage)
    check_language(CUDA)
    if (NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA demande mais aucun compilateur CUDA utilisable n'a ete trouve")
    endif ()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(ML_CUDA_BUILT ON)
endif ()

set(ML_CORE_SOURCES
        src/layer.cpp
        src/matrix.cpp
        src/network.cpp
        src/dataset.cpp
        src/compute_device.cpp
        src/tensor.cpp src/optimizer.cpp src/preprocessing.cpp src/checkpoint.cpp src/autograd.cpp src/nd.cpp src/module.cpp src/embedding.cpp src/convolution.cpp src/recurrent.cpp src/gated_recurrent.cpp src/attention.cpp)
add_library(ML STATIC ${ML_CORE_SOURCES})
add_library(clarni::ML ALIAS ML)
target_include_directories(ML PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/headers")
target_compile_features(ML PUBLIC cxx_std_23)

if (ML_CUDA_BUILT)
    target_sources(ML PRIVATE src/gpu/matrix.cu src/gpu/tensor.cu)
    target_link_libraries(ML PRIVATE CUDA::cudart_static)
    if (ML_USE_CUBLAS)
        target_link_libraries(ML PRIVATE CUDA::cublas)
        target_compile_definitions(ML PRIVATE ML_USE_CUBLAS=1)
    endif ()
    target_compile_definitions(ML PRIVATE HAS_CUDA=1 HAS_CUDA_KERNELS=1)
    set_target_properties(ML PROPERTIES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
            CUDA_RUNTIME_LIBRARY Static)
else ()
    target_sources(ML PRIVATE src/gpu/backend_stub.cpp src/gpu/tensor_stub.cpp)
    # Ne pas definir HAS_CUDA=0 si votre futur code utilise #ifdef HAS_CUDA.
endif ()

if (ML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (ML_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif ()

include(CTest)
if (BUILD_TESTING)
    add_subdirectory(tests)
endif ()
