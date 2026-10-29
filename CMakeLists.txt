cmake_minimum_required(VERSION 4.2)
project(clarni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ML_ENABLE_CUDA "Compiler le backend CUDA (necessite nvcc/MSVC sous Windows)" OFF)
option(ML_BUILD_EXAMPLES "Compiler les exemples" ON)
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

add_library(ML STATIC
        src/layer.cpp
        src/matrix.cpp
        src/network.cpp
        src/dataset.cpp
        src/compute_device.cpp)
target_sources(ML PRIVATE src/tensor.cpp)
add_library(clarni::ML ALIAS ML)
target_include_directories(ML PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/headers")
target_compile_features(ML PUBLIC cxx_std_23)

if (ML_CUDA_BUILT)
    target_sources(ML PRIVATE src/gpu/matrix.cu)
    target_link_libraries(ML PRIVATE CUDA::cudart_static)
    target_compile_definitions(ML PRIVATE HAS_CUDA=1 HAS_CUDA_KERNELS=1)
    set_target_properties(ML PROPERTIES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
            CUDA_RUNTIME_LIBRARY Static)
else ()
    target_sources(ML PRIVATE src/gpu/backend_stub.cpp)
    # Ne pas definir HAS_CUDA=0 si votre futur code utilise #ifdef HAS_CUDA.
endif ()

if (ML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

include(CTest)
if (BUILD_TESTING)
    add_subdirectory(tests)
endif ()
