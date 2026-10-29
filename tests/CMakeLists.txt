foreach (component matrix functions network device gradients modes activations tensor optimizers tasks training checkpoints autograd residency cuda nd embedding convolution recurrent gated_recurrent attention inference)
    add_executable(clarni_test_${component} test_${component}.cpp)
    target_link_libraries(clarni_test_${component} PRIVATE ML)
    add_test(NAME clarni.${component} COMMAND clarni_test_${component})
endforeach ()
target_compile_definitions(clarni_test_device PRIVATE
        CLARNI_EXPECT_CUDA=$<BOOL:${ML_CUDA_BUILT}>)
set_tests_properties(clarni.cuda PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(clarni.residency PROPERTIES SKIP_RETURN_CODE 77)

add_executable(clarni_test_residency_mock test_residency.cpp mock_resident_backend.cpp)
foreach (source IN LISTS ML_CORE_SOURCES)
    target_sources(clarni_test_residency_mock PRIVATE "${PROJECT_SOURCE_DIR}/${source}")
endforeach ()
target_include_directories(clarni_test_residency_mock PRIVATE "${PROJECT_SOURCE_DIR}/headers")
target_compile_definitions(clarni_test_residency_mock PRIVATE CLARNI_MOCK_GPU=1)
add_test(NAME clarni.residency_mock COMMAND clarni_test_residency_mock)

# Ce test substitue un faux backend a l'implementation CUDA/stub : pas de lien ML.
add_executable(clarni_test_dispatch test_dispatch.cpp
        "${PROJECT_SOURCE_DIR}/src/matrix.cpp"
        "${PROJECT_SOURCE_DIR}/src/compute_device.cpp")
target_include_directories(clarni_test_dispatch PRIVATE "${PROJECT_SOURCE_DIR}/headers")
add_test(NAME clarni.dispatch COMMAND clarni_test_dispatch)

find_package(Threads REQUIRED)
target_link_libraries(clarni_test_inference PRIVATE Threads::Threads)
