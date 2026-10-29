add_executable(clarni_example_matrix matrix_product.cpp)
target_link_libraries(clarni_example_matrix PRIVATE ML)

add_executable(clarni_example_network network_smoke.cpp)
target_link_libraries(clarni_example_network PRIVATE ML)

add_executable(clarni_example_xor xor.cpp)
target_link_libraries(clarni_example_xor PRIVATE ML)

foreach (example pipeline classification autograd)
    add_executable(clarni_example_${example} ${example}.cpp)
    target_link_libraries(clarni_example_${example} PRIVATE ML)
endforeach ()

