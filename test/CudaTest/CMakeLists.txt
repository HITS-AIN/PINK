include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

include_directories(SYSTEM 
    ${CUDA_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIR}
)

cuda_add_executable(
    CudaTest
    main.cpp
    compare_trainer_cpu.cu
    compare_trainer_mixed.cu
    mapper.cu
    mixed_precision.cu
    resize.cu
    rotate_90_degrees_list.cu
    update_neurons.cu
)
    
target_link_libraries(
    CudaTest
    CudaLib
    ${GTEST_BOTH_LIBRARIES}
)

add_test(
    CudaTest
    ${CMAKE_BINARY_DIR}/bin/CudaTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/CudaTest.xml
)
