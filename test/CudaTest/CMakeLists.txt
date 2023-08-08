include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(
    CudaTest
    circular_ed.cu
    compare_trainer_3d_cpu.cu
    compare_trainer_cpu.cu
    compare_trainer_mixed.cu
    main.cpp
    mapper.cu
    mixed_precision.cu
    resize.cu
    rotate_90_degrees_list.cu
    update_neurons.cu
)

target_link_libraries(
    CudaTest
    CudaLib
    GTest::gtest
    GTest::gtest_main
)

add_test(
    NAME CudaTest
    COMMAND CudaTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/CudaTest.xml
)
