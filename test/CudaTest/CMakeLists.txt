include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIR}
)

cuda_add_executable(
    CudaTest
    main.cpp
    mixed_precision.cpp
    #EuclideanDistanceTest.cpp
    RotationTest.cu
    rotate_90_degrees_list.cu
    TrainerCompare.cu
    update_neurons.cu
    #UpdateNeuronsTest.cpp
)
    
target_link_libraries(
    CudaTest
    CudaLib
    ImageProcessingLib
    SelfOrganizingMapLib
    ${GTEST_BOTH_LIBRARIES}
)

add_test(
    CudaTest
    ${CMAKE_BINARY_DIR}/bin/CudaTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/CudaTest.xml
)
