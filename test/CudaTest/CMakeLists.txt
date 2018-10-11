include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    CudaTest
    main.cpp
    mixed_precision.cpp
    EuclideanDistanceTest.cpp
    RotationTest.cpp
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
