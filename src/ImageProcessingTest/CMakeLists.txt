include_directories(
    ..
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    ImageProcessingTest
    main.cpp
    ImageProcessingTest.cpp
)
    
target_link_libraries(
    ImageProcessingTest
    ImageProcessingLib
    ${GTEST_BOTH_LIBRARIES}
)

ADD_TEST(ImageProcessingTest ${CMAKE_BINARY_DIR}/bin/ImageProcessingTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/ImageProcessingTest.xml)
