include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

include_directories(SYSTEM
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    ImageProcessingTest
    resize.cpp
    main.cpp
    rotate.cpp
)
    
target_link_libraries(
    ImageProcessingTest
    ${GTEST_BOTH_LIBRARIES}
)

ADD_TEST(
    ImageProcessingTest
    ${CMAKE_BINARY_DIR}/test/ImageProcessingTest/ImageProcessingTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/ImageProcessingTest.xml
)
