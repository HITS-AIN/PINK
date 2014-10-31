
# Add C++11 support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
    ..
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    ImageProcessingTest
    main.cpp
    ImageTest.cpp
    ImageProcessingTest.cpp
)
    
target_link_libraries(
    ImageProcessingTest
    ImageProcessingLib
    ${GTEST_BOTH_LIBRARIES}
)

ADD_TEST(
    ImageProcessingTest
    ${CMAKE_BINARY_DIR}/bin/ImageProcessingTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/ImageProcessingTest.xml
)
