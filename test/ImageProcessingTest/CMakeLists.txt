include_directories(
    ${PROJECT_SOURCE_DIR}/src
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

file(COPY ${PROJECT_SOURCE_DIR}/test/ImageProcessingTest
     DESTINATION ${PROJECT_BINARY_DIR}/test)
