include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(
    ImageProcessingTest
    resize.cpp
    main.cpp
    rotate.cpp
)

target_link_libraries(
    ImageProcessingTest
    GTest::gtest
    GTest::gtest_main
)

add_test(
    NAME ImageProcessingTest
    COMMAND ImageProcessingTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/ImageProcessingTest.xml
)
