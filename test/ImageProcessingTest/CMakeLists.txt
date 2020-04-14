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

add_test(
    NAME ImageProcessingTest
    COMMAND ImageProcessingTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/ImageProcessingTest.xml
)
