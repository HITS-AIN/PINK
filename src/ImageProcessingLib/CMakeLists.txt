include_directories(
    ..
    ${GTEST_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
)

add_library(
    ImageProcessingLib
    SHARED
    ImageProcessing.c
)

target_link_libraries(
    ImageProcessingLib
    ${GTEST_BOTH_LIBRARIES}
    ${PYTHON_LIBRARIES}
)
