include_directories(
    ..
)

add_library(
    ImageProcessingLib
    SHARED
    ImageProcessing.c
)

target_link_libraries(
    ImageProcessingLib
)
