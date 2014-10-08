include_directories(
    ..
)

add_library(
    ImageProcessingLib
    SHARED
    rotate.c
)

target_link_libraries(
    ImageProcessingLib
)
