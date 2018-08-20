include_directories(
    ..
)

add_library(
    ImageProcessingLib
    STATIC
    Image.cpp
    ImageProcessing.cpp
)

target_link_libraries(
    ImageProcessingLib
    UtilitiesLib
)
