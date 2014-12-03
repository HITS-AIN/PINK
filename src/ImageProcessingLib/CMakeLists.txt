# Add C++11 support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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
