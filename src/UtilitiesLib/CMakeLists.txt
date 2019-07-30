include_directories(
    ..
)

add_library(
    UtilitiesLib
    STATIC
    get_file_header.cpp
    InputData.cpp
    Point.cpp
)
