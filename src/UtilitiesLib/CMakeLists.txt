include_directories(
    ..
)

add_library(
    UtilitiesLib
    STATIC
    CheckArrays.cpp
    get_file_header.cpp
    InputData.cpp
    Point.cpp
)
