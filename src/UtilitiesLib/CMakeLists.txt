include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

add_library(
    UtilitiesLib
    STATIC
    DimensionIO.cpp
    get_file_header.cpp
    InputData.cpp
    Point.cpp
)
