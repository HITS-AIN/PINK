include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

add_library(
    UtilitiesLib
    STATIC
    DimensionIO.cpp
    DistributionFunctor.cpp
    get_file_header.cpp
    InputData.cpp
)
