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

set_target_properties(UtilitiesLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
