include_directories(
    ..
    ${CUDA_INCLUDE_DIRS}
)

add_library(
    SelfOrganizingMapLib
    STATIC
    SelfOrganizingMap.cpp
)

target_link_libraries(
    SelfOrganizingMapLib
    ImageProcessingLib
    UtilitiesLib
)
