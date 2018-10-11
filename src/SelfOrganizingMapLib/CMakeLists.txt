include_directories(
    ..
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
