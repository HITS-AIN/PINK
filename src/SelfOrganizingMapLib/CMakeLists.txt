include_directories(
    ..
)

add_library(
    SelfOrganizingMapLib
    STATIC
    mapping.cpp
    SelfOrganizingMap.cpp
    SOM.cpp
    training.cpp
)

target_link_libraries(
    SelfOrganizingMapLib
    ImageProcessingLib
    UtilitiesLib
)
