include_directories(
    ..
)

add_library(
    SelfOrganizingMapLib
    STATIC
    SelfOrganizingMap.cpp
    SOM.cpp
)

target_link_libraries(
    SelfOrganizingMapLib
    ImageProcessingLib
    UtilitiesLib
)