include_directories(
    ..
)

add_library(
    SelfOrganizingMapLib
    STATIC
    SOM.cpp
)

target_link_libraries(
    SelfOrganizingMapLib
    UtilitiesLib
)