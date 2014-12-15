 
# Add C++11 support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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
