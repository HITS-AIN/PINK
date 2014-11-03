 
# Add C++11 support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(PINK_USE_CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPINK_USE_CUDA")
endif(PINK_USE_CUDA)

include_directories(
    ..
)

add_executable(
    Pink
    main.cpp
    trainSelfOrganizingMap.cpp
    SelfOrganizingMap.cpp
)

target_link_libraries(
    Pink
    CudaLib
    ImageProcessingLib
    UtilitiesLib
)

install( 
    TARGETS Pink
    RUNTIME DESTINATION bin
)
