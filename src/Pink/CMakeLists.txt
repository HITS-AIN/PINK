 
# Add C++11 support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
    ..
    ${PYTHON_INCLUDE_DIRS}
)

add_executable(
    Pink
    main.cpp
)

target_link_libraries(
    Pink
    CudaLib
    ${PYTHON_LIBRARIES}
)
