include_directories(
    ..
)

add_executable(
    Pink
    main.cpp
    main_cpu.cpp
)

target_link_libraries(
    Pink
    SelfOrganizingMapLib
    UtilitiesLib
)

if(PINK_USE_CUDA)
    target_link_libraries(
        Pink
        CudaLib
    )
endif()

install( 
    TARGETS Pink
    RUNTIME DESTINATION bin
)
