include_directories(
    ..
)

add_executable(
    Pink
    main.cpp
)

target_link_libraries(
    Pink
    ImageProcessingLib
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

add_executable(
    Pink2
    main2.cpp
)

target_link_libraries(
    Pink2
    ImageProcessingLib
    SelfOrganizingMapLib
    UtilitiesLib
)

if(PINK_USE_CUDA)
    target_link_libraries(
        Pink2
        CudaLib
    )
endif()

install( 
    TARGETS Pink2
    RUNTIME DESTINATION bin
)
