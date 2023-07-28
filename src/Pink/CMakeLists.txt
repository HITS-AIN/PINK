include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(
    Pink
    main.cpp
    main_cpu.cpp
)

target_link_libraries(
    Pink
    UtilitiesLib
)

if(PINK_USE_CUDA)
    target_link_libraries(
        Pink
        CudaLib
    )
endif()

if(NOT DEFINED SKBUILD)
    install(TARGETS Pink DESTINATION .)
endif()
