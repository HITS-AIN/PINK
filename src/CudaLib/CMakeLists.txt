include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

add_library(
    CudaLib
    SHARED
    CudaLib.cu
    dot_dp4a.cu
    main_gpu.cu
)

target_link_libraries(
    CudaLib
    UtilitiesLib
)

install(TARGETS CudaLib DESTINATION pink)
