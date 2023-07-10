include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

include_directories(SYSTEM
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(
    CudaLib
    SHARED
    CudaLib.cu
    dot_dp4a.cu
    main_gpu.cu
)

target_link_libraries(
    CudaLib
    UtilitiesLib
    ${CUDA_LIBRARIES}
)

install(TARGETS CudaLib DESTINATION .)
