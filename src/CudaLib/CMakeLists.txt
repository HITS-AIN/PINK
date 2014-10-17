include_directories(
    ..
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(
    CudaLib
    STATIC
    cuda_rotate.cu
)

target_link_libraries(
    CudaLib
    ${CUDA_LIBRARIES}
)
