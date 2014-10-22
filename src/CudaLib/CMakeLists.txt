include_directories(
    ..
    ${CUDA_INCLUDE_DIRS}
)

list(APPEND CUDA_NVCC_FLAGS "-arch=compute_35 -code=compute_35,sm_35")

cuda_add_library(
    CudaLib
    STATIC
    cuda_rotate.cu
    cuda_print_properties.cu
)

target_link_libraries(
    CudaLib
    ${CUDA_LIBRARIES}
)
