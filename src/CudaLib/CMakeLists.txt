include_directories(
    ..
    ${CUDA_INCLUDE_DIRS}
)

list(APPEND CUDA_NVCC_FLAGS "-arch=compute_35 -code=compute_35,sm_35")

if(cmake_build_type_tolower STREQUAL "debug")
    list(APPEND CUDA_NVCC_FLAGS "-g -G")
endif()

cuda_add_library(
    CudaLib
    STATIC
    cuda_calculateEuclideanDistanceWithoutSquareRoot.cu
    cuda_trainSelfOrganizingMap.cu
    cuda_rotate.cu
    cuda_print_properties.cu
)

target_link_libraries(
    CudaLib
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)
