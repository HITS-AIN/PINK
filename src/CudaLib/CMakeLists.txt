include_directories(
    ..
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(
    CudaLib
    SHARED
    cuda_functions.cu
    cuda_rotate.cu
    cuda_print_properties.cu
    euclidean_distance_kernel.cu
    dot_dp4a.cu
    main_gpu.cu
)

target_link_libraries(
    CudaLib
    SelfOrganizingMapLib
    UtilitiesLib
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

install( 
    TARGETS CudaLib
    LIBRARY DESTINATION lib
)
