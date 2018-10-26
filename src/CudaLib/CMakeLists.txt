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
    dot_dp4a.cu
    main_gpu.cu
    trigonometricValues.cpp
)

target_link_libraries(
    CudaLib
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

install( 
    TARGETS CudaLib
    LIBRARY DESTINATION lib
)
