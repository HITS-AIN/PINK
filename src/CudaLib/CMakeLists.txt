include_directories(
    ..
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(
    CudaLib
    SHARED
    cuda_calculateEuclideanDistanceWithoutSquareRoot.cu
    cuda_functions.cu
    cuda_generateEuclideanDistanceMatrix.cpp
    cuda_generateEuclideanDistanceMatrix_firstStep.cu
    cuda_generateEuclideanDistanceMatrix_firstStep_multiGPU.cu
    cuda_generateEuclideanDistanceMatrix_firstStep_opt.cu
    cuda_generateEuclideanDistanceMatrix_firstStep_opt2.cu
    cuda_generateEuclideanDistanceMatrix_firstStep_opt3.cu
    cuda_generateEuclideanDistanceMatrix_secondStep.cu
    cuda_rotate.cu
    cuda_print_properties.cu
    dot_dp4a.cu
    generate_rotated_images_gpu.cu
    main_gpu.cu
    trigonometricValues.cpp
    update_neurons.cu
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
