
# No propagation of host compiler flags to CUDA compiler via -Xcompiler 
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Add C++11 support for host compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Set flags for CUDA compiler
#list(APPEND CUDA_NVCC_FLAGS "-arch=compute_35 -code=compute_35,sm_35")
list(APPEND CUDA_NVCC_FLAGS "-arch=compute_61 -code=compute_61,sm_61")
#list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_61 -code=sm_61")
list(APPEND CUDA_NVCC_FLAGS_RELEASE "-O3")
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-O0 -g -G -DDEBUG_MODE")
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBINFO "-O3 -g -G -DDEBUG_MODE")
list(APPEND CUDA_NVCC_FLAGS_PROFILING "-O3 -g -G -pg -DDEBUG_MODE")

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
    cuda_generateRotatedImages.cu
    cuda_mapping.cpp
    cuda_trainSelfOrganizingMap.cpp
    cuda_rotate.cu
    cuda_print_properties.cu
    cuda_updateNeurons.cu
    euclidean_distance.cu
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
