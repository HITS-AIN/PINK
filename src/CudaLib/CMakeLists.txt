
# No propagation of host compiler flags to CUDA compiler via -Xcompiler 
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Add C++11 support for host compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Set flags for CUDA compiler
list(APPEND CUDA_NVCC_FLAGS "-arch=compute_35 -code=compute_35,sm_35")
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
    cuda_generateEuclideanDistanceMatrix_algo2.cu
    cuda_generateEuclideanDistanceMatrix_algo2_firstStep.cu
    cuda_generateEuclideanDistanceMatrix_algo2_secondStep.cu
    cuda_generateEuclideanDistanceMatrix.cu
    cuda_generateEuclideanDistanceMatrix_secondStep.cu
    cuda_generateRotatedImages.cu
    cuda_trainSelfOrganizingMap.cpp
    cuda_rotate.cu
    cuda_print_properties.cu
    rotateAndCrop.cu
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
