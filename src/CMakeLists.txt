add_subdirectory(Pink)
add_subdirectory(UtilitiesLib)

if(CUDA_FOUND)
    add_subdirectory(CudaLib)
endif()