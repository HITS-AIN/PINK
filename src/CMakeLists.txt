add_subdirectory(Pink)
add_subdirectory(UtilitiesLib)

if(CUDA_FOUND)
    add_subdirectory(CudaLib)
endif()

if(pybind11_FOUND)
    add_subdirectory(pink)
endif()