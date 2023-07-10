if(NOT DEFINED SKBUILD)
    add_subdirectory(Pink)
endif()

add_subdirectory(UtilitiesLib)

if(CUDA_FOUND)
    add_subdirectory(CudaLib)
endif()

if(pybind11_FOUND)
    add_subdirectory(pink)
endif()