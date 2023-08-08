add_subdirectory(ImageProcessingTest)
add_subdirectory(SelfOrganizingMapTest)
add_subdirectory(UtilitiesTest)

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(CudaTest)
endif()

if(pybind11_FOUND)
    add_subdirectory(PythonBindingTest)
endif()