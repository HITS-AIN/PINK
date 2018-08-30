add_subdirectory(ImageProcessingTest)
add_subdirectory(SelfOrganizingMapTest)
add_subdirectory(UtilitiesTest)

if(CUDA_FOUND)
    add_subdirectory(CudaTest)
endif()