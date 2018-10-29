include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    ${PYTHON_INCLUDE_DIRS}
)

pybind11_add_module(
    pink MODULE
    pink.cpp
)

target_link_libraries(
    pink PRIVATE
    ImageProcessingLib
)

if(PINK_USE_CUDA)
    target_link_libraries(
        pink PRIVATE
        CudaLib
    )
endif()
