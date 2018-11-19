include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PYTHON_INCLUDE_DIRS}
)

if(PINK_USE_CUDA)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
    )
endif()

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
