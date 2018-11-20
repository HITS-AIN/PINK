include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PYTHON_INCLUDE_DIRS}
)

if(PINK_USE_CUDA)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
    )
endif()

cuda_add_library(
    pink
    SHARED
    pink.cu
)

target_link_libraries(
    pink
    ImageProcessingLib
    pybind11::module
    SelfOrganizingMapLib
    UtilitiesLib
)

if(PINK_USE_CUDA)
    target_link_libraries(
        pink
        CudaLib
    )
endif()

set_target_properties(pink PROPERTIES PREFIX "")

install( 
    TARGETS pink
    LIBRARY DESTINATION lib
)
