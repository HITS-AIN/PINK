include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PYTHON_INCLUDE_DIRS}
)

if(PINK_USE_CUDA)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
    )

    cuda_add_library(
        pink
        SHARED
        pink.cu
    )
    
    target_link_libraries(
        pink
        CudaLib
        pybind11::module
        UtilitiesLib
    )
else()
    pybind11_add_module(
        pink MODULE
        pink.cpp
    )
    
    target_link_libraries(
        pink PRIVATE
        UtilitiesLib
    )
endif()

set_target_properties(pink PROPERTIES PREFIX "")

install( 
    TARGETS pink
    LIBRARY DESTINATION lib
)
