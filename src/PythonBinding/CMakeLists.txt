include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

include_directories(SYSTEM
    ${PYTHON_INCLUDE_DIRS}
)

if(PINK_USE_CUDA)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
    )

    cuda_add_library(
        PythonBindingLib
        SHARED
        DynamicData.cpp
        DynamicMapper.cu
        DynamicSOM.cpp
        DynamicTrainer.cu
    )
    
    target_link_libraries(
        PythonBindingLib
        CudaLib
        UtilitiesLib
    )
else()
    add_library(
        PythonBindingLib
        SHARED
        DynamicData.cpp
        DynamicSOM.cpp
    )
    
    target_link_libraries(
        PythonBindingLib PRIVATE
        UtilitiesLib
    )
endif()

pybind11_add_module(
    pink MODULE
    pink.cpp
)

target_link_libraries(
    pink PRIVATE
    PythonBindingLib
    UtilitiesLib
)

set_target_properties(pink PROPERTIES PREFIX "")

install( 
    TARGETS pink
    LIBRARY DESTINATION lib
)
