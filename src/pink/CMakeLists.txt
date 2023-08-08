if (APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

include_directories(SYSTEM
    ${PYTHON_INCLUDE_DIRS}
)

add_library(
    PythonBindingLib
    SHARED
    DynamicData.cpp
    DynamicMapper.cu
    DynamicSOM.cpp
    DynamicTrainer.cu
)

target_link_libraries(
    PythonBindingLib PRIVATE
    UtilitiesLib
)

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(
        PythonBindingLib PRIVATE
        CudaLib
    )
endif()

pybind11_add_module(
    _pink MODULE
    pink.cpp
)

target_link_libraries(
    _pink PRIVATE
    PythonBindingLib
    UtilitiesLib
)

set_target_properties(_pink PROPERTIES PREFIX "")

install(TARGETS _pink PythonBindingLib DESTINATION pink)
