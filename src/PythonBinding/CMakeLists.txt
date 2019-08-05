include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PYTHON_INCLUDE_DIRS}
)

pybind11_add_module(
    pink MODULE
    pink.cpp
    DynamicData.cpp
    DynamicSOM.cpp
)

target_link_libraries(
    pink PRIVATE
    UtilitiesLib
)

set_target_properties(pink PROPERTIES PREFIX "")

install( 
    TARGETS pink
    LIBRARY DESTINATION lib
)
