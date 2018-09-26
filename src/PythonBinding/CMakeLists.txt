include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

pybind11_add_module(
    pink MODULE
    pink.cpp
)

target_link_libraries(
    pink PRIVATE
    ImageProcessingLib
    SelfOrganizingMapLib
)
