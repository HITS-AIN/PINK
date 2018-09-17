include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

add_library(
    PINK
    SHARED    
    som.cpp
)
    
target_link_libraries(
    PINK
    ${CONAN_LIBS}
)

pybind11_add_module(pink MODULE som.cpp)
