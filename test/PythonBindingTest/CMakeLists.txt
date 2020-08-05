include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

include_directories(SYSTEM
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    PythonBindingTest
    DynamicTrainer.cpp
    main.cpp
)
    
target_link_libraries(
    PythonBindingTest
    PythonBindingLib
    UtilitiesLib
    ${CONAN_LIBS}
)

add_test(
    NAME PythonBindingTest
    COMMAND PythonBindingTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/PythonBindingTest.xml
)
