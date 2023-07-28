include_directories(
    ${PROJECT_SOURCE_DIR}/src
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
    GTest::gtest
    GTest::gtest_main
)

add_test(
    NAME PythonBindingTest
    COMMAND PythonBindingTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/PythonBindingTest.xml
)
