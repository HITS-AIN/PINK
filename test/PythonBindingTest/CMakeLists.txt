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
    ${GTEST_BOTH_LIBRARIES}
)

ADD_TEST(
    PythonBindingTest
    ${CMAKE_BINARY_DIR}/test/PythonBindingTest/PythonBindingTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/PythonBindingTest.xml
)
