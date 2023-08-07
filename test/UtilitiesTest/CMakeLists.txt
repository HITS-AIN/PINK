include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(
    UtilitiesTest
    main.cpp
    DimensionIOTest.cpp
    DistributionFunctorTest.cpp
    ipowTest.cpp
    ProgressBarTest.cpp
)

target_link_libraries(
    UtilitiesTest
    UtilitiesLib
    GTest::gtest
    GTest::gtest_main
)

add_test(
    NAME UtilitiesTest
    COMMAND UtilitiesTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/UtilitiesTest.xml
)
