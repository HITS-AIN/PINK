include_directories(
    ..
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    UtilitiesTest
    main.cpp
    DistanceFunctorTest.cpp
    DistributionFunctorTest.cpp
)
    
target_link_libraries(
    UtilitiesTest
    UtilitiesLib
    ${GTEST_BOTH_LIBRARIES}
)

ADD_TEST(
    UtilitiesTest
    ${CMAKE_BINARY_DIR}/bin/UtilitiesTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/UtilitiesTest.xml
)
