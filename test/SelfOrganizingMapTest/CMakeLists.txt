include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    SelfOrganizingMapTest
    main.cpp
    Cartesian.cpp
    Trainer.cpp
)
    
target_link_libraries(
    SelfOrganizingMapTest
    SelfOrganizingMapLib
    ${GTEST_BOTH_LIBRARIES}
)

ADD_TEST(
    SelfOrganizingMapTest
    ${CMAKE_BINARY_DIR}/bin/SelfOrganizingMapTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/SelfOrganizingMapTest.xml
)
