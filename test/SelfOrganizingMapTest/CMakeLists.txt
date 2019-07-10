include_directories(
    ${GTEST_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(
    SelfOrganizingMapTest
    main.cpp
    Cartesian.cpp
    Data.cpp
    DataIterator.cpp
    Hexagonal.cpp
    Mapper.cpp
    Trainer.cpp
)
    
target_link_libraries(
    SelfOrganizingMapTest
    UtilitiesLib
    ${GTEST_BOTH_LIBRARIES}
)

ADD_TEST(
    SelfOrganizingMapTest
    ${CMAKE_BINARY_DIR}/bin/SelfOrganizingMapTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/SelfOrganizingMapTest.xml
)
