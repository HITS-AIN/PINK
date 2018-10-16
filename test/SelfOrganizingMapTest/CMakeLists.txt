include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
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
    ${CUDA_LIBRARIES}
)

ADD_TEST(
    SelfOrganizingMapTest
    ${CMAKE_BINARY_DIR}/bin/SelfOrganizingMapTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/SelfOrganizingMapTest.xml
)
