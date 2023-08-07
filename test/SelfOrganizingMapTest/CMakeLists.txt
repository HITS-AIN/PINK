include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(
    SelfOrganizingMapTest
    add_binary_section.cpp
    Cartesian.cpp
    circular_ed.cpp
    Data.cpp
    DataIterator.cpp
    DataIteratorShuffled.cpp
    euclidean_distance.cpp
    generate_rotated_images.cpp
    Hexagonal.cpp
    main.cpp
    Mapper.cpp
    Trainer.cpp
)

target_link_libraries(
    SelfOrganizingMapTest
    UtilitiesLib
    GTest::gtest
    GTest::gtest_main
)

add_test(
    NAME SelfOrganizingMapTest
    COMMAND SelfOrganizingMapTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/SelfOrganizingMapTest.xml
)
