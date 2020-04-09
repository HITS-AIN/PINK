include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

include_directories(SYSTEM
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    SelfOrganizingMapTest
    add_binary_section.cpp
    Cartesian.cpp
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
    ${GTEST_BOTH_LIBRARIES}
)

add_test(
    NAME SelfOrganizingMapTest
    COMMAND SelfOrganizingMapTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/SelfOrganizingMapTest.xml
)
