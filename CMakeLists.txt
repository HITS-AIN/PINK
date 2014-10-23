project(SelfOrganizingMap C CXX)
set(SelfOrganizingMap VERSION "0.1.1")

cmake_minimum_required(VERSION 2.6.2)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# guard against bad build-type strings
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(    NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release"
   AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
    message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

# Add openmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_VERBOSE_MAKEFILE true)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

FIND_PACKAGE(PythonLibs)

find_package(CUDA)
if(CUDA_FOUND)
    set(PINK_USE_CUDA true)
    add_subdirectory(src/CudaLib)
endif(CUDA_FOUND)

find_package(GTest)
if(GTEST_FOUND)
    ENABLE_TESTING()
    add_subdirectory(src/CudaTest)
    add_subdirectory(src/ImageProcessingTest)
endif(GTEST_FOUND)

add_subdirectory(src/ImageProcessingLib)
add_subdirectory(src/Pink)

FIND_PACKAGE(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(
    	doc
    	${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doxygen/Doxyfile
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/doxygen
            COMMENT "Generating API documentation with Doxygen" VERBATIM
    )    
endif(DOXYGEN_FOUND)
