project(Pink C CXX)

set(PINK_VERSION_MAJOR 0)
set(PINK_VERSION_MINOR 8)

cmake_minimum_required(VERSION 2.6.2)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

# guard against bad build-type strings
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(    NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release"
   AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo"
   AND NOT cmake_build_type_tolower STREQUAL "profiling")
    message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG_MODE")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG_MODE")
set(CMAKE_CXX_FLAGS_PROFILING "-O3 -g -pg -DDEBUG_MODE")

# Add openmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

configure_file (
  "${PROJECT_SOURCE_DIR}/bashrc.in"
  "${PROJECT_SOURCE_DIR}/bashrc"
)

configure_file (
  "${PROJECT_SOURCE_DIR}/src/UtilitiesLib/Version.h.in"
  "${PROJECT_SOURCE_DIR}/src/UtilitiesLib/Version.h"
)

# set RPATH for build and installed executable
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

set(CMAKE_VERBOSE_MAKEFILE false)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

FIND_PACKAGE(PythonLibs)
if(PYTHONLIBS_FOUND)
    set(PINK_USE_PYTHON true)
endif()

find_package(CUDA)
if(CUDA_FOUND)
    set(PINK_USE_CUDA true)
    add_subdirectory(src/CudaLib)
endif()

find_package(GTest)
if(GTEST_FOUND)
    ENABLE_TESTING()
    add_subdirectory(src/ImageProcessingTest)
    add_subdirectory(src/UtilitiesTest)
endif()

if(GTEST_FOUND AND CUDA_FOUND)
    add_subdirectory(src/CudaTest)    
endif()

add_subdirectory(src/ImageProcessingLib)
add_subdirectory(src/Pink)
add_subdirectory(src/SelfOrganizingMapLib)
add_subdirectory(src/UtilitiesLib)

FIND_PACKAGE(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(
    	doc
    	${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/doxygen/Doxyfile
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/doxygen
            COMMENT "Generating API documentation with Doxygen" VERBATIM
    )    
endif()

install(FILES
    ${PROJECT_SOURCE_DIR}/scripts/jpg2binary.py
    ${PROJECT_SOURCE_DIR}/scripts/numpy2binary.py
    ${PROJECT_SOURCE_DIR}/scripts/show.py
    DESTINATION scripts
)

install(FILES
    ${PROJECT_SOURCE_DIR}/bashrc
    DESTINATION .
)
