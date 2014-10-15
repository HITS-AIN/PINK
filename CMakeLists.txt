project(SelfOrganizingMap C CXX)

cmake_minimum_required(VERSION 2.6.2)

set(CMAKE_MODULE_PATH ${SecurityGuard_SOURCE_DIR}/cmake/modules)
set(SelfOrganizingMap VERSION "0.1.1")

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

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -fopenmp -DBOOST_DATE_TIME_POSIX_TIME_STD_CONFIG -DSECURITYGUARD_VERSION=\"${SecurityGuard_VERSION}\"")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DSECURITYGUARD_DEBUG_MODE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

if( ${CMAKE_CXX_COMPILER} MATCHES "icpc" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__GXX_EXPERIMENTAL_CXX0X__ -DNO_INITIALIZER_LIST_SUPPORTED")
endif()

if( ${CMAKE_COMPILER_IS_GNUCXX} )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=10000 -msse2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -funroll-loops -ffast-math -mfpmath=sse -march=core2 -mtune=core2 -DBLASBOOSTER_DEBUG_MODE")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops --param max-unroll-times=100 -ffast-math -mfpmath=sse -march=core2 -mtune=core2")
endif()

set(CMAKE_VERBOSE_MAKEFILE false)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
    
FIND_PACKAGE(Boost 1.55 COMPONENTS chrono date_time filesystem serialization system timer REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    add_definitions(-DUSE_BOOST -DHAVE_SHARED_POINTER)
endif(Boost_FOUND)

FIND_PACKAGE(PythonLibs)

FIND_PACKAGE(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(
        ${PROJECT_SOURCE_DIR}/doxygen/Doxyfile
        ${PROJECT_BINARY_DIR}/doxygen/Doxyfile
    )
    add_custom_target(
    	doc
    	${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doxygen/Doxyfile
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/doxygen
            COMMENT "Generating API documentation with Doxygen" VERBATIM
    )    
endif(DOXYGEN_FOUND)

add_subdirectory(src/ImageProcessingLib)

find_package(GTest)
if(GTEST_FOUND)
    ENABLE_TESTING()
    add_subdirectory(src/ImageProcessingTest)
endif(GTEST_FOUND)
