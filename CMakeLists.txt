cmake_minimum_required(VERSION 3.0)
project(Pink VERSION 1.0 LANGUAGES C CXX)

message(STATUS "${PROJECT_NAME} version ${PROJECT_VERSION}")

# Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

# Set release as default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "release")
endif()

message(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(    NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release"
   AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo"
   AND NOT cmake_build_type_tolower STREQUAL "profiling")
    message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11")

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Download conan
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.11/conan.cmake"
                 "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

# Conan setup
include(${CMAKE_BINARY_DIR}/conan.cmake)
conan_cmake_run(CONANFILE conanfile.py
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Get Git revision
if(EXISTS ${PROJECT_SOURCE_DIR}/.git)
    find_package(Git)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            OUTPUT_VARIABLE "GIT_REVISION"
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "Git revision: ${GIT_REVISION}")
    else()
        set(GIT_REVISION 0)
    endif()
endif()

# Forward Pink version and Git revision into code
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

find_package(CUDA)
if(CUDA_FOUND)
    set(PINK_USE_CUDA true)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPINK_USE_CUDA")
endif()

add_subdirectory(src)

find_package(GTest)
if(GTEST_FOUND)
    ENABLE_TESTING()
    add_subdirectory(test)
endif()

find_package(Doxygen)
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
    ${PROJECT_SOURCE_DIR}/scripts/jpg2binaryOnlyOneChannel.py
    ${PROJECT_SOURCE_DIR}/scripts/numpy2binary.py
    ${PROJECT_SOURCE_DIR}/scripts/showHeatmap.py
    ${PROJECT_SOURCE_DIR}/scripts/showImages.py
    ${PROJECT_SOURCE_DIR}/scripts/showSOM.py
    DESTINATION scripts
)

if(UNIX)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_SOURCE_GENERATOR "TGZ;TBZ2")
    set(CPACK_GENERATOR "TGZ;TBZ2;DEB;RPM")
    set(CPACK_PACKAGE_MAINTAINER "Bernd Doser")
    set(CPACK_PACKAGE_VENDOR "HITS gGmbH - Astroinformatics")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Parallelized rotation and flipping INvariant Kohonen maps")
    set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
    set(CPACK_PACKAGE_CONTACT "Bernd Doser <bernd.doser@h-its.org>")
    include(CPack)
endif()
