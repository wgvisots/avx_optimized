##############################################################################
# Copyright 2024 - 2026 W. Greg Visotski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

cmake_minimum_required(VERSION 3.21)

project(avx_optimized
        VERSION 0.2.0
        DESCRIPTION "AVX2/AVX-512 optimized signal-processing primitives with runtime dispatch"
        HOMEPAGE_URL "https://github.com/wgvisots/avx_optimized"
        LANGUAGES CXX)

include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(GenerateExportHeader)

option(AVX_OPTIMIZED_BUILD_TESTS "Build the unit tests" ${PROJECT_IS_TOP_LEVEL})
option(AVX_OPTIMIZED_BUILD_BENCHMARKS "Build the benchmark program" OFF)
option(AVX_OPTIMIZED_BUILD_DOCS "Add an 'avx_optimized_docs' target that builds the API reference with Doxygen" OFF)
option(AVX_OPTIMIZED_INSTALL "Generate install and package rules" ${PROJECT_IS_TOP_LEVEL})
option(AVX_OPTIMIZED_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(AVX_OPTIMIZED_FETCH_DEPENDENCIES
       "Download dependencies that find_package() cannot find (mathfun, GoogleTest)" ON)
option(AVX_OPTIMIZED_OPTIMIZE_KERNELS
       "Always compile the AVX2/AVX-512 kernels with optimization, even in Debug builds (see src/kernels.h)" ON)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "avx_optimized currently supports GCC and Clang only (found ${CMAKE_CXX_COMPILER_ID}).")
endif()

# The AVX2/AVX-512 kernels are only built for x86-64; elsewhere the portable scalar kernels are used.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    set(_is_x86_64 ON)
else()
    set(_is_x86_64 OFF)
endif()
cmake_dependent_option(AVX_OPTIMIZED_X86_KERNELS "Build the AVX2 and AVX-512 kernels" ON "_is_x86_64" OFF)
set(AVX_OPTIMIZED_X86 ${AVX_OPTIMIZED_X86_KERNELS})
if(NOT AVX_OPTIMIZED_X86)
    message(STATUS "avx_optimized: building the portable scalar kernels only")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AvxOptimizedDependencies)

# -----------------------------------------------------------------------------
# Library
# -----------------------------------------------------------------------------

add_library(avx_optimized
            src/complex_fir_filter.cpp
            src/cpu.cpp
            src/kernels_scalar.cpp
            src/vectorized.cpp
            src/version.cpp)
add_library(avx_optimized::avx_optimized ALIAS avx_optimized)

configure_file(include/avx_optimized/version.h.in
               "${CMAKE_CURRENT_BINARY_DIR}/include/avx_optimized/version.h" @ONLY)
generate_export_header(avx_optimized
                       BASE_NAME AVX_OPTIMIZED
                       EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/avx_optimized/export.h")

get_target_property(_avx_optimized_type avx_optimized TYPE)
if(_avx_optimized_type STREQUAL "STATIC_LIBRARY")
    target_compile_definitions(avx_optimized PUBLIC AVX_OPTIMIZED_STATIC_DEFINE)
endif()

target_include_directories(avx_optimized
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        src)

# C++20 for the library and everything that uses it (public, so consumers inherit it).
target_compile_features(avx_optimized PUBLIC cxx_std_20)

if(PROJECT_VERSION_MAJOR EQUAL 0)
    set(_soversion "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}") # 0.x: every minor release may break ABI
else()
    set(_soversion "${PROJECT_VERSION_MAJOR}")
endif()

set_target_properties(avx_optimized PROPERTIES
                      CXX_EXTENSIONS OFF
                      CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON
                      POSITION_INDEPENDENT_CODE ON
                      VERSION ${PROJECT_VERSION}
                      SOVERSION ${_soversion}
                      EXPORT_NAME avx_optimized)

if(PROJECT_IS_TOP_LEVEL)
    set(AVX_OPTIMIZED_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
        $<$<BOOL:${AVX_OPTIMIZED_WARNINGS_AS_ERRORS}>:-Werror>)
    target_compile_options(avx_optimized PRIVATE ${AVX_OPTIMIZED_WARNING_FLAGS})
endif()

# Per-instruction-set kernels.  Only these files are compiled with the wider instruction sets;
# everything else, including all public headers, stays at the baseline, so the library loads and
# runs on any x86-64 CPU.  Each is an object library so its flags stay isolated and its object
# file can be inspected by the symbol check in test/.
if(AVX_OPTIMIZED_X86)
    target_compile_definitions(avx_optimized PRIVATE AVX_OPTIMIZED_HAVE_X86_KERNELS=1)

    set(AVX_OPTIMIZED_ISA_KERNEL_TARGETS "")
    foreach(_isa avx2 avx512)
        set(_target avx_optimized_kernels_${_isa})
        add_library(${_target} OBJECT src/kernels_${_isa}.cpp)
        list(APPEND AVX_OPTIMIZED_ISA_KERNEL_TARGETS ${_target})
        # mathfun is header-only and private, so consumers of the installed library never need it.
        target_link_libraries(${_target} PRIVATE mathfun::mathfun)
        target_include_directories(${_target} PRIVATE
                                   $<TARGET_PROPERTY:avx_optimized,INCLUDE_DIRECTORIES>)
        target_compile_definitions(${_target} PRIVATE
                                   $<TARGET_PROPERTY:avx_optimized,COMPILE_DEFINITIONS>)
        target_compile_features(${_target} PRIVATE cxx_std_20)
        set_target_properties(${_target} PROPERTIES
                              CXX_EXTENSIONS OFF
                              CXX_VISIBILITY_PRESET hidden
                              VISIBILITY_INLINES_HIDDEN ON
                              POSITION_INDEPENDENT_CODE ON)
        if(PROJECT_IS_TOP_LEVEL)
            target_compile_options(${_target} PRIVATE ${AVX_OPTIMIZED_WARNING_FLAGS})
        endif()
        if(AVX_OPTIMIZED_OPTIMIZE_KERNELS)
            # Unoptimized code can emit std:: inline helpers out-of-line from these files, and the
            # linker may then share them with baseline code (see the rules in src/kernels.h).
            # Optimizing these files avoids that, and also keeps Debug test runs fast.
            target_compile_options(${_target} PRIVATE $<$<CONFIG:Debug>:-O2>)
        endif()
        # Added as objects rather than linked, so the object libraries never appear in the
        # exported link interface.
        target_sources(avx_optimized PRIVATE $<TARGET_OBJECTS:${_target}>)
    endforeach()
    target_compile_options(avx_optimized_kernels_avx2 PRIVATE -mavx2 -mfma)
    target_compile_options(avx_optimized_kernels_avx512 PRIVATE
                           -mavx2 -mfma -mavx512f -mavx512bw -mavx512dq -mavx512vl)
endif()

# -----------------------------------------------------------------------------
# Tests and benchmarks
# -----------------------------------------------------------------------------

if(AVX_OPTIMIZED_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(AVX_OPTIMIZED_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

# -----------------------------------------------------------------------------
# API reference (cmake --build <dir> --target avx_optimized_docs; output in <dir>/docs/html)
# -----------------------------------------------------------------------------

if(AVX_OPTIMIZED_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    set(DOXYGEN_PROJECT_NAME "avx_optimized")
    set(DOXYGEN_PROJECT_NUMBER ${PROJECT_VERSION})
    set(DOXYGEN_PROJECT_BRIEF ${PROJECT_DESCRIPTION})
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    set(DOXYGEN_STRIP_FROM_INC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/include")
    set(DOXYGEN_STRIP_FROM_PATH "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
    set(DOXYGEN_EXCLUDE_PATTERNS "*/export.h")
    set(DOXYGEN_EXCLUDE_SYMBOLS "avx_optimized::detail")
    set(DOXYGEN_MACRO_EXPANSION YES)
    set(DOXYGEN_PREDEFINED "AVX_OPTIMIZED_EXPORT=")
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
    set(DOXYGEN_WARN_AS_ERROR FAIL_ON_WARNINGS) # every public declaration must be documented
    set(DOXYGEN_MARKDOWN_ID_STYLE GITHUB)       # so the README's #section links resolve
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_HAVE_DOT NO)
    doxygen_add_docs(avx_optimized_docs
                     "${CMAKE_CURRENT_SOURCE_DIR}/include/avx_optimized"
                     "${CMAKE_CURRENT_BINARY_DIR}/include/avx_optimized/version.h"
                     "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
                     "${CMAKE_CURRENT_SOURCE_DIR}/docs/MIGRATION.md"
                     "${CMAKE_CURRENT_SOURCE_DIR}/CONTRIBUTING.md"
                     COMMENT "Building the avx_optimized API reference")
endif()

# -----------------------------------------------------------------------------
# Install and packaging
# -----------------------------------------------------------------------------

if(AVX_OPTIMIZED_INSTALL)
    set(_config_dir "${CMAKE_INSTALL_LIBDIR}/cmake/avx_optimized")

    install(TARGETS avx_optimized
            EXPORT avx_optimizedTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/avx_optimized
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/avx_optimized/export.h"
                  "${CMAKE_CURRENT_BINARY_DIR}/include/avx_optimized/version.h"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/avx_optimized)
    install(EXPORT avx_optimizedTargets
            NAMESPACE avx_optimized::
            DESTINATION ${_config_dir})

    configure_package_config_file(cmake/avx_optimizedConfig.cmake.in
                                  "${CMAKE_CURRENT_BINARY_DIR}/avx_optimizedConfig.cmake"
                                  INSTALL_DESTINATION ${_config_dir})
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/avx_optimizedConfigVersion.cmake"
                                     COMPATIBILITY SameMinorVersion)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/avx_optimizedConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/avx_optimizedConfigVersion.cmake"
            DESTINATION ${_config_dir})
    install(FILES LICENSE README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})

    set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
    set(CPACK_PACKAGE_HOMEPAGE_URL ${PROJECT_HOMEPAGE_URL})
    set(CPACK_PACKAGE_CONTACT "W. Greg Visotski")
    set(CPACK_PACKAGE_VENDOR "NMERC")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    if(NOT DEFINED CPACK_PACKAGE_RELEASE)
        set(CPACK_PACKAGE_RELEASE 1)
    endif()
    set(CPACK_PACKAGE_FILE_NAME
        "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
    set(CPACK_DEBIAN_PACKAGE_NAME "avx-optimized") # Debian package names may not contain '_'
    set(CPACK_DEBIAN_PACKAGE_SECTION "libdevel")
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
    include(CPack)
endif()
