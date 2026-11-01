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

# Locates (or downloads) the build-time dependencies:
#   mathfun::mathfun  header-only vectorized math (atan2 etc.); x86 builds only
#   GTest::gtest_main unit tests only
#
# An installed copy found by find_package() always wins.  Otherwise, if
# AVX_OPTIMIZED_FETCH_DEPENDENCIES is ON, a pinned version is downloaded.

include(FetchContent)

# Pinned mathfun revision used when no installed copy is found.
set(AVX_OPTIMIZED_MATHFUN_REPOSITORY "https://github.com/wgvisots/avx_mathfun.git"
    CACHE STRING "Git repository to fetch mathfun from")
set(AVX_OPTIMIZED_MATHFUN_TAG "291a17c2e8d462664107099a36fc4a0a9ad0017e"
    CACHE STRING "mathfun revision to fetch")

if(AVX_OPTIMIZED_X86 AND NOT TARGET mathfun::mathfun)
    find_package(mathfun CONFIG QUIET)
    if(TARGET mathfun::mathfun)
        message(STATUS "avx_optimized: using installed mathfun (${mathfun_DIR})")
    elseif(AVX_OPTIMIZED_FETCH_DEPENDENCIES)
        message(STATUS "avx_optimized: fetching mathfun ${AVX_OPTIMIZED_MATHFUN_TAG}")
        # mathfun is header-only.  SOURCE_SUBDIR points at a directory without a CMakeLists.txt so
        # that only the sources are downloaded, without pulling mathfun's tests and packaging
        # into this build.
        FetchContent_Declare(avx_optimized_mathfun
                             GIT_REPOSITORY ${AVX_OPTIMIZED_MATHFUN_REPOSITORY}
                             GIT_TAG ${AVX_OPTIMIZED_MATHFUN_TAG}
                             SOURCE_SUBDIR do-not-configure)
        FetchContent_MakeAvailable(avx_optimized_mathfun)

        # Installed mathfun is included as <mathfun/...>; reproduce that layout.
        set(_mathfun_include "${CMAKE_BINARY_DIR}/_deps/mathfun-include")
        file(COPY "${avx_optimized_mathfun_SOURCE_DIR}/src/avx_mathfun.h"
                  "${avx_optimized_mathfun_SOURCE_DIR}/src/avx512_mathfun.h"
             DESTINATION "${_mathfun_include}/mathfun")
        add_library(mathfun::mathfun INTERFACE IMPORTED)
        set_target_properties(mathfun::mathfun PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_mathfun_include}")
    else()
        message(FATAL_ERROR
                "avx_optimized: mathfun was not found.  Install it (https://github.com/wgvisots/avx_mathfun), "
                "point CMAKE_PREFIX_PATH at it, or configure with -DAVX_OPTIMIZED_FETCH_DEPENDENCIES=ON.")
    endif()
endif()

if(AVX_OPTIMIZED_BUILD_TESTS AND NOT TARGET GTest::gtest_main)
    find_package(GTest CONFIG QUIET)
    if(NOT TARGET GTest::gtest_main)
        find_package(GTest QUIET) # FindGTest module fallback
    endif()
    if(TARGET GTest::gtest_main)
        message(STATUS "avx_optimized: using installed GoogleTest")
    elseif(AVX_OPTIMIZED_FETCH_DEPENDENCIES)
        message(STATUS "avx_optimized: fetching GoogleTest")
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(googletest
                             URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
                             URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
                             DOWNLOAD_EXTRACT_TIMESTAMP ON)
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR "avx_optimized: GoogleTest was not found.  Install it, or configure with "
                            "-DAVX_OPTIMIZED_FETCH_DEPENDENCIES=ON (or -DAVX_OPTIMIZED_BUILD_TESTS=OFF).")
    endif()
endif()
