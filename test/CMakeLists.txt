###########################################################################
# CMakeLists.txt
#
# Copyright 2024, 2025 W. Greg Visotski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###########################################################################

cmake_minimum_required(VERSION 3.20)

set(PROJ avx_mathfun_tests)
project(${PROJ})

find_package(GTest REQUIRED)

set(INC_DIRS "./"
             ${gtest_INCLUDE_DIR})

set(LIBS avx_optimized_math)

################################
# GTest
################################
enable_testing()


################################
# Unit Tests
################################
file(GLOB_RECURSE SRCS *.cpp)

# Add test cpp file
add_executable( runUnitTests ${SRCS} )

target_include_directories(runUnitTests PUBLIC ${INC_DIRS})
target_link_libraries(runUnitTests ${LIBS})

# Link test executable against gtest & gtest_main
target_link_libraries(runUnitTests gtest gtest_main)
add_test(NAME runUnitTests
         COMMAND runUnitTests
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})


