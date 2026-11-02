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

include(GoogleTest)

add_executable(avx_optimized_tests
               test_aligned.cpp
               test_complex_fir_filter.cpp
               test_convolution.cpp
               test_cpu.cpp
               test_spans.cpp
               test_vectorized.cpp)

target_link_libraries(avx_optimized_tests PRIVATE avx_optimized::avx_optimized GTest::gtest_main)
set_target_properties(avx_optimized_tests PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(avx_optimized_tests PRIVATE -Wall -Wextra -Wpedantic
                       $<$<BOOL:${AVX_OPTIMIZED_WARNINGS_AS_ERRORS}>:-Werror>)

# Each test runs against every instruction set the CPU supports (they are value-parameterized
# over Isa).  The extra runs below also check the AVX_OPTIMIZED_ISA environment override.
gtest_discover_tests(avx_optimized_tests DISCOVERY_MODE PRE_TEST)

foreach(_value scalar avx2 AVX512 Scalar not-an-isa)
    add_test(NAME avx_optimized_env_isa_${_value}
             COMMAND avx_optimized_tests --gtest_filter=Cpu.*)
    set_tests_properties(avx_optimized_env_isa_${_value} PROPERTIES ENVIRONMENT "AVX_OPTIMIZED_ISA=${_value}")
endforeach()

# A misspelled value must produce a warning rather than being silently ignored.
add_test(NAME avx_optimized_env_isa_warns_on_typo
         COMMAND avx_optimized_tests --gtest_filter=Cpu.ReportsWhatWasDetected)
set_tests_properties(avx_optimized_env_isa_warns_on_typo PROPERTIES
                     ENVIRONMENT "AVX_OPTIMIZED_ISA=avx-2"
                     PASS_REGULAR_EXPRESSION "ignoring unrecognised AVX_OPTIMIZED_ISA value 'avx-2'")

# The warning must not pass control characters (such as terminal escape sequences) through, and
# must cut long values short.
string(ASCII 27 _escape)
add_test(NAME avx_optimized_env_isa_warning_is_printable
         COMMAND avx_optimized_tests --gtest_filter=Cpu.ReportsWhatWasDetected)
set_tests_properties(avx_optimized_env_isa_warning_is_printable PROPERTIES
                     ENVIRONMENT "AVX_OPTIMIZED_ISA=avx${_escape}[2Jx0123456789012345678901234567890123456789"
                     PASS_REGULAR_EXPRESSION "value 'avx\\?\\[2Jx012345678901234567890123\\.\\.\\.'")

# Guards against the multi-versioning ODR trap described in src/kernels.h.
if(AVX_OPTIMIZED_ISA_KERNEL_TARGETS AND CMAKE_NM)
    set(_objects "")
    foreach(_target IN LISTS AVX_OPTIMIZED_ISA_KERNEL_TARGETS)
        list(APPEND _objects "$<TARGET_OBJECTS:${_target}>")
    endforeach()
    add_test(NAME avx_optimized_isa_object_symbols
             COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} "-DOBJECTS=$<JOIN:${_objects},|>"
                     -P ${PROJECT_SOURCE_DIR}/cmake/CheckIsaObjectSymbols.cmake)
endif()
