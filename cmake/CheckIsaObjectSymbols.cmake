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

# Fails if an AVX2/AVX-512 kernel object file defines a weak (COMDAT) symbol.
#
# Weak symbols are inline functions and templates that were emitted out-of-line.  The linker
# keeps one copy of each, chosen arbitrarily, so a copy compiled with AVX-512 could end up being
# called from baseline code and crash with SIGILL on CPUs without AVX-512.  See src/kernels.h.
# mathfun's own namespaces are allowed: they are only ever compiled for their own ISA.
#
# Usage: cmake -DNM=<nm> -DOBJECTS=<obj1>|<obj2>|... -P CheckIsaObjectSymbols.cmake

string(REPLACE "|" ";" objects "${OBJECTS}")
set(offenders "")
foreach(object IN LISTS objects)
    execute_process(COMMAND "${NM}" --defined-only --demangle "${object}"
                    OUTPUT_VARIABLE symbols
                    RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "${NM} failed on ${object}")
    endif()
    string(REPLACE ";" "," symbols "${symbols}")
    string(REPLACE "\n" ";" lines "${symbols}")
    foreach(line IN LISTS lines)
        if(line MATCHES "^[0-9a-fA-F]* *[WVu] (.*)$")
            set(symbol "${CMAKE_MATCH_1}")
            # Allowed: mathfun's namespaces, and two compiler-generated helpers that contain no
            # vector instructions (the personality reference, and clang's terminate trampoline).
            if(NOT symbol MATCHES "^(avx_mathfun::|avx512_mathfun::|DW\\.ref\\.__gxx_personality_v0$|__clang_call_terminate$)")
                get_filename_component(name "${object}" NAME)
                string(APPEND offenders "\n  ${name}: ${symbol}")
            endif()
        endif()
    endforeach()
endforeach()

if(offenders)
    message(FATAL_ERROR "ISA kernel objects define weak symbols that could be shared with baseline code "
                        "(see the rules in src/kernels.h):${offenders}")
endif()
message(STATUS "No shared weak symbols in ISA kernel objects")
