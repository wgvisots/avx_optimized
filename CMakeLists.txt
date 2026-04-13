##############################################################################
# Copyright (C) 2024 - 2026 W. Greg Visotski
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#
#(this is the zlib license)
##############################################################################

cmake_minimum_required(VERSION 3.20)

set(PROJ avx_optimized_math)
project(${PROJ})

set(INC_DIRS "${CMAKE_SOURCE_DIR}/src" )
file(GLOB LIB_SRCS "./src/*.cpp")
file(GLOB LIB_HDRS "./src/*.h")

find_package(mathfun REQUIRED)

set(LIBS "pthread" )

add_library(${PROJ} SHARED ${LIB_SRCS})
#target_include_directories(${PROJ} PUBLIC ${INC_DIRS})
target_link_libraries(${PROJ} ${LIBS})

enable_testing()
add_subdirectory(test)

set_property(TARGET ${PROJ}
             PROPERTY PUBLIC_HEADER "${LIB_HDRS}")

install(TARGETS ${PROJ}
        EXPORT "${PROJ}Config"
        ARCHIVE DESTINATION lib/${PROJ}
        LIBRARY DESTINATION lib/${PROJ}
        PUBLIC_HEADER DESTINATION include/${PROJ})

export(TARGETS ${PROJ}
       NAMESPACE ${PROJ}::
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJ}Config.cmake"
)
install(EXPORT "${PROJ}Config"
        DESTINATION lib/${PROJ}/cmake
        NAMESPACE ${PROJ}::
)

set(CPACK_PACKAGE_VERSION ${VERSION})
set(CPACK_PACKAGE_NAME ${PROJ})
set(CPACK_PACKAGE_CONTACT "W. Greg Visotski")
set(CPACK_PACKAGE_VENDOR "NMERC")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
include(CPack)
