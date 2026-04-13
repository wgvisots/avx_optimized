/**********************************************************************
 * Copyright 2024 - 2026 W. Greg Visotski
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **********************************************************************/

#pragma once

#include <cstdlib>
#include <complex>
#include <vector>
#include <immintrin.h>

#include "AlignedAllocator.h"

namespace aligned
{

    /*
     * A bit on alignment:
     * All __m256 variables need to be aligned on 32 byte boundaries.  This
     * *should* be done with 'alignof(__m256) varName;', but if
     * 'alignof(std::max_align_t) < alignof(__m256)', then the behavior
     * of the alignof is undefined, and it might not throw a warning.
     *
     * On GCC 4.9 and VC, max_align_t gives you 16 and not 32, so you need to
     * use the compiler specific '__attribute__ ((aligned (32)))' (trailing).
     * This gives you a maximum alignment of '__BIGGEST_ALIGNMENT__' (32).
     *
     * Same holds true of 64, for __m512
     */
    #ifndef __GNUG__
    #error "Building on Windows is not currently supported.  This library uses multiple GCC compiler extensions."
    #endif

    #define ALIGN64_END   __attribute__ ((aligned (64)))
    #define ALIGN32_END   __attribute__ ((aligned (32)))
    #define ALIGN16_END   __attribute__ ((aligned (16)))

    template<typename T>
    T *allocateAligned(const aligned::Alignment align, const std::size_t numElements){
        std::size_t len = numElements * sizeof(T);
        len = len + (static_cast<std::size_t>(align) - (len % static_cast<std::size_t>(align)));
        return static_cast<T*>(std::aligned_alloc(static_cast<std::size_t>(align), len));
    }

    template<typename T>
    void freeAligned(T *t)
    {
        free(t);
    }

    //provided mostly as examples:
    template< class T > using alignedVector = std::vector< T, aligned::AlignedAllocator<T, Alignment::AVX> >;
    using alignedComplexVector = alignedVector<std::complex<float> >;
    using alignedFloatVector   = alignedVector<float>;

    //NOTE: If you are getting warning: ignoring attributes on template argument ‘__m256’, then you have to carefully handle
    //      going between __m256 and float[8] using this structure.  Intel tried to help us with this by using the
    //      __may_alias__ attribute, but your compiler can't handle that as a template argument.  Therefore, strict
    //      aliasing rules apply to __m256s that are in this container.
    //
    //      More to the point: the attribute that is getting ignored is __may_alias__.
    using alignedVecVector     = alignedVector<__m256>;
};


