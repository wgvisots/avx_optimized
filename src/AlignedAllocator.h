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
#include <memory>
#include <limits>

namespace aligned{

    enum class Alignment : std::size_t
    {
        Normal = sizeof(void*),
        SSE    = 16,
        AVX    = 32,
        AVX512 = 64
    };

    /**
     * Just a stupid simple allocator to get aligned memory for a std::vector
     *
     * Future -> maybe do something cool like described here:
     * https://indiegamedev.net/2022/03/27/custom-c20-memory-allocators-for-stl-containers/
     *
     * @tparam T
     * @tparam ALIGNMENT_TYPE
     */
    template<typename T, aligned::Alignment ALIGNMENT_TYPE = aligned::Alignment::AVX>
    class AlignedAllocator{
    public:
        static_assert(static_cast<std::size_t>(ALIGNMENT_TYPE) >= alignof(T), "You may be violating minimum alignment requirements.");
        using value_type = T;
        static constexpr std::align_val_t ALIGNMENT{static_cast<std::align_val_t>(ALIGNMENT_TYPE)};

        /**
         * Thanks to Bo Persson
         * @see https://stackoverflow.com/a/48062758/2191065
         */
        template<class OTHER>
        struct rebind{
            using other = AlignedAllocator<OTHER, ALIGNMENT_TYPE>;
        };

        constexpr AlignedAllocator() noexcept = default;
        constexpr AlignedAllocator( const AlignedAllocator& ) noexcept = default;

        template<typename U>
        constexpr explicit AlignedAllocator( AlignedAllocator<U, ALIGNMENT_TYPE> const& ) noexcept{}

        [[nodiscard]] static constexpr T* allocate(std::size_t n){
            if ( n > (std::numeric_limits<std::size_t>::max() / sizeof(T))){
                throw std::bad_array_new_length();
            }
            T* ret = static_cast<T*>(::operator new[](n * sizeof(T), ALIGNMENT));
            return ret;
        }

        static void deallocate(T* allocatedPointer, [[maybe_unused]] std::size_t n)
        {
            ::operator delete[]( allocatedPointer, ALIGNMENT );
        }
    };
}