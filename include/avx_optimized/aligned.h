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

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

namespace avx_optimized {

/**
 * Alignment suitable for every vector width this library uses: one AVX-512 register, which is
 * also one cache line on every x86-64 CPU that matters.
 */
inline constexpr std::size_t kSimdAlignment = 64;

/**
 * A minimal standard-conforming allocator that returns memory aligned to `Alignment` bytes.
 *
 * None of this library's functions *require* aligned memory -- every kernel uses unaligned loads
 * and stores.  Aligned buffers are still faster, because they avoid loads that straddle two cache
 * lines.  That matters most for AVX-512, where every misaligned 64-byte access is a split.
 *
 * @tparam T         element type
 * @tparam Alignment alignment in bytes; a power of two that is at least alignof(T)
 */
template <typename T, std::size_t Alignment = kSimdAlignment>
class AlignedAllocator {
public:
    static_assert(Alignment != 0 && (Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

    using value_type                             = T;              ///< Element type.
    using size_type                              = std::size_t;    ///< Type of allocation sizes.
    using difference_type                        = std::ptrdiff_t; ///< Type of pointer differences.
    using propagate_on_container_move_assignment = std::true_type; ///< Moving a container moves its allocator.
    using is_always_equal                        = std::true_type; ///< All instances can free each other's memory.

    /// The same allocator for another element type, as node-based containers need.  Spelled out
    /// because std::allocator_traits can only rebind allocators whose template parameters are all
    /// types, and Alignment is not.
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>; ///< AlignedAllocator<U, Alignment>.
    };

    constexpr AlignedAllocator() noexcept = default;

    /// Conversion from the allocator for another element type.  Implicit on purpose: the standard
    /// allocator requirements need this conversion to be implicit.
    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>& /*other*/) noexcept {}

    /// Allocate uninitialized, aligned storage for `count` elements.
    /// @throws std::bad_array_new_length if the size overflows, std::bad_alloc if out of memory
    [[nodiscard]] T* allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t{Alignment}));
    }

    /// Free storage returned by allocate().
    void deallocate(T* pointer, std::size_t /*count*/) noexcept {
        ::operator delete(pointer, std::align_val_t{Alignment});
    }
};

/// All AlignedAllocators with the same alignment are interchangeable.
template <typename T, typename U, std::size_t Alignment>
constexpr bool operator==(const AlignedAllocator<T, Alignment>& /*a*/,
                          const AlignedAllocator<U, Alignment>& /*b*/) noexcept {
    return true;
}

/// All AlignedAllocators with the same alignment are interchangeable.
template <typename T, typename U, std::size_t Alignment>
constexpr bool operator!=(const AlignedAllocator<T, Alignment>& /*a*/,
                          const AlignedAllocator<U, Alignment>& /*b*/) noexcept {
    return false;
}

/// A std::vector whose storage is aligned to `Alignment` bytes (64 by default).
template <typename T, std::size_t Alignment = kSimdAlignment>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

} // namespace avx_optimized
