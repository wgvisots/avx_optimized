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

#include <complex>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <type_traits>

#include "gtest/gtest.h"

#include "avx_optimized/aligned.h"

namespace {
bool isAligned(const void* p, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
}
} // namespace

TEST(AlignedVector, DefaultsTo64ByteAlignment) {
    for (std::size_t n : {1, 3, 16, 17, 1000}) {
        avx_optimized::AlignedVector<float>               f(n);
        avx_optimized::AlignedVector<std::complex<float>> c(n);
        avx_optimized::AlignedVector<std::int16_t>        s(n);
        EXPECT_TRUE(isAligned(f.data(), 64)) << n;
        EXPECT_TRUE(isAligned(c.data(), 64)) << n;
        EXPECT_TRUE(isAligned(s.data(), 64)) << n;
    }
}

TEST(AlignedVector, HonorsCustomAlignment) {
    avx_optimized::AlignedVector<double, 128> big(5);
    avx_optimized::AlignedVector<char, 32>    small(5);
    EXPECT_TRUE(isAligned(big.data(), 128));
    EXPECT_TRUE(isAligned(small.data(), 32));
}

TEST(AlignedVector, StaysAlignedWhenGrowing) {
    avx_optimized::AlignedVector<float> v;
    EXPECT_TRUE(v.empty());
    for (int i = 0; i < 5000; ++i) {
        v.push_back(static_cast<float>(i));
        ASSERT_TRUE(isAligned(v.data(), 64));
    }
    EXPECT_EQ(v.size(), 5000U);
    EXPECT_EQ(v[4999], 4999.0F);
}

TEST(AlignedVector, CopyMoveAndSwap) {
    avx_optimized::AlignedVector<int> a(100, 7);
    avx_optimized::AlignedVector<int> b = a;
    EXPECT_EQ(a, b);
    avx_optimized::AlignedVector<int> c = std::move(b);
    EXPECT_EQ(c, a);
    avx_optimized::AlignedVector<int> d(3, 1);
    d.swap(c);
    EXPECT_EQ(d, a);
    EXPECT_TRUE(isAligned(d.data(), 64));
}

TEST(AlignedAllocator, AllInstancesAreEqualAndRebind) {
    avx_optimized::AlignedAllocator<float>  f;
    avx_optimized::AlignedAllocator<double> d(f); // converting constructor
    EXPECT_TRUE(f == avx_optimized::AlignedAllocator<float>{});
    EXPECT_TRUE(f == d);
    EXPECT_FALSE(f != d);
    static_assert(std::allocator_traits<avx_optimized::AlignedAllocator<float>>::is_always_equal::value);
    static_assert(std::is_same_v<std::allocator_traits<avx_optimized::AlignedAllocator<float, 32>>::rebind_alloc<int>,
                                 avx_optimized::AlignedAllocator<int, 32>>);

    // Node-based containers rebind the allocator to their node type.
    std::list<int, avx_optimized::AlignedAllocator<int>>                                        list = {1, 2, 3};
    std::map<int, int, std::less<>, avx_optimized::AlignedAllocator<std::pair<const int, int>>> map  = {
        {1, 2}
    };
    EXPECT_EQ(list.size(), 3U);
    EXPECT_EQ(map.at(1), 2);
}

TEST(AlignedAllocator, RejectsOverflowingRequests) {
    avx_optimized::AlignedAllocator<double> alloc;
    EXPECT_THROW((void)alloc.allocate(std::numeric_limits<std::size_t>::max() / 4), std::bad_array_new_length);
}
