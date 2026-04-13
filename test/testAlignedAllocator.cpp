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

#include <cstdint>
#include <iostream>

#include "gtest/gtest.h"

#include "../src/Aligned.h"

TEST(AlignedAllocator, TestAligned){
    std::size_t* thing = aligned::allocateAligned<std::size_t>(aligned::Alignment::AVX, 10);
    EXPECT_TRUE(0 == ((std::uint64_t)thing % 32));

    using allocator = aligned::AlignedAllocator<std::size_t, aligned::Alignment::AVX>;
    std::vector<std::size_t, allocator> vec;
    vec.resize(1);

    std::size_t* vptr = vec.data();
    EXPECT_TRUE(0 == ((std::uint64_t)vptr % 32)) << std::endl;
}

TEST(AlignedAllocator, TestVec1) {
    aligned::alignedVector<float> v0;
    EXPECT_TRUE(v0.empty());
}

TEST(AlignedAllocator, TestVec2) {
    aligned::alignedVector<float> v1(16);
    EXPECT_EQ(16, v1.size());
}

TEST(AlignedAllocator, TestVec3) {
    aligned::alignedVector<float> v1(32);
    EXPECT_EQ(32, v1.size());
}