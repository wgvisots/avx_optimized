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

#include <iostream>
#include <numeric>

#include "gtest/gtest.h"

#include "../src/Aligned.h"
#include "../src/Convolution.h"

TEST(ConvolutionTest, TestCreate) {
    //16 samples
    aligned::alignedFloatVector kernel(32, 1.0F);
    aligned::alignedFloatVector data(200, 0.0F);
    data[100] = 1.0f;

    Convolution con(kernel.data(), kernel.size());
}

/**
 * What we're going to do here is make a kernel where all values are equal.  The Convolution will normalize this to
 * a total of 1.  This'll have the effect of outputting, for each sample under the kernel, the average of all samples
 * under the filter.
 *
 * We are then going to make our data all zeros, with a single 1 value.  This means that after we filter, we can sum
 * the result and get a value of 1 (with floating point rounding error).
 */
TEST(ConvolutionTest, TestConvolve) {
    //16 samples
    aligned::alignedFloatVector kernel(32, 1.0F);
    aligned::alignedFloatVector data(200, 0.0F);
    data[100] = 1.0f;

    Convolution con(kernel.data(), kernel.size());

    aligned::alignedFloatVector result(data.size(), 0.0);

    con.convolve(data.data(), result.data(), data.size());

    float v = std::accumulate(result.begin(), result.end(), 0.0F);

    // std::for_each(result.begin(), result.end(), [](const float& f){std::cout << f << ", ";});
    // std::cout << std::endl;
    EXPECT_NEAR(v, 1.0F, 0.1F);
}
