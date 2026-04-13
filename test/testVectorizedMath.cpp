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

#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "../src/vectorized_math.h"
#include "../src/Aligned.h"

//TODO -> need many more tests

TEST(VectorizedMathTest, TestVectorMultiply) {
    aligned::alignedComplexVector v1;
    aligned::alignedComplexVector v2;
    aligned::alignedComplexVector v3;

    //note: intentionally not a multiple of 16
    for (std::size_t i = 0; i < 1000; i++) {
        std::complex<float> a{(float)rand(), (float)rand()};
        std::complex<float> b{(float)rand(), (float)rand()};
        float na = norm(a);
        float nb = norm(b);
        v1.emplace_back(a.real()/na, a.imag()/na);
        v2.emplace_back(b.real()/nb, b.imag()/nb);
    }

    for (std::size_t i = 0; i < v1.size(); i++) {
        v3.emplace_back(v1[i] * v2[i]);
    }

    vectorized::vectorMultiply(reinterpret_cast<float*>(v1.data()), reinterpret_cast<float*>(v2.data()), v1.size() * 2);

    for (std::size_t i = 0; i < v1.size(); i++) {
        const float diff = norm(v1[i] - v3[i]);
        //NOTE: I expect some difference here, as vectorized performs IEEE-754 compliant math, and non-vectorized uses
        //      hardware and compiler dependent (probably higher precision) math.
        EXPECT_NEAR(diff, 0.0, 1.0e-6);
    }

}

TEST(VectorizedMathTest, TestApplyLinearLossAdjustment) {
    aligned::alignedFloatVector v1;
    for (std::size_t i = 0; i < 100; i++) {
        v1.emplace_back(1.0f);
    }

    const float start = 0.0F;
    const float end = 1.0F;

    vectorized::applyLinearLossAdjustment(v1.data(), v1.size(), start, end);

    for (std::size_t i = 0; i < v1.size(); i+=2) {
        EXPECT_NEAR(v1[i],   (float)i/(float)v1.size(), 1.0e-6);
        EXPECT_NEAR(v1[i+1], (float)i/(float)v1.size(), 1.0e-6);
    }
}