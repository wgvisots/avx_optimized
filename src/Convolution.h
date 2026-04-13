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

#include <memory>

#include "Aligned.h"

/**
 * This is a tool for performing complex convolution.  The intent is for this to be used on digital RF data (AKA: IQ data).
 * Each "sample" consists of two floats: an in-phase and quadrature (IQ).  They can be considered a std::complex<float>,
 * but I use a flat array of floats for performance reasons.
 *
 * This assumes you are feeding it a continious stream of data, with no breaks in data between calls.  It keeps a
 * kernel's worth of data from each call to be fed into the next call, so your output is smooth with no discontinuities.
 *
 * This class SIMD vectorizes the convolution, which gives some pretty big performance gains, but comes with some
 * caveats:
 * 1) All your inputs/outputs shall be 32-byte aligned.  ('kernel', 'in', and 'out')
 * 2) Short kernels get less performance boosts.
 * 3) Kernels should, but don't need to be, a multiple of 4 samples long.
 * 4) Kernels less than 4 samples shall not be used.
 * 4) Data ('in'/'out') should be much longer than the kernel.
 * 5) Note the use of RESTRICT keyword below - it is your responsibility to obey it.
 *
 * Sizes are given in the number of numbers in each array, not the number samples.  Sizes must, therefore, be an even
 * number.
 *
 * Given all these restrictions, seriously consider using frequency domain convolution (FFT-based) instead.
 */
class Convolution {
public:
    /**
     * Create a convolution object with a given kernel.
     *
     * @param kernel Your kernel, composed of std::complex<float>, but provided as a vector of floats (I,Q,I,Q...)
     * @param kernelSize Number of floats in the kernel (2X number of samples)
     */
    Convolution (const float* kernel, std::size_t kernelSize);
    ~Convolution() = default;

    /**
     *
     * @param in array of std::complex<float> (I,Q,I,Q...), as an array of 'size' floats
     * @param out the complex convolution result, as an array of floats - truncated to the same length as the input
     * @param size number of floats (not complexes) in 'in' (2X number of samples)
     */
    void convolve(const float* __restrict__ in, float* __restrict__ out, std::size_t size);

private:
    const std::size_t kernelLength; // in floats, not complexes
    const aligned::alignedFloatVector theKernel;

    //NOTE: Using raw pointers here, because some compiler versions keep ignoring attributes on __m256 when I try to
    //      use them in an aligned::alignedVector.  And other unclear warnings when using a std::unique_ptr.
    aligned::alignedVecVector kernelReverseUV;
    aligned::alignedVecVector kernelReverseVU;

    //This is used for continuation between calls to convolve.
    aligned::alignedFloatVector leftover;

    //This holds four copies of our data, each offset by one float.  While that sounds terrible, it allows us to
    //perform aligned operations in each of our convolution operations.
    aligned::alignedFloatVector dataAligned[4];
};