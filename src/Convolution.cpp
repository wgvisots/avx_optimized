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

#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>

#include "Convolution.h"

#include <algorithm>
#include <numeric>

#include "vectorized_math.h"

namespace {
const __m256& negateEveryOther() {
    static __m256 ret ALIGN32_END;
    static bool set{false};
    if (!set) {
        std::mutex lock;
        std::lock_guard lk(lock);
        if (!set) {
            set = true;
            ret = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
        }
    }
    return ret;
}

aligned::alignedFloatVector copyAndNormalizeKernel(const float* kernel, const std::size_t& kernelSize) {
    assert(0 == (kernelSize%2));
    aligned::alignedFloatVector ret(kernel, kernel + kernelSize);
    std::complex<float>* cVec = reinterpret_cast<std::complex<float>*>(ret.data());
    std::complex<float> sum = std::accumulate(cVec,
                                              cVec + (kernelSize / 2),
                                              std::complex<float>(0.0F, 0.0F));

    std::for_each(cVec,
                  cVec + (kernelSize / 2),
                  [&sum](std::complex<float>& v) { v /= sum; });
    return ret;
}
}

Convolution::Convolution(const float* kernel, const std::size_t kernelSize) : kernelLength(kernelSize),
                                                                              theKernel(copyAndNormalizeKernel(
                                                                                  kernel, kernelSize)) {
    this->kernelReverseUV.resize(kernelSize, vectorized::zeroVec());
    this->kernelReverseVU.resize(kernelSize, vectorized::zeroVec());

    const __m256& neg = negateEveryOther();
    for (unsigned int i = 0; i < kernelSize; i += 2) {
        __m256 u ALIGN32_END;
        __m256 v ALIGN32_END;
        v = _mm256_broadcast_ss(&(this->theKernel[(kernelSize - i) - 1]));
        u = _mm256_broadcast_ss(&(this->theKernel[(kernelSize - (i + 1)) - 1]));

        this->kernelReverseUV[i] = _mm256_blend_ps(u, v, 0XAA); // u v u v (8X32)
        this->kernelReverseVU[i] = _mm256_blend_ps(u, v, 0X55); // u -v u -v (8X32)
        this->kernelReverseUV[i] = _mm256_mul_ps(this->kernelReverseUV[i], neg); // u -v u -v (8X32)
    }

    this->leftover.resize(kernelSize, 0.0F);
}

void Convolution::convolve(const float* __restrict__ in, float* __restrict__ out, const std::size_t size) {
    const std::size_t longerLength = size + this->kernelLength;
    const std::size_t inAlignedSize = longerLength + (8 - (longerLength % 8));
    //this is to make sure that we end on a full __m256
    if (inAlignedSize > this->dataAligned[0].size()) {
        for (aligned::alignedFloatVector& v: this->dataAligned) {
            v.resize(inAlignedSize);
        }
    }

    __m256 dataBlock ALIGN32_END;
    __m256 prod ALIGN32_END;
    __m256 acc ALIGN32_END;
    __m256 ii ALIGN32_END;
    __m256 qq ALIGN32_END;

    /* We need a set of four copies of our input, each one offset by one whole sample (two floats) from its predecessor.
     */
    for (std::size_t i = 0; i < 8; i += 2) {
        //2 for complex
        std::memcpy(&(this->dataAligned[i >> 1][0]), (in + i), (size - i) * sizeof(float));
    }

    const std::size_t end = size - this->kernelLength;
    for (std::size_t i = 0; i < end; i += 8) {
        acc = _mm256_setzero_ps();
        for (unsigned int j = 0; j < this->kernelLength; j += 8) {
            const std::size_t offset = i + j;
            int kk = 0;
            for (int k = 0; k < 4; k++, kk += 2) {
                assert((j+kk) < this->kernelLength);
                dataBlock = _mm256_load_ps(&(this->dataAligned[k][0]) + offset);
                ii = _mm256_mul_ps(this->kernelReverseUV[j + kk], dataBlock);
                qq = _mm256_mul_ps(this->kernelReverseVU[j + kk], dataBlock);
                ii = _mm256_hadd_ps(ii, qq); //i i q q i i q q
                prod = _mm256_permute_ps(ii, 0xD8); //i q i q i q i q
                acc = _mm256_add_ps(acc, prod);
            }
        }
        _mm256_storeu_ps(out + i, acc);
    }

    //and the remainder
    out[end] = 0.0;
    out[end - 1] = 0.0;
    for (unsigned int k = 0; k < this->kernelLength; k += 2) {
        const float* kernelSample = &(this->theKernel[kernelLength - k - 1]);
        const float* dataSample = &(this->dataAligned[0][end + k]);
        const std::complex<float>* kernelSampleComplex = reinterpret_cast<const std::complex<float>*>(kernelSample);
        const std::complex<float>* dataSampleComplex = reinterpret_cast<const std::complex<float>*>(dataSample);
        *reinterpret_cast<std::complex<float>*>(&(out[end - 1])) += *dataSampleComplex * *kernelSampleComplex;
    }

    //TODO -> I don't think I want this
    std::copy(in + ((size - 1) - this->kernelLength), in + size, this->leftover.data());
}
