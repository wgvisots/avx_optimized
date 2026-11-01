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

#include <complex>
#include <cstddef>
#include <stdexcept>

#include "avx_optimized/complex_fir_filter.h"
#include "avx_optimized/vectorized.h"

namespace avx_optimized {

/**
 * Compatibility facade with the interface of the pre-0.2 `Convolution` class: complex data
 * passed as interleaved floats (I, Q, I, Q, ...), with sizes counted in floats.  It forwards to
 * ComplexFirFilter, which new code should use directly.
 *
 * Porting old code usually only needs the new include and namespace:
 *
 *     #include <avx_optimized/convolution.h>
 *     avx_optimized::Convolution conv(kernel, kernelSizeInFloats);
 *     conv.convolve(in, out, sizeInFloats);
 *
 * Results differ from the old class, which was broken in several ways (see docs/MIGRATION.md):
 *  - The kernel is used as given.  The old class divided it by its complex sum, which for
 *    zero-mean kernels such as matched-filter templates meant dividing by roughly zero.  If you
 *    really need that scaling, divide the kernel by its sum before passing it in.
 *  - Output is causal and continuous across calls: out sample k is the full convolution of the
 *    stream up to input sample k.  The old class wrote "valid" outputs (full overlap only), which
 *    places everything delay() samples earlier than here.  So a matched-filter peak that used to
 *    appear at the pulse start now appears at the pulse end, delay() samples later.
 *  - Any even, nonzero kernel size works, alignment is not required, and in == out is allowed.
 */
class Convolution {
public:
    /**
     * @param kernel     complex taps as interleaved floats (I, Q, I, Q, ...), in convolution order
     * @param kernelSize number of floats in `kernel` (2 x number of taps); must be even and nonzero
     * @throws std::invalid_argument if kernelSize is odd or zero, or kernel is null
     */
    Convolution(const float* kernel, std::size_t kernelSize)
        : filter_(checkedKernel(kernel, kernelSize), kernelSize / 2) {}

    /**
     * Convolve the next part of the stream.
     *
     * @param in   complex samples as interleaved floats
     * @param out  receives one complex output per input sample (may be the same pointer as `in`)
     * @param size number of floats in `in` and `out` (2 x number of samples); must be even (a
     *             trailing odd float is left untouched)
     */
    void convolve(const float* in, float* out, std::size_t size) noexcept {
        filter_.filter(avx_optimized::asComplex(in), avx_optimized::asComplex(out), size / 2);
    }

    /// End the stream: write the last 2 * delay() floats of output; see ComplexFirFilter::flush().
    void flush(float* out) noexcept { filter_.flush(avx_optimized::asComplex(out)); }

    /// Forget the stream history.
    void reset() noexcept { filter_.reset(); }

    /// kernelSize / 2 - 1, in samples; see ComplexFirFilter::delay().
    [[nodiscard]] std::size_t delay() const noexcept { return filter_.delay(); }

    /// The underlying filter.
    [[nodiscard]] ComplexFirFilter& firFilter() noexcept { return filter_; }

    /// The underlying filter.
    [[nodiscard]] const ComplexFirFilter& firFilter() const noexcept { return filter_; }

private:
    static const std::complex<float>* checkedKernel(const float* kernel, std::size_t kernelSize) {
        if (kernelSize % 2 != 0) {
            throw std::invalid_argument("avx_optimized::Convolution: kernelSize counts floats and must be even");
        }
        return avx_optimized::asComplex(kernel); // null and zero are rejected by ComplexFirFilter
    }

    ComplexFirFilter filter_;
};

} // namespace avx_optimized
