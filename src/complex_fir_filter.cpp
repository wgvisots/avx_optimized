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

#include "avx_optimized/complex_fir_filter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "avx_optimized/vectorized.h"
#include "kernels.h"

namespace avx_optimized {
namespace {

// Input is processed in blocks of at most this many samples.  The working buffer is sized from it
// once, so filter() never allocates.  4096 complex samples (32 KiB) plus the history fits
// comfortably in L2.
constexpr std::size_t kBlockSize = 4096;

template <typename T>
const T* checkedTaps(const T* taps, std::size_t numTaps) {
    if (taps == nullptr || numTaps == 0) {
        throw std::invalid_argument("avx_optimized::ComplexFirFilter: at least one tap is required");
    }
    return taps;
}

bool allImaginaryPartsZero(const std::vector<std::complex<float>>& taps) {
    return std::all_of(taps.begin(), taps.end(), [](const std::complex<float>& tap) { return tap.imag() == 0.0F; });
}

} // namespace

ComplexFirFilter::ComplexFirFilter(const std::complex<float>* taps, std::size_t numTaps)
    : taps_(checkedTaps(taps, numTaps), taps + numTaps)
    , realTaps_(allImaginaryPartsZero(taps_))
    , reversedTapsReal_(numTaps)
    , reversedTapsImag_(realTaps_ ? 0 : numTaps)
    , buffer_(numTaps - 1 + kBlockSize, std::complex<float>(0.0F, 0.0F)) {
    for (std::size_t j = 0; j < numTaps; ++j) {
        reversedTapsReal_[j] = taps_[numTaps - 1 - j].real();
        if (!realTaps_) {
            reversedTapsImag_[j] = taps_[numTaps - 1 - j].imag();
        }
    }
}

ComplexFirFilter::ComplexFirFilter(const float* taps, std::size_t numTaps)
    : taps_(checkedTaps(taps, numTaps), taps + numTaps) // each float becomes {tap, 0}
    , realTaps_(true)
    , reversedTapsReal_(taps, taps + numTaps)
    , reversedTapsImag_()
    , buffer_(numTaps - 1 + kBlockSize, std::complex<float>(0.0F, 0.0F)) {
    std::reverse(reversedTapsReal_.begin(), reversedTapsReal_.end());
}

ComplexFirFilter::ComplexFirFilter(std::span<const std::complex<float>> taps)
    : ComplexFirFilter(taps.data(), taps.size()) {}

ComplexFirFilter::ComplexFirFilter(std::span<const float> taps)
    : ComplexFirFilter(taps.data(), taps.size()) {}

ComplexFirFilter ComplexFirFilter::matchedFilter(std::span<const std::complex<float>> templ, TemplateScaling scaling) {
    return matchedFilter(templ.data(), templ.size(), scaling);
}

ComplexFirFilter
ComplexFirFilter::matchedFilter(const std::complex<float>* templ, std::size_t length, TemplateScaling scaling) {
    checkedTaps(templ, length);

    double scaleFactor = 1.0;
    if (scaling == TemplateScaling::UnitEnergy) {
        double energy = 0.0;
        for (std::size_t n = 0; n < length; ++n) {
            energy += std::norm(std::complex<double>(templ[n]));
        }
        if (!(energy > 0.0) || !std::isfinite(energy)) {
            throw std::invalid_argument("avx_optimized::ComplexFirFilter::matchedFilter: cannot scale a template with "
                                        "zero or non-finite energy to unit energy");
        }
        scaleFactor = 1.0 / std::sqrt(energy);
    }

    std::vector<std::complex<float>> taps(length);
    for (std::size_t m = 0; m < length; ++m) {
        const std::complex<double> sample(templ[length - 1 - m]);
        taps[m] = std::complex<float>(std::conj(sample) * scaleFactor);
    }
    return ComplexFirFilter(taps);
}

void ComplexFirFilter::filter(const std::complex<float>* in, std::complex<float>* out, std::size_t count) noexcept {
    run(in, out, count);
}

void ComplexFirFilter::filter(std::span<const std::complex<float>> in, std::span<std::complex<float>> out) {
    if (out.size() != in.size()) {
        throw std::invalid_argument("avx_optimized::ComplexFirFilter::filter: out has " + std::to_string(out.size())
                                    + " samples but must have " + std::to_string(in.size()));
    }
    run(in.data(), out.data(), in.size());
}

void ComplexFirFilter::filter(std::span<std::complex<float>> io) noexcept {
    run(io.data(), io.data(), io.size());
}

void ComplexFirFilter::flush(std::complex<float>* out) noexcept {
    run(nullptr, out, delay());
}

std::span<std::complex<float>> ComplexFirFilter::flush(std::span<std::complex<float>> out) {
    if (out.size() < delay()) {
        throw std::invalid_argument("avx_optimized::ComplexFirFilter::flush: out has " + std::to_string(out.size())
                                    + " samples but needs at least delay() = " + std::to_string(delay()));
    }
    run(nullptr, out.data(), delay());
    return out.first(delay());
}

void ComplexFirFilter::reset() noexcept {
    std::fill(buffer_.begin(), buffer_.end(), std::complex<float>(0.0F, 0.0F));
}

/// Filters `count` samples from `in`, or `count` zeros if `in` is null.
void ComplexFirFilter::run(const std::complex<float>* in, std::complex<float>* out, std::size_t count) noexcept {
    const avx_optimized::detail::KernelTable& kernels    = avx_optimized::detail::activeKernels();
    const std::size_t                         history    = delay();
    std::complex<float>* const                newSamples = buffer_.data() + history;

    while (count > 0) {
        const std::size_t block = std::min(count, kBlockSize);

        // Copying the input before writing any output is what makes in == out safe.
        if (in != nullptr) {
            std::copy(in, in + block, newSamples);
            in += block;
        } else {
            std::fill(newSamples, newSamples + block, std::complex<float>(0.0F, 0.0F));
        }

        if (realTaps_) {
            kernels.complexFirRealTaps(asFloats(buffer_.data()),
                                       reversedTapsReal_.data(),
                                       taps_.size(),
                                       asFloats(out),
                                       block);
        } else {
            kernels.complexFir(asFloats(buffer_.data()),
                               reversedTapsReal_.data(),
                               reversedTapsImag_.data(),
                               taps_.size(),
                               asFloats(out),
                               block);
        }

        // Keep the newest `history` samples for the next block (the ranges may overlap, but the
        // destination is in front of the source, which std::copy allows).
        std::copy(buffer_.begin() + static_cast<std::ptrdiff_t>(block),
                  buffer_.begin() + static_cast<std::ptrdiff_t>(block + history),
                  buffer_.begin());

        out   += block;
        count -= block;
    }
}

} // namespace avx_optimized
