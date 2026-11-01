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

// Portable kernels.  The fallback on CPUs without AVX2 and the only implementation on non-x86
// builds.  Compiled with the project's baseline flags, so ordinary C++ is fine here.

#include <algorithm>
#include <cmath>
#include <limits>

#include "kernels.h"

namespace avx_optimized::detail {
namespace {

void int16ToFloat(const std::int16_t* in, float* out, std::size_t count, float scale) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>(in[i]) * scale;
    }
}

void floatToInt16(const float* in, std::int16_t* out, std::size_t count, float scale) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        float value = in[i] * scale;
        if (std::isnan(value)) {
            value = 0.0F;
        }
        value  = std::min(std::max(value, -32768.0F), 32767.0F);
        out[i] = static_cast<std::int16_t>(std::nearbyint(value));
    }
}

void add(float* io, const float* other, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        io[i] += other[i];
    }
}

void addSaturated(std::int16_t* io, const std::int16_t* other, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        const int sum = static_cast<int>(io[i]) + static_cast<int>(other[i]);
        io[i]         = static_cast<std::int16_t>(std::clamp(sum, -32768, 32767));
    }
}

void scale(float* io, float factor, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        io[i] *= factor;
    }
}

void reverse(const float* in, float* out, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        out[count - 1 - i] = in[i];
    }
}

void scaleIQ(float* io, float iFactor, float qFactor, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        io[2 * i]     *= iFactor;
        io[2 * i + 1] *= qFactor;
    }
}

void complexMultiply(float* io, const float* other, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        const float ar = io[2 * i];
        const float ai = io[2 * i + 1];
        const float br = other[2 * i];
        const float bi = other[2 * i + 1];

        io[2 * i]     = ar * br - ai * bi;
        io[2 * i + 1] = ar * bi + ai * br;
    }
}

float magnitudeSquaredOf(const float* sample) noexcept {
    return sample[0] * sample[0] + sample[1] * sample[1];
}

void magnitudeSquared(const float* in, float* out, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = magnitudeSquaredOf(in + 2 * i);
    }
}

void magnitude(const float* in, float* out, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = std::sqrt(magnitudeSquaredOf(in + 2 * i));
    }
}

void toPolar(const float* in, float* magnitudeOut, float* phaseOut, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        magnitudeOut[i] = std::sqrt(magnitudeSquaredOf(in + 2 * i));
        phaseOut[i]     = std::atan2(in[2 * i + 1], in[2 * i]);
    }
}

void applyLinearGainRamp(float* io, float start, float step, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        const float gain = start + static_cast<float>(i) * step;

        io[2 * i]     *= gain;
        io[2 * i + 1] *= gain;
    }
}

float decibelsOf(float power) noexcept {
    if (std::isnan(power) || power < 0.0F) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (power == 0.0F) {
        return -std::numeric_limits<float>::infinity();
    }
    return 10.0F * std::log10(power); // std::log10 handles +infinity and subnormals
}

void powerToDecibels(const float* power, float* decibels, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        decibels[i] = decibelsOf(power[i]);
    }
}

void complexPowerToDecibels(const float* in, float* decibels, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        decibels[i] = decibelsOf(magnitudeSquaredOf(in + 2 * i));
    }
}

void complexFir(const float* in,
                const float* tapsReal,
                const float* tapsImag,
                std::size_t  numTaps,
                float*       out,
                std::size_t  count) noexcept {
    // Accumulate x*Re(h) and x*Im(h) separately and combine at the end, the same way the vector
    // kernels do, so all implementations round alike.
    for (std::size_t k = 0; k < count; ++k) {
        const float* x = in + 2 * k;

        float realTimesRe = 0.0F; // sum x.re * h.re
        float imagTimesRe = 0.0F; // sum x.im * h.re
        float realTimesIm = 0.0F; // sum x.re * h.im
        float imagTimesIm = 0.0F; // sum x.im * h.im
        for (std::size_t j = 0; j < numTaps; ++j) {
            const float xr = x[2 * j];
            const float xi = x[2 * j + 1];

            realTimesRe += xr * tapsReal[j];
            imagTimesRe += xi * tapsReal[j];
            realTimesIm += xr * tapsImag[j];
            imagTimesIm += xi * tapsImag[j];
        }
        out[2 * k]     = realTimesRe - imagTimesIm;
        out[2 * k + 1] = imagTimesRe + realTimesIm;
    }
}

void complexFirRealTaps(const float* in,
                        const float* taps,
                        std::size_t  numTaps,
                        float*       out,
                        std::size_t  count) noexcept {
    for (std::size_t k = 0; k < count; ++k) {
        const float* x = in + 2 * k;

        float real = 0.0F;
        float imag = 0.0F;
        for (std::size_t j = 0; j < numTaps; ++j) {
            real += x[2 * j] * taps[j];
            imag += x[2 * j + 1] * taps[j];
        }
        out[2 * k]     = real;
        out[2 * k + 1] = imag;
    }
}

} // namespace

extern const KernelTable kScalarKernels = {
    .isa                    = Isa::Scalar,
    .int16ToFloat           = &int16ToFloat,
    .floatToInt16           = &floatToInt16,
    .add                    = &add,
    .addSaturated           = &addSaturated,
    .scale                  = &scale,
    .reverse                = &reverse,
    .scaleIQ                = &scaleIQ,
    .complexMultiply        = &complexMultiply,
    .magnitudeSquared       = &magnitudeSquared,
    .magnitude              = &magnitude,
    .toPolar                = &toPolar,
    .applyLinearGainRamp    = &applyLinearGainRamp,
    .powerToDecibels        = &powerToDecibels,
    .complexPowerToDecibels = &complexPowerToDecibels,
    .complexFir             = &complexFir,
    .complexFirRealTaps     = &complexFirRealTaps,
};

} // namespace avx_optimized::detail
