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

// The public functions of vectorized.h.  Each one forwards to the active instruction set's kernel
// (see kernels.h), converting std::complex<float> pointers to the interleaved floats the kernels
// work on.

#include "avx_optimized/vectorized.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "kernels.h"

namespace avx_optimized {
namespace {

const avx_optimized::detail::KernelTable& kernels() noexcept {
    return avx_optimized::detail::activeKernels();
}

/// Throws std::invalid_argument unless `actual` == `required`, naming the function and argument.
void requireSize(const char* function, const char* argument, std::size_t actual, std::size_t required) {
    if (actual != required) {
        throw std::invalid_argument(std::string("avx_optimized::") + function + ": " + argument + " has "
                                    + std::to_string(actual) + " elements but must have " + std::to_string(required));
    }
}

} // namespace

void int16ToFloat(const std::int16_t* in, float* out, std::size_t count, float scale) noexcept {
    kernels().int16ToFloat(in, out, count, scale);
}

void floatToInt16(const float* in, std::int16_t* out, std::size_t count, float scale) noexcept {
    kernels().floatToInt16(in, out, count, scale);
}

void int16ToComplex(const std::int16_t*  interleavedIQ,
                    std::complex<float>* out,
                    std::size_t          count,
                    float                scale) noexcept {
    kernels().int16ToFloat(interleavedIQ, asFloats(out), 2 * count, scale);
}

void complexToInt16(const std::complex<float>* in,
                    std::int16_t*              interleavedIQ,
                    std::size_t                count,
                    float                      scale) noexcept {
    kernels().floatToInt16(asFloats(in), interleavedIQ, 2 * count, scale);
}

void add(float* io, const float* other, std::size_t count) noexcept {
    kernels().add(io, other, count);
}

void add(std::complex<float>* io, const std::complex<float>* other, std::size_t count) noexcept {
    kernels().add(asFloats(io), asFloats(other), 2 * count);
}

void addSaturated(std::int16_t* io, const std::int16_t* other, std::size_t count) noexcept {
    kernels().addSaturated(io, other, count);
}

void scale(float* io, float factor, std::size_t count) noexcept {
    kernels().scale(io, factor, count);
}

void scale(std::complex<float>* io, float factor, std::size_t count) noexcept {
    kernels().scale(asFloats(io), factor, 2 * count);
}

void scaleIQ(std::complex<float>* io, float iFactor, float qFactor, std::size_t count) noexcept {
    kernels().scaleIQ(asFloats(io), iFactor, qFactor, count);
}

void complexMultiply(std::complex<float>* io, const std::complex<float>* other, std::size_t count) noexcept {
    kernels().complexMultiply(asFloats(io), asFloats(other), count);
}

void reverse(const float* in, float* out, std::size_t count) noexcept {
    kernels().reverse(in, out, count);
}

void applyLinearGainRamp(std::complex<float>* io, float startGain, float endGain, std::size_t count) noexcept {
    if (count == 0) {
        return;
    }
    // The step is computed here, not in the kernels, so every instruction set uses the same value.
    const float step = count > 1 ? (endGain - startGain) / static_cast<float>(count - 1) : 0.0F;

    // Kernels index samples with 32-bit integers, so give them at most 2^30 samples at a time.
    constexpr std::size_t kMaxChunk = std::size_t{1} << 30;
    for (std::size_t offset = 0; offset < count; offset += kMaxChunk) {
        const std::size_t chunk = std::min(kMaxChunk, count - offset);
        const float       start = startGain + static_cast<float>(offset) * step;
        kernels().applyLinearGainRamp(asFloats(io + offset), start, step, chunk);
    }
}

void magnitudeSquared(const std::complex<float>* in, float* out, std::size_t count) noexcept {
    kernels().magnitudeSquared(asFloats(in), out, count);
}

void magnitude(const std::complex<float>* in, float* out, std::size_t count) noexcept {
    kernels().magnitude(asFloats(in), out, count);
}

void toPolar(const std::complex<float>* in, float* magnitudeOut, float* phaseOut, std::size_t count) noexcept {
    kernels().toPolar(asFloats(in), magnitudeOut, phaseOut, count);
}

void powerToDecibels(const float* power, float* decibels, std::size_t count) noexcept {
    kernels().powerToDecibels(power, decibels, count);
}

void powerToDecibels(const std::complex<float>* in, float* decibels, std::size_t count) noexcept {
    kernels().complexPowerToDecibels(asFloats(in), decibels, count);
}

// ---------------------------------------------------------------------------------------------
// Span forms: check sizes, then forward to the pointer forms above.
// ---------------------------------------------------------------------------------------------

void int16ToFloat(std::span<const std::int16_t> in, std::span<float> out, float scale) {
    requireSize("int16ToFloat", "out", out.size(), in.size());
    avx_optimized::int16ToFloat(in.data(), out.data(), in.size(), scale);
}

void floatToInt16(std::span<const float> in, std::span<std::int16_t> out, float scale) {
    requireSize("floatToInt16", "out", out.size(), in.size());
    avx_optimized::floatToInt16(in.data(), out.data(), in.size(), scale);
}

void int16ToComplex(std::span<const std::int16_t> interleavedIQ, std::span<std::complex<float>> out, float scale) {
    requireSize("int16ToComplex", "interleavedIQ", interleavedIQ.size(), 2 * out.size());
    avx_optimized::int16ToComplex(interleavedIQ.data(), out.data(), out.size(), scale);
}

void complexToInt16(std::span<const std::complex<float>> in, std::span<std::int16_t> interleavedIQ, float scale) {
    requireSize("complexToInt16", "interleavedIQ", interleavedIQ.size(), 2 * in.size());
    avx_optimized::complexToInt16(in.data(), interleavedIQ.data(), in.size(), scale);
}

void add(std::span<float> io, std::span<const float> other) {
    requireSize("add", "other", other.size(), io.size());
    avx_optimized::add(io.data(), other.data(), io.size());
}

void add(std::span<std::complex<float>> io, std::span<const std::complex<float>> other) {
    requireSize("add", "other", other.size(), io.size());
    avx_optimized::add(io.data(), other.data(), io.size());
}

void addSaturated(std::span<std::int16_t> io, std::span<const std::int16_t> other) {
    requireSize("addSaturated", "other", other.size(), io.size());
    avx_optimized::addSaturated(io.data(), other.data(), io.size());
}

void scale(std::span<float> io, float factor) noexcept {
    avx_optimized::scale(io.data(), factor, io.size());
}

void scale(std::span<std::complex<float>> io, float factor) noexcept {
    avx_optimized::scale(io.data(), factor, io.size());
}

void scaleIQ(std::span<std::complex<float>> io, float iFactor, float qFactor) noexcept {
    avx_optimized::scaleIQ(io.data(), iFactor, qFactor, io.size());
}

void complexMultiply(std::span<std::complex<float>> io, std::span<const std::complex<float>> other) {
    requireSize("complexMultiply", "other", other.size(), io.size());
    avx_optimized::complexMultiply(io.data(), other.data(), io.size());
}

void reverse(std::span<const float> in, std::span<float> out) {
    requireSize("reverse", "out", out.size(), in.size());
    avx_optimized::reverse(in.data(), out.data(), in.size());
}

void applyLinearGainRamp(std::span<std::complex<float>> io, float startGain, float endGain) noexcept {
    avx_optimized::applyLinearGainRamp(io.data(), startGain, endGain, io.size());
}

void magnitudeSquared(std::span<const std::complex<float>> in, std::span<float> out) {
    requireSize("magnitudeSquared", "out", out.size(), in.size());
    avx_optimized::magnitudeSquared(in.data(), out.data(), in.size());
}

void magnitude(std::span<const std::complex<float>> in, std::span<float> out) {
    requireSize("magnitude", "out", out.size(), in.size());
    avx_optimized::magnitude(in.data(), out.data(), in.size());
}

void toPolar(std::span<const std::complex<float>> in, std::span<float> magnitudeOut, std::span<float> phaseOut) {
    requireSize("toPolar", "magnitude", magnitudeOut.size(), in.size());
    requireSize("toPolar", "phase", phaseOut.size(), in.size());
    avx_optimized::toPolar(in.data(), magnitudeOut.data(), phaseOut.data(), in.size());
}

void powerToDecibels(std::span<const float> power, std::span<float> decibels) {
    requireSize("powerToDecibels", "decibels", decibels.size(), power.size());
    avx_optimized::powerToDecibels(power.data(), decibels.data(), power.size());
}

void powerToDecibels(std::span<const std::complex<float>> in, std::span<float> decibels) {
    requireSize("powerToDecibels", "decibels", decibels.size(), in.size());
    avx_optimized::powerToDecibels(in.data(), decibels.data(), in.size());
}

} // namespace avx_optimized
