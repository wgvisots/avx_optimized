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
#include <cstdint>
#include <span>

#include "avx_optimized/export.h"

/**
 * @file
 * Vectorized array operations.
 *
 * Conventions shared by every function here:
 *  - Every function comes in two forms.  The span form takes std::span arguments, so std::vector,
 *    AlignedVector, std::array, C arrays and subspans can be passed directly; their sizes are
 *    checked (see the end of this file).  The pointer form takes raw pointers and a `count`.
 *  - In the pointer form, `count` is always the **last** parameter and counts *elements of the
 *    pointed-to type*: floats for float arrays, complex samples for complex arrays.  The I/Q
 *    conversions (int16ToComplex(), complexToInt16()) count complex samples too.
 *  - Any `count` is fine, including 0 (pointers may then be null).  Otherwise every pointer must be
 *    valid for `count` elements.  No alignment is required; see aligned.h for why you may still
 *    want it.
 *  - In-place functions take `io`.  A second input to an in-place function (`other`) may be the
 *    same pointer as `io`, but must not partially overlap it.  For out-of-place functions, inputs
 *    and outputs must not overlap at all.
 *  - Complex data is std::complex<float>, stored as {real, imag}.  Buffers of interleaved I/Q
 *    floats can be viewed as complex samples with asComplex(), and back with asFloats().
 *  - Every function is safe to call from several threads at once (on different data).
 *  - Results may differ in the last bit or two between instruction sets (FMA contraction,
 *    vectorized log and atan2), except for functions documented as exact.
 *  - The implementation is chosen at runtime; see cpu.h.
 */
namespace avx_optimized {

// ---------------------------------------------------------------------------------------------
// Interleaved I/Q buffers
// ---------------------------------------------------------------------------------------------

/**
 * View a buffer of interleaved I/Q floats (I, Q, I, Q, ...) as complex samples.  Sample k is
 * floats 2k and 2k+1, so a buffer of 2n floats holds n samples.
 *
 * The C++ standard guarantees the opposite direction (std::complex<float> is laid out as float[2];
 * see asFloats()).  This direction is what every C++ signal-processing library relies on and works
 * with all mainstream compilers, but strictly it is outside the standard's guarantees.
 */
inline std::complex<float>* asComplex(float* interleaved) noexcept {
    return reinterpret_cast<std::complex<float>*>(interleaved);
}

/// @copydoc asComplex(float*)
inline const std::complex<float>* asComplex(const float* interleaved) noexcept {
    return reinterpret_cast<const std::complex<float>*>(interleaved);
}

/// View complex samples as interleaved I/Q floats (2 floats per sample).  Guaranteed by the standard.
inline float* asFloats(std::complex<float>* samples) noexcept {
    return reinterpret_cast<float*>(samples);
}

/// @copydoc asFloats(std::complex<float>*)
inline const float* asFloats(const std::complex<float>* samples) noexcept {
    return reinterpret_cast<const float*>(samples);
}

// ---------------------------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------------------------

/**
 * out[i] = in[i] * scale.  Exact (identical on every instruction set).
 *
 * Use scale = 1.0F / 32768.0F to map the int16 range onto [-1, 1).
 */
AVX_OPTIMIZED_EXPORT void
int16ToFloat(const std::int16_t* in, float* out, std::size_t count, float scale = 1.0F) noexcept;

/**
 * out[i] = in[i] * scale, rounded to the nearest integer and saturated to [-32768, 32767].
 * NaN becomes 0.  Exact (identical on every instruction set).
 *
 * Rounding follows the current floating-point rounding mode, which defaults to round half to even
 * (2.5 -> 2, 3.5 -> 4).  Use scale = 32767.0F to map [-1, 1] onto the int16 range.  Note that
 * dividing by 32768 on the way in and multiplying by 32767 on the way out does not round-trip
 * exactly (32767 -> 0.99997 -> 32766); use the same factor in both directions if that matters.
 */
AVX_OPTIMIZED_EXPORT void
floatToInt16(const float* in, std::int16_t* out, std::size_t count, float scale = 1.0F) noexcept;

/**
 * Convert interleaved int16 I/Q samples (I, Q, I, Q, ...), as delivered by most radios, to complex
 * samples, multiplying by `scale`.  `count` is the number of complex samples, so `interleavedIQ`
 * holds 2 * count values.  Exact.
 *
 * Typical use: int16ToComplex(raw, samples, n, 1.0F / 32768.0F) gives samples in [-1, 1).
 */
AVX_OPTIMIZED_EXPORT void int16ToComplex(const std::int16_t*  interleavedIQ,
                                         std::complex<float>* out,
                                         std::size_t          count,
                                         float                scale = 1.0F) noexcept;

/**
 * Convert complex samples to interleaved int16 I/Q, multiplying by `scale` and rounding and
 * saturating as floatToInt16() does.  `count` is the number of complex samples, so
 * `interleavedIQ` receives 2 * count values.  Exact.
 *
 * Typical use: complexToInt16(samples, raw, n, 32767.0F) for samples in [-1, 1].
 */
AVX_OPTIMIZED_EXPORT void complexToInt16(const std::complex<float>* in,
                                         std::int16_t*              interleavedIQ,
                                         std::size_t                count,
                                         float                      scale = 1.0F) noexcept;

// ---------------------------------------------------------------------------------------------
// Element-wise arithmetic
// ---------------------------------------------------------------------------------------------

/// io[i] += other[i].  Exact.
AVX_OPTIMIZED_EXPORT void add(float* io, const float* other, std::size_t count) noexcept;

/// io[i] += other[i] for complex samples.  Exact.
AVX_OPTIMIZED_EXPORT void add(std::complex<float>* io, const std::complex<float>* other, std::size_t count) noexcept;

/// io[i] = saturate(io[i] + other[i]) to [-32768, 32767].  Exact.
AVX_OPTIMIZED_EXPORT void addSaturated(std::int16_t* io, const std::int16_t* other, std::size_t count) noexcept;

/// io[i] *= factor.  Exact.
AVX_OPTIMIZED_EXPORT void scale(float* io, float factor, std::size_t count) noexcept;

/// io[i] *= factor for complex samples (both parts scaled by the same real factor).  Exact.
AVX_OPTIMIZED_EXPORT void scale(std::complex<float>* io, float factor, std::size_t count) noexcept;

/// Scale the real (I) and imaginary (Q) parts separately: io[i] = {re * iFactor, im * qFactor}.  Exact.
AVX_OPTIMIZED_EXPORT void scaleIQ(std::complex<float>* io, float iFactor, float qFactor, std::size_t count) noexcept;

/// io[i] *= other[i] (complex multiplication).
AVX_OPTIMIZED_EXPORT void
complexMultiply(std::complex<float>* io, const std::complex<float>* other, std::size_t count) noexcept;

/// out[count - 1 - i] = in[i].  `in` and `out` must not overlap.  Exact.
AVX_OPTIMIZED_EXPORT void reverse(const float* in, float* out, std::size_t count) noexcept;

/**
 * Multiply each sample by a gain that changes linearly from `startGain` (applied to the first
 * sample) to `endGain` (applied to the last):
 *
 *     io[k] *= startGain + (endGain - startGain) * k / (count - 1)
 *
 * With count == 1 the single sample gets startGain.  The usual use is undoing free-space path
 * loss that changes across a capture.  Gains scale amplitudes: to apply a power ratio, pass its
 * square root.
 */
AVX_OPTIMIZED_EXPORT void
applyLinearGainRamp(std::complex<float>* io, float startGain, float endGain, std::size_t count) noexcept;

// ---------------------------------------------------------------------------------------------
// Magnitude, phase and power
// ---------------------------------------------------------------------------------------------

/**
 * out[i] = |in[i]|^2 = re^2 + im^2 (signal power).
 *
 * Note: computed directly rather than with hypot-style scaling, so components beyond about
 * 1.8e19 overflow to infinity.
 */
AVX_OPTIMIZED_EXPORT void magnitudeSquared(const std::complex<float>* in, float* out, std::size_t count) noexcept;

/// out[i] = |in[i]| = sqrt(re^2 + im^2).  Same overflow caveat as magnitudeSquared().
AVX_OPTIMIZED_EXPORT void magnitude(const std::complex<float>* in, float* out, std::size_t count) noexcept;

/**
 * Convert to polar form: magnitude[i] = |in[i]|, phase[i] = atan2(im, re) in radians, in [-pi, pi]
 * (the same convention as std::arg).
 *
 * The phase is accurate to a few float ulps on every instruction set; special cases (signed zeros,
 * infinities) follow std::atan2.
 */
AVX_OPTIMIZED_EXPORT void
toPolar(const std::complex<float>* in, float* magnitude, float* phase, std::size_t count) noexcept;

/**
 * Convert power values to decibels: decibels[i] = 10 * log10(power[i]).
 *
 * 0 gives -infinity, +infinity gives +infinity, and negative values and NaN give NaN.
 * Subnormal values are converted accurately (unless the CPU's denormals-are-zero mode is enabled,
 * in which case they count as 0 and give -infinity).  For amplitude ratios (20 * log10), convert
 * magnitudeSquared() instead.  Accurate to about 1e-5 dB.
 */
AVX_OPTIMIZED_EXPORT void powerToDecibels(const float* power, float* decibels, std::size_t count) noexcept;

/// The power of each sample in decibels: decibels[i] = 10 * log10(|in[i]|^2).  See the float overload.
AVX_OPTIMIZED_EXPORT void powerToDecibels(const std::complex<float>* in, float* decibels, std::size_t count) noexcept;

// ---------------------------------------------------------------------------------------------
// Span forms
//
// The same functions taking std::span, with the element counts taken from the spans.  Results are
// identical to the pointer forms.  Where a function takes several arrays, their sizes must match
// as stated; otherwise std::invalid_argument is thrown, naming the function and both sizes, and
// nothing is written.  Functions taking a single array cannot mismatch and do not throw.
// ---------------------------------------------------------------------------------------------

/// int16ToFloat() over spans; `out` must be the same size as `in`.
AVX_OPTIMIZED_EXPORT void int16ToFloat(std::span<const std::int16_t> in, std::span<float> out, float scale = 1.0F);

/// floatToInt16() over spans; `out` must be the same size as `in`.
AVX_OPTIMIZED_EXPORT void floatToInt16(std::span<const float> in, std::span<std::int16_t> out, float scale = 1.0F);

/// int16ToComplex() over spans; `interleavedIQ` must hold exactly 2 values per sample of `out`.
AVX_OPTIMIZED_EXPORT void
int16ToComplex(std::span<const std::int16_t> interleavedIQ, std::span<std::complex<float>> out, float scale = 1.0F);

/// complexToInt16() over spans; `interleavedIQ` must have room for exactly 2 values per sample of `in`.
AVX_OPTIMIZED_EXPORT void
complexToInt16(std::span<const std::complex<float>> in, std::span<std::int16_t> interleavedIQ, float scale = 1.0F);

/// add() over spans; `other` must be the same size as `io`.
AVX_OPTIMIZED_EXPORT void add(std::span<float> io, std::span<const float> other);

/// add() of complex samples over spans; `other` must be the same size as `io`.
AVX_OPTIMIZED_EXPORT void add(std::span<std::complex<float>> io, std::span<const std::complex<float>> other);

/// addSaturated() over spans; `other` must be the same size as `io`.
AVX_OPTIMIZED_EXPORT void addSaturated(std::span<std::int16_t> io, std::span<const std::int16_t> other);

/// scale() over a span.
AVX_OPTIMIZED_EXPORT void scale(std::span<float> io, float factor) noexcept;

/// scale() of complex samples over a span.
AVX_OPTIMIZED_EXPORT void scale(std::span<std::complex<float>> io, float factor) noexcept;

/// scaleIQ() over a span.
AVX_OPTIMIZED_EXPORT void scaleIQ(std::span<std::complex<float>> io, float iFactor, float qFactor) noexcept;

/// complexMultiply() over spans; `other` must be the same size as `io`.
AVX_OPTIMIZED_EXPORT void complexMultiply(std::span<std::complex<float>>       io,
                                          std::span<const std::complex<float>> other);

/// reverse() over spans; `out` must be the same size as `in`.
AVX_OPTIMIZED_EXPORT void reverse(std::span<const float> in, std::span<float> out);

/// applyLinearGainRamp() over a span: startGain on the first sample, endGain on the last.
AVX_OPTIMIZED_EXPORT void
applyLinearGainRamp(std::span<std::complex<float>> io, float startGain, float endGain) noexcept;

/// magnitudeSquared() over spans; `out` must be the same size as `in`.
AVX_OPTIMIZED_EXPORT void magnitudeSquared(std::span<const std::complex<float>> in, std::span<float> out);

/// magnitude() over spans; `out` must be the same size as `in`.
AVX_OPTIMIZED_EXPORT void magnitude(std::span<const std::complex<float>> in, std::span<float> out);

/// toPolar() over spans; `magnitude` and `phase` must both be the same size as `in`.
AVX_OPTIMIZED_EXPORT void
toPolar(std::span<const std::complex<float>> in, std::span<float> magnitude, std::span<float> phase);

/// powerToDecibels() of power values over spans; `decibels` must be the same size as `power`.
AVX_OPTIMIZED_EXPORT void powerToDecibels(std::span<const float> power, std::span<float> decibels);

/// powerToDecibels() of complex samples over spans; `decibels` must be the same size as `in`.
AVX_OPTIMIZED_EXPORT void powerToDecibels(std::span<const std::complex<float>> in, std::span<float> decibels);

} // namespace avx_optimized
