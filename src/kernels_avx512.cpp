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

// AVX-512 kernels (F + BW + DQ + VL, i.e. the x86-64-v4 level).  This file is compiled with those
// extensions enabled and is only called after runtime detection confirms the CPU (and OS) support
// all of them.  Follow the ISA translation-unit rules in kernels.h.  The structure mirrors
// kernels_avx2.cpp, so the two can be read side by side.
//
// Partial blocks at the end of an array use masked loads and stores, which never fault on the
// masked-off elements, so each element goes through the same arithmetic wherever it sits.

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include <mathfun/avx512_mathfun.h>

#include "kernels.h"

namespace avx_optimized::detail {
namespace {

constexpr std::size_t kFloats = 16; // floats per __m512

// ---------------------------------------------------------------------------------------------
// Masks and partial loads/stores
// ---------------------------------------------------------------------------------------------

/// Mask with the first `count` lanes enabled; count >= 16 enables all.
inline __mmask16 mask16(std::size_t count) noexcept {
    return count >= 16 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>((1U << count) - 1U);
}

/// Mask with the first `count` lanes enabled; count >= 32 enables all.
inline __mmask32 mask32(std::size_t count) noexcept {
    return count >= 32 ? static_cast<__mmask32>(0xFFFFFFFFU) : static_cast<__mmask32>((1U << count) - 1U);
}

/// Load `count` (0..16) floats from p; other lanes are zero.
inline __m512 load(const float* p, std::size_t count) noexcept {
    return _mm512_maskz_loadu_ps(mask16(count), p);
}

/// Store the first `count` (0..16) lanes of v to p.
inline void store(float* p, std::size_t count, __m512 v) noexcept {
    _mm512_mask_storeu_ps(p, mask16(count), v);
}

/// Elements left in a loop that steps by `step`: min(count - i, step).
inline std::size_t blockSize(std::size_t count, std::size_t i, std::size_t step) noexcept {
    return count - i < step ? count - i : step;
}

// ---------------------------------------------------------------------------------------------
// Loop skeletons
// ---------------------------------------------------------------------------------------------

/// io[i] = op(io[i], other[i]) for `count` floats, 16 at a time.
template <typename Op>
inline void binaryInPlace(float* io, const float* other, std::size_t count, Op op) noexcept {
    std::size_t i = 0;
    for (; i + 4 * kFloats <= count; i += 4 * kFloats) {
        const __m512 r0 = op(_mm512_loadu_ps(io + i), _mm512_loadu_ps(other + i));
        const __m512 r1 = op(_mm512_loadu_ps(io + i + 16), _mm512_loadu_ps(other + i + 16));
        const __m512 r2 = op(_mm512_loadu_ps(io + i + 32), _mm512_loadu_ps(other + i + 32));
        const __m512 r3 = op(_mm512_loadu_ps(io + i + 48), _mm512_loadu_ps(other + i + 48));

        _mm512_storeu_ps(io + i, r0);
        _mm512_storeu_ps(io + i + 16, r1);
        _mm512_storeu_ps(io + i + 32, r2);
        _mm512_storeu_ps(io + i + 48, r3);
    }
    for (; i < count; i += kFloats) {
        const std::size_t n = blockSize(count, i, kFloats);
        store(io + i, n, op(load(io + i, n), load(other + i, n)));
    }
}

/// io[i] = op(io[i]) for `count` floats, 16 at a time.
template <typename Op>
inline void unaryInPlace(float* io, std::size_t count, Op op) noexcept {
    std::size_t i = 0;
    for (; i + 4 * kFloats <= count; i += 4 * kFloats) {
        const __m512 r0 = op(_mm512_loadu_ps(io + i));
        const __m512 r1 = op(_mm512_loadu_ps(io + i + 16));
        const __m512 r2 = op(_mm512_loadu_ps(io + i + 32));
        const __m512 r3 = op(_mm512_loadu_ps(io + i + 48));

        _mm512_storeu_ps(io + i, r0);
        _mm512_storeu_ps(io + i + 16, r1);
        _mm512_storeu_ps(io + i + 32, r2);
        _mm512_storeu_ps(io + i + 48, r3);
    }
    for (; i < count; i += kFloats) {
        const std::size_t n = blockSize(count, i, kFloats);
        store(io + i, n, op(load(io + i, n)));
    }
}

/// Load `count` (0..16) complex samples starting at `in`, split into real and imaginary parts, in order.
inline void loadComplex(const float* in, std::size_t count, __m512& re, __m512& im) noexcept {
    const __m512i evenIndex = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i oddIndex  = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    // in + kFloats is only formed when it is inside the array.
    const std::size_t floats = 2 * count;
    const __m512      low    = load(in, floats);
    const __m512      high   = floats > kFloats ? load(in + kFloats, floats - kFloats) : _mm512_setzero_ps();

    re = _mm512_permutex2var_ps(low, evenIndex, high);
    im = _mm512_permutex2var_ps(low, oddIndex, high);
}

/// out[i] = op(re(in[i]), im(in[i])) for `count` complex samples, 16 at a time.
template <typename Op>
inline void complexToReal(const float* in, float* out, std::size_t count, Op op) noexcept {
    for (std::size_t i = 0; i < count; i += kFloats) {
        const std::size_t n = blockSize(count, i, kFloats);

        __m512 re;
        __m512 im;
        loadComplex(in + 2 * i, n, re, im);
        store(out + i, n, op(re, im));
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise math
// ---------------------------------------------------------------------------------------------

inline __m512 magnitudeSquaredOf(__m512 re, __m512 im) noexcept {
    return _mm512_fmadd_ps(re, re, _mm512_mul_ps(im, im));
}

/// (a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re) for 8 interleaved complex pairs.
inline __m512 complexMul(__m512 a, __m512 b) noexcept {
    const __m512 bRe      = _mm512_moveldup_ps(b);      // br br ...
    const __m512 bIm      = _mm512_movehdup_ps(b);      // bi bi ...
    const __m512 aSwapped = _mm512_permute_ps(a, 0xB1); // ai ar ...

    return _mm512_fmaddsub_ps(a, bRe, _mm512_mul_ps(aSwapped, bIm)); // ar*br - ai*bi, ai*br + ar*bi
}

/// 10 * log10(power) with the special cases documented in vectorized.h, computed exactly as in
/// kernels_avx2.cpp (rather than relying on avx512_mathfun's own special-case handling) so the
/// two instruction sets agree.
inline __m512 decibelsOf(__m512 power) noexcept {
    const __m512 zero             = _mm512_setzero_ps();
    const __m512 infinity         = _mm512_set1_ps(__builtin_inff());
    const __m512 negativeInfinity = _mm512_set1_ps(-__builtin_inff());
    const __m512 notANumber       = _mm512_set1_ps(__builtin_nanf(""));
    const __m512 smallestNormal   = _mm512_set1_ps(1.17549435e-38F); // FLT_MIN
    const __m512 twoTo24          = _mm512_set1_ps(16777216.0F);
    const __m512 ln2Times24       = _mm512_set1_ps(16.635532333438687F);
    const __m512 tenOverLn10      = _mm512_set1_ps(4.3429448190325182F);

    // Subnormals: scale by 2^24 (exact) into the normal range, then subtract 24 ln 2 from the log.
    const __mmask16 isPositive  = _mm512_cmp_ps_mask(power, zero, _CMP_GT_OQ);
    const __mmask16 isSubnormal = _mm512_mask_cmp_ps_mask(isPositive, power, smallestNormal, _CMP_LT_OQ);
    const __m512    scaled      = _mm512_mask_mul_ps(power, isSubnormal, power, twoTo24);
    const __m512    rawLog      = avx512_mathfun::log512_ps(scaled);
    const __m512    logE        = _mm512_mask_sub_ps(rawLog, isSubnormal, rawLog, ln2Times24);

    const __mmask16 isZero     = _mm512_cmp_ps_mask(power, zero, _CMP_EQ_OQ);
    const __mmask16 isInfinity = _mm512_cmp_ps_mask(power, infinity, _CMP_EQ_OQ);
    const __mmask16 isInvalid  = _mm512_cmp_ps_mask(power, zero, _CMP_NGE_UQ); // negative or NaN

    __m512 decibels = _mm512_mul_ps(logE, tenOverLn10);
    decibels        = _mm512_mask_mov_ps(decibels, isZero, negativeInfinity);
    decibels        = _mm512_mask_mov_ps(decibels, isInfinity, infinity);
    decibels        = _mm512_mask_mov_ps(decibels, isInvalid, notANumber);
    return decibels;
}

/// 16 floats -> 16 int16: scale, NaN -> 0, clamp to the int16 range, round per MXCSR.
inline __m256i toInt16(__m512 x, __m512 scale) noexcept {
    x = _mm512_mul_ps(x, scale);
    x = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(x, x, _CMP_ORD_Q), x); // NaN -> 0
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-32768.0F)), _mm512_set1_ps(32767.0F));
    return _mm512_cvtepi32_epi16(_mm512_cvtps_epi32(x)); // already in range, so plain narrowing
}

/// 16 int16 -> 16 floats, multiplied by scale.
inline __m512 int16x16ToFloat(__m256i v, __m512 scale) noexcept {
    return _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v)), scale);
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

void int16ToFloat(const std::int16_t* in, float* out, std::size_t count, float scale) noexcept {
    const __m512 vScale = _mm512_set1_ps(scale);

    for (std::size_t i = 0; i < count; i += 16) {
        const __mmask16 m = mask16(count - i);
        _mm512_mask_storeu_ps(out + i, m, int16x16ToFloat(_mm256_maskz_loadu_epi16(m, in + i), vScale));
    }
}

void floatToInt16(const float* in, std::int16_t* out, std::size_t count, float scale) noexcept {
    const __m512 vScale = _mm512_set1_ps(scale);

    for (std::size_t i = 0; i < count; i += 16) {
        const __mmask16 m = mask16(count - i);
        _mm256_mask_storeu_epi16(out + i, m, toInt16(_mm512_maskz_loadu_ps(m, in + i), vScale));
    }
}

void add(float* io, const float* other, std::size_t count) noexcept {
    binaryInPlace(io, other, count, [](__m512 a, __m512 b) noexcept { return _mm512_add_ps(a, b); });
}

void addSaturated(std::int16_t* io, const std::int16_t* other, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; i += 32) {
        const __mmask32 m = mask32(count - i);
        const __m512i   sum =
            _mm512_adds_epi16(_mm512_maskz_loadu_epi16(m, io + i), _mm512_maskz_loadu_epi16(m, other + i));
        _mm512_mask_storeu_epi16(io + i, m, sum);
    }
}

void scale(float* io, float factor, std::size_t count) noexcept {
    const __m512 vFactor = _mm512_set1_ps(factor);
    unaryInPlace(io, count, [vFactor](__m512 v) noexcept { return _mm512_mul_ps(v, vFactor); });
}

void reverse(const float* in, float* out, std::size_t count) noexcept {
    const __m512i backwards = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    std::size_t i = 0;
    for (; i + kFloats <= count; i += kFloats) {
        _mm512_storeu_ps(out + (count - i - kFloats), _mm512_permutexvar_ps(backwards, _mm512_loadu_ps(in + i)));
    }
    for (; i < count; ++i) { // pure copies, so a scalar tail is equivalent
        out[count - 1 - i] = in[i];
    }
}

void scaleIQ(float* io, float iFactor, float qFactor, std::size_t count) noexcept {
    // clang-format off
    const __m512 vFactors = _mm512_setr_ps(iFactor, qFactor, iFactor, qFactor, iFactor, qFactor, iFactor, qFactor,
                                           iFactor, qFactor, iFactor, qFactor, iFactor, qFactor, iFactor, qFactor);
    // clang-format on
    unaryInPlace(io, 2 * count, [vFactors](__m512 v) noexcept { return _mm512_mul_ps(v, vFactors); });
}

void complexMultiply(float* io, const float* other, std::size_t count) noexcept {
    binaryInPlace(io, other, 2 * count, [](__m512 a, __m512 b) noexcept { return complexMul(a, b); });
}

void magnitudeSquared(const float* in, float* out, std::size_t count) noexcept {
    complexToReal(in, out, count, [](__m512 re, __m512 im) noexcept { return magnitudeSquaredOf(re, im); });
}

void magnitude(const float* in, float* out, std::size_t count) noexcept {
    complexToReal(in, out, count, [](__m512 re, __m512 im) noexcept {
        return _mm512_sqrt_ps(magnitudeSquaredOf(re, im));
    });
}

void toPolar(const float* in, float* magnitudeOut, float* phaseOut, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; i += kFloats) {
        const std::size_t n = blockSize(count, i, kFloats);

        __m512 re;
        __m512 im;
        loadComplex(in + 2 * i, n, re, im);
        store(magnitudeOut + i, n, _mm512_sqrt_ps(magnitudeSquaredOf(re, im)));
        store(phaseOut + i, n, avx512_mathfun::atan2(im, re));
    }
}

void applyLinearGainRamp(float* io, float start, float step, std::size_t count) noexcept {
    // Sample index of each float lane: 0 0 1 1 ... 7 7 (re and im share a gain).
    const __m512i laneSample = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const __m512  vStart     = _mm512_set1_ps(start);
    const __m512  vStep      = _mm512_set1_ps(step);

    for (std::size_t i = 0; i < count; i += 8) { // complex samples; 8 per vector
        const std::size_t floats = 2 * blockSize(count, i, 8);
        const __m512i     sample = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(i)), laneSample);
        const __m512      gain   = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sample), vStep, vStart);

        store(io + 2 * i, floats, _mm512_mul_ps(load(io + 2 * i, floats), gain));
    }
}

void powerToDecibels(const float* power, float* decibels, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; i += kFloats) {
        const std::size_t n = blockSize(count, i, kFloats);
        store(decibels + i, n, decibelsOf(load(power + i, n)));
    }
}

void complexPowerToDecibels(const float* in, float* decibels, std::size_t count) noexcept {
    complexToReal(in, decibels, count, [](__m512 re, __m512 im) noexcept {
        return decibelsOf(magnitudeSquaredOf(re, im));
    });
}

/// See firCombine in kernels_avx2.cpp: y.re = acc.re - accIm.im, y.im = acc.im + accIm.re.
inline __m512 firCombine(__m512 acc, __m512 accIm) noexcept {
    return _mm512_fmaddsub_ps(acc, _mm512_set1_ps(1.0F), _mm512_permute_ps(accIm, 0xB1));
}

void complexFir(const float* in,
                const float* tapsReal,
                const float* tapsImag,
                std::size_t  numTaps,
                float*       out,
                std::size_t  count) noexcept {
    std::size_t k = 0; // output sample index; 8 complex samples per vector

    // 32 outputs at a time: 8 independent accumulators hide FMA latency.
    for (; k + 32 <= count; k += 32) {
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        __m512 b0 = a0, b1 = a0, b2 = a0, b3 = a0;

        const float* x = in + 2 * k;
        for (std::size_t j = 0; j < numTaps; ++j) {
            const __m512 hr = _mm512_set1_ps(tapsReal[j]);
            const __m512 hi = _mm512_set1_ps(tapsImag[j]);
            const __m512 x0 = _mm512_loadu_ps(x + 2 * j);
            const __m512 x1 = _mm512_loadu_ps(x + 2 * j + 16);
            const __m512 x2 = _mm512_loadu_ps(x + 2 * j + 32);
            const __m512 x3 = _mm512_loadu_ps(x + 2 * j + 48);

            a0 = _mm512_fmadd_ps(x0, hr, a0);
            b0 = _mm512_fmadd_ps(x0, hi, b0);
            a1 = _mm512_fmadd_ps(x1, hr, a1);
            b1 = _mm512_fmadd_ps(x1, hi, b1);
            a2 = _mm512_fmadd_ps(x2, hr, a2);
            b2 = _mm512_fmadd_ps(x2, hi, b2);
            a3 = _mm512_fmadd_ps(x3, hr, a3);
            b3 = _mm512_fmadd_ps(x3, hi, b3);
        }

        float* y = out + 2 * k;
        _mm512_storeu_ps(y, firCombine(a0, b0));
        _mm512_storeu_ps(y + 16, firCombine(a1, b1));
        _mm512_storeu_ps(y + 32, firCombine(a2, b2));
        _mm512_storeu_ps(y + 48, firCombine(a3, b3));
    }

    // Remaining outputs, 8 at a time; the last block is masked.
    for (; k < count; k += 8) {
        const __mmask16 m = mask16(2 * blockSize(count, k, 8));
        const float*    x = in + 2 * k;

        __m512 a = _mm512_setzero_ps();
        __m512 b = a;
        for (std::size_t j = 0; j < numTaps; ++j) {
            const __m512 xj = _mm512_maskz_loadu_ps(m, x + 2 * j);

            a = _mm512_fmadd_ps(xj, _mm512_set1_ps(tapsReal[j]), a);
            b = _mm512_fmadd_ps(xj, _mm512_set1_ps(tapsImag[j]), b);
        }
        _mm512_mask_storeu_ps(out + 2 * k, m, firCombine(a, b));
    }
}

/// Real-tap FIR: one FMA per tap per vector, half the work of complexFir.
void complexFirRealTaps(const float* in,
                        const float* taps,
                        std::size_t  numTaps,
                        float*       out,
                        std::size_t  count) noexcept {
    std::size_t k = 0; // output sample index; 8 complex samples per vector

    // 64 outputs at a time: 8 independent accumulators hide FMA latency.
    for (; k + 64 <= count; k += 64) {
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        __m512 a4 = a0, a5 = a0, a6 = a0, a7 = a0;

        const float* x = in + 2 * k;
        for (std::size_t j = 0; j < numTaps; ++j) {
            const __m512 h  = _mm512_set1_ps(taps[j]);
            const float* xj = x + 2 * j;

            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(xj), h, a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(xj + 16), h, a1);
            a2 = _mm512_fmadd_ps(_mm512_loadu_ps(xj + 32), h, a2);
            a3 = _mm512_fmadd_ps(_mm512_loadu_ps(xj + 48), h, a3);
            a4 = _mm512_fmadd_ps(_mm512_loadu_ps(xj + 64), h, a4);
            a5 = _mm512_fmadd_ps(_mm512_loadu_ps(xj + 80), h, a5);
            a6 = _mm512_fmadd_ps(_mm512_loadu_ps(xj + 96), h, a6);
            a7 = _mm512_fmadd_ps(_mm512_loadu_ps(xj + 112), h, a7);
        }

        float* y = out + 2 * k;
        _mm512_storeu_ps(y, a0);
        _mm512_storeu_ps(y + 16, a1);
        _mm512_storeu_ps(y + 32, a2);
        _mm512_storeu_ps(y + 48, a3);
        _mm512_storeu_ps(y + 64, a4);
        _mm512_storeu_ps(y + 80, a5);
        _mm512_storeu_ps(y + 96, a6);
        _mm512_storeu_ps(y + 112, a7);
    }

    // Remaining outputs, 8 at a time; the last block is masked.
    for (; k < count; k += 8) {
        const __mmask16 m = mask16(2 * blockSize(count, k, 8));
        const float*    x = in + 2 * k;

        __m512 a = _mm512_setzero_ps();
        for (std::size_t j = 0; j < numTaps; ++j) {
            a = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, x + 2 * j), _mm512_set1_ps(taps[j]), a);
        }
        _mm512_mask_storeu_ps(out + 2 * k, m, a);
    }
}

} // namespace

extern const KernelTable kAvx512Kernels = {
    .isa                    = Isa::Avx512,
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
