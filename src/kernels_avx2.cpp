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

// AVX2 + FMA kernels.  This file is compiled with -mavx2 -mfma and is only called after runtime
// detection confirms the CPU supports both.  Follow the ISA translation-unit rules in kernels.h.
//
// Partial blocks at the end of an array go through the same vector code using masked loads and
// stores, so each element's result does not depend on its position or on the array length.

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include <mathfun/avx_mathfun.h>

#include "kernels.h"

namespace avx_optimized::detail {
namespace {

constexpr std::size_t kFloats = 8; // floats per __m256

// ---------------------------------------------------------------------------------------------
// Partial (masked) loads and stores
// ---------------------------------------------------------------------------------------------

/// Mask with the first `count` (0..8) 32-bit lanes enabled.
inline __m256i laneMask(std::size_t count) noexcept {
    const __m256i laneIndex = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    return _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(count)), laneIndex);
}

/// Load the first `count` (0..8) floats from p; other lanes are zero.  Never touches memory past p[count-1].
inline __m256 loadPartial(const float* p, std::size_t count) noexcept {
    return _mm256_maskload_ps(p, laneMask(count));
}

/// Store the first `count` (0..8) lanes of v to p.
inline void storePartial(float* p, std::size_t count, __m256 v) noexcept {
    _mm256_maskstore_ps(p, laneMask(count), v);
}

/// Load `count` (0..8) floats: a plain load for a full vector, a masked one otherwise.
inline __m256 load(const float* p, std::size_t count) noexcept {
    return count == kFloats ? _mm256_loadu_ps(p) : loadPartial(p, count);
}

/// Store the first `count` (0..8) lanes of v: a plain store for a full vector, a masked one otherwise.
inline void store(float* p, std::size_t count, __m256 v) noexcept {
    if (count == kFloats) {
        _mm256_storeu_ps(p, v);
    } else {
        storePartial(p, count, v);
    }
}

/// For a partial block of `count` (0..16) floats held in two vectors: how many are in the first.
inline std::size_t lowPart(std::size_t count) noexcept {
    return count < kFloats ? count : kFloats;
}

/// Load floats 8..count-1 of a partial 16-float block starting at p (zeros if count <= 8).  The
/// pointer p + 8 is only formed when it is inside the array.
inline __m256 loadHigh(const float* p, std::size_t count) noexcept {
    return count > kFloats ? load(p + kFloats, count - kFloats) : _mm256_setzero_ps();
}

/// Store lanes of v as floats 8..count-1 of a partial 16-float block starting at p.
inline void storeHigh(float* p, std::size_t count, __m256 v) noexcept {
    if (count > kFloats) {
        store(p + kFloats, count - kFloats, v);
    }
}

/// Elements left in a loop that steps by `step`: min(count - i, step).
inline std::size_t blockSize(std::size_t count, std::size_t i, std::size_t step) noexcept {
    return count - i < step ? count - i : step;
}

// ---------------------------------------------------------------------------------------------
// Loop skeletons
// ---------------------------------------------------------------------------------------------

/// io[i] = op(io[i], other[i]) for `count` floats, 8 at a time.
template <typename Op>
inline void binaryInPlace(float* io, const float* other, std::size_t count, Op op) noexcept {
    std::size_t i = 0;
    for (; i + 4 * kFloats <= count; i += 4 * kFloats) {
        const __m256 r0 = op(_mm256_loadu_ps(io + i), _mm256_loadu_ps(other + i));
        const __m256 r1 = op(_mm256_loadu_ps(io + i + 8), _mm256_loadu_ps(other + i + 8));
        const __m256 r2 = op(_mm256_loadu_ps(io + i + 16), _mm256_loadu_ps(other + i + 16));
        const __m256 r3 = op(_mm256_loadu_ps(io + i + 24), _mm256_loadu_ps(other + i + 24));

        _mm256_storeu_ps(io + i, r0);
        _mm256_storeu_ps(io + i + 8, r1);
        _mm256_storeu_ps(io + i + 16, r2);
        _mm256_storeu_ps(io + i + 24, r3);
    }
    for (; i < count; i += kFloats) {
        const std::size_t n = blockSize(count, i, kFloats);
        store(io + i, n, op(load(io + i, n), load(other + i, n)));
    }
}

/// io[i] = op(io[i]) for `count` floats, 8 at a time.
template <typename Op>
inline void unaryInPlace(float* io, std::size_t count, Op op) noexcept {
    std::size_t i = 0;
    for (; i + 4 * kFloats <= count; i += 4 * kFloats) {
        const __m256 r0 = op(_mm256_loadu_ps(io + i));
        const __m256 r1 = op(_mm256_loadu_ps(io + i + 8));
        const __m256 r2 = op(_mm256_loadu_ps(io + i + 16));
        const __m256 r3 = op(_mm256_loadu_ps(io + i + 24));

        _mm256_storeu_ps(io + i, r0);
        _mm256_storeu_ps(io + i + 8, r1);
        _mm256_storeu_ps(io + i + 16, r2);
        _mm256_storeu_ps(io + i + 24, r3);
    }
    for (; i < count; i += kFloats) {
        const std::size_t n = blockSize(count, i, kFloats);
        store(io + i, n, op(load(io + i, n)));
    }
}

/*
 * Split 8 interleaved complex values (a = c0..c3, b = c4..c7) into real and imaginary parts.
 * shuffle_ps works within 128-bit lanes, so the parts come out in the order
 * c0 c1 c4 c5 c2 c3 c6 c7.  Element-wise math does not care; restoreOrder() fixes the result.
 */
inline void deinterleave(__m256 a, __m256 b, __m256& re, __m256& im) noexcept {
    re = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    im = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
}

inline __m256 restoreOrder(__m256 v) noexcept {
    return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(v), _MM_SHUFFLE(3, 1, 2, 0)));
}

/// Load 8 complex samples starting at `in`, split into real and imaginary parts (in
/// deinterleave()'s lane order).
inline void loadComplex(const float* in, __m256& re, __m256& im) noexcept {
    deinterleave(_mm256_loadu_ps(in), _mm256_loadu_ps(in + 8), re, im);
}

/// As loadComplex(), for the last `count` (1..7) samples of an array.
inline void loadComplexPartial(const float* in, std::size_t count, __m256& re, __m256& im) noexcept {
    const std::size_t floats = 2 * count;
    deinterleave(loadPartial(in, lowPart(floats)), loadHigh(in, floats), re, im);
}

/// out[i] = op(re(in[i]), im(in[i])) for `count` complex samples, 8 at a time.
template <typename Op>
inline void complexToReal(const float* in, float* out, std::size_t count, Op op) noexcept {
    __m256      re;
    __m256      im;
    std::size_t i = 0;
    for (; i + kFloats <= count; i += kFloats) {
        loadComplex(in + 2 * i, re, im);
        _mm256_storeu_ps(out + i, restoreOrder(op(re, im)));
    }
    if (i < count) {
        loadComplexPartial(in + 2 * i, count - i, re, im);
        storePartial(out + i, count - i, restoreOrder(op(re, im)));
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise math
// ---------------------------------------------------------------------------------------------

inline __m256 magnitudeSquaredOf(__m256 re, __m256 im) noexcept {
    return _mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im));
}

/// (a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re) for 4 interleaved complex pairs.
inline __m256 complexMul(__m256 a, __m256 b) noexcept {
    const __m256 bRe      = _mm256_moveldup_ps(b);      // br br ...
    const __m256 bIm      = _mm256_movehdup_ps(b);      // bi bi ...
    const __m256 aSwapped = _mm256_permute_ps(a, 0xB1); // ai ar ...

    return _mm256_fmaddsub_ps(a, bRe, _mm256_mul_ps(aSwapped, bIm)); // ar*br - ai*bi, ai*br + ar*bi
}

/// 10 * log10(power) with the special cases documented in vectorized.h.  avx_mathfun's log
/// flushes subnormals to FLT_MIN and does not handle 0 or infinity, so those are handled here.
inline __m256 decibelsOf(__m256 power) noexcept {
    const __m256 zero             = _mm256_setzero_ps();
    const __m256 infinity         = _mm256_set1_ps(__builtin_inff());
    const __m256 negativeInfinity = _mm256_set1_ps(-__builtin_inff());
    const __m256 notANumber       = _mm256_set1_ps(__builtin_nanf(""));
    const __m256 smallestNormal   = _mm256_set1_ps(1.17549435e-38F); // FLT_MIN
    const __m256 twoTo24          = _mm256_set1_ps(16777216.0F);
    const __m256 ln2Times24       = _mm256_set1_ps(16.635532333438687F);
    const __m256 tenOverLn10      = _mm256_set1_ps(4.3429448190325182F);

    // Subnormals: scale by 2^24 (exact) into the normal range, then subtract 24 ln 2 from the log.
    const __m256 isBelowNormal = _mm256_cmp_ps(power, smallestNormal, _CMP_LT_OQ);
    const __m256 isPositive    = _mm256_cmp_ps(power, zero, _CMP_GT_OQ);
    const __m256 isSubnormal   = _mm256_and_ps(isBelowNormal, isPositive);
    const __m256 scaled        = _mm256_blendv_ps(power, _mm256_mul_ps(power, twoTo24), isSubnormal);
    const __m256 logE          = _mm256_sub_ps(avx_mathfun::log256_ps(scaled), _mm256_and_ps(isSubnormal, ln2Times24));

    const __m256 isZero     = _mm256_cmp_ps(power, zero, _CMP_EQ_OQ);
    const __m256 isInfinity = _mm256_cmp_ps(power, infinity, _CMP_EQ_OQ);
    const __m256 isInvalid  = _mm256_cmp_ps(power, zero, _CMP_NGE_UQ); // negative or NaN

    __m256 decibels = _mm256_mul_ps(logE, tenOverLn10);
    decibels        = _mm256_blendv_ps(decibels, negativeInfinity, isZero);
    decibels        = _mm256_blendv_ps(decibels, infinity, isInfinity);
    decibels        = _mm256_blendv_ps(decibels, notANumber, isInvalid);
    return decibels;
}

/// Round to int32 after scaling, with NaN -> 0 and the value clamped to the int16 range first.
inline __m256i toClampedInt32(__m256 x, __m256 scale) noexcept {
    x = _mm256_mul_ps(x, scale);
    x = _mm256_and_ps(x, _mm256_cmp_ps(x, x, _CMP_ORD_Q)); // NaN -> 0
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-32768.0F)), _mm256_set1_ps(32767.0F));
    return _mm256_cvtps_epi32(x);
}

/// 16 floats -> 16 int16 in order.
inline __m256i toInt16(__m256 a, __m256 b, __m256 scale) noexcept {
    // packs works per 128-bit lane, giving a0-3 b0-3 a4-7 b4-7; the permute restores the order.
    const __m256i packed = _mm256_packs_epi32(toClampedInt32(a, scale), toClampedInt32(b, scale));
    return _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
}

/// 16 int16 -> 16 floats (in two vectors), multiplied by scale.
inline void int16x16ToFloat(__m256i v, __m256 scale, __m256& low, __m256& high) noexcept {
    const __m256i lowInts  = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
    const __m256i highInts = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));

    low  = _mm256_mul_ps(_mm256_cvtepi32_ps(lowInts), scale);
    high = _mm256_mul_ps(_mm256_cvtepi32_ps(highInts), scale);
}

// ---------------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------------

void int16ToFloat(const std::int16_t* in, float* out, std::size_t count, float scale) noexcept {
    const __m256 vScale = _mm256_set1_ps(scale);

    __m256      low;
    __m256      high;
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        int16x16ToFloat(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i)), vScale, low, high);
        _mm256_storeu_ps(out + i, low);
        _mm256_storeu_ps(out + i + 8, high);
    }
    if (i < count) {
        const std::size_t rest = count - i;

        alignas(32) std::int16_t buffer[16] = {};
        for (std::size_t k = 0; k < rest; ++k) {
            buffer[k] = in[i + k];
        }
        int16x16ToFloat(_mm256_load_si256(reinterpret_cast<const __m256i*>(buffer)), vScale, low, high);
        store(out + i, lowPart(rest), low);
        storeHigh(out + i, rest, high);
    }
}

void floatToInt16(const float* in, std::int16_t* out, std::size_t count, float scale) noexcept {
    const __m256 vScale = _mm256_set1_ps(scale);

    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i v = toInt16(_mm256_loadu_ps(in + i), _mm256_loadu_ps(in + i + 8), vScale);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), v);
    }
    if (i < count) {
        const std::size_t rest = count - i;
        const __m256i     v    = toInt16(load(in + i, lowPart(rest)), loadHigh(in + i, rest), vScale);

        alignas(32) std::int16_t buffer[16];
        _mm256_store_si256(reinterpret_cast<__m256i*>(buffer), v);
        for (std::size_t k = 0; k < rest; ++k) {
            out[i + k] = buffer[k];
        }
    }
}

void add(float* io, const float* other, std::size_t count) noexcept {
    binaryInPlace(io, other, count, [](__m256 a, __m256 b) noexcept { return _mm256_add_ps(a, b); });
}

void addSaturated(std::int16_t* io, const std::int16_t* other, std::size_t count) noexcept {
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        auto*       p = reinterpret_cast<__m256i*>(io + i);
        const auto* q = reinterpret_cast<const __m256i*>(other + i);
        _mm256_storeu_si256(p, _mm256_adds_epi16(_mm256_loadu_si256(p), _mm256_loadu_si256(q)));
    }
    for (; i < count; ++i) { // integer math is exact, so a scalar tail is equivalent
        int sum = static_cast<int>(io[i]) + static_cast<int>(other[i]);
        sum     = sum > 32767 ? 32767 : sum;
        sum     = sum < -32768 ? -32768 : sum;
        io[i]   = static_cast<std::int16_t>(sum);
    }
}

void scale(float* io, float factor, std::size_t count) noexcept {
    const __m256 vFactor = _mm256_set1_ps(factor);
    unaryInPlace(io, count, [vFactor](__m256 v) noexcept { return _mm256_mul_ps(v, vFactor); });
}

void reverse(const float* in, float* out, std::size_t count) noexcept {
    const __m256i backwards = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    std::size_t i = 0;
    for (; i + kFloats <= count; i += kFloats) {
        _mm256_storeu_ps(out + (count - i - kFloats), _mm256_permutevar8x32_ps(_mm256_loadu_ps(in + i), backwards));
    }
    for (; i < count; ++i) { // pure copies, so a scalar tail is equivalent
        out[count - 1 - i] = in[i];
    }
}

void scaleIQ(float* io, float iFactor, float qFactor, std::size_t count) noexcept {
    const __m256 vFactors = _mm256_setr_ps(iFactor, qFactor, iFactor, qFactor, iFactor, qFactor, iFactor, qFactor);
    unaryInPlace(io, 2 * count, [vFactors](__m256 v) noexcept { return _mm256_mul_ps(v, vFactors); });
}

void complexMultiply(float* io, const float* other, std::size_t count) noexcept {
    binaryInPlace(io, other, 2 * count, [](__m256 a, __m256 b) noexcept { return complexMul(a, b); });
}

void magnitudeSquared(const float* in, float* out, std::size_t count) noexcept {
    complexToReal(in, out, count, [](__m256 re, __m256 im) noexcept { return magnitudeSquaredOf(re, im); });
}

void magnitude(const float* in, float* out, std::size_t count) noexcept {
    complexToReal(in, out, count, [](__m256 re, __m256 im) noexcept {
        return _mm256_sqrt_ps(magnitudeSquaredOf(re, im));
    });
}

void toPolar(const float* in, float* magnitudeOut, float* phaseOut, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; i += kFloats) {
        const std::size_t n = blockSize(count, i, kFloats);

        __m256 re;
        __m256 im;
        if (n == kFloats) {
            loadComplex(in + 2 * i, re, im);
        } else {
            loadComplexPartial(in + 2 * i, n, re, im);
        }
        store(magnitudeOut + i, n, restoreOrder(_mm256_sqrt_ps(magnitudeSquaredOf(re, im))));
        store(phaseOut + i, n, restoreOrder(avx_mathfun::atan2(im, re))); // atan2 dominates; branches are free here
    }
}

void applyLinearGainRamp(float* io, float start, float step, std::size_t count) noexcept {
    // Sample index of each float lane: 0 0 1 1 2 2 3 3 (re and im share a gain).
    const __m256i laneSample = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const __m256  vStart     = _mm256_set1_ps(start);
    const __m256  vStep      = _mm256_set1_ps(step);

    const auto gainAt = [&](std::size_t firstSample) noexcept {
        const __m256i sample = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(firstSample)), laneSample);
        return _mm256_fmadd_ps(_mm256_cvtepi32_ps(sample), vStep, vStart);
    };

    std::size_t i = 0; // complex samples; 4 per vector
    for (; i + 8 <= count; i += 8) {
        const __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(io + 2 * i), gainAt(i));
        const __m256 v1 = _mm256_mul_ps(_mm256_loadu_ps(io + 2 * i + 8), gainAt(i + 4));

        _mm256_storeu_ps(io + 2 * i, v0);
        _mm256_storeu_ps(io + 2 * i + 8, v1);
    }
    for (; i < count; i += 4) {
        const std::size_t floats = 2 * blockSize(count, i, 4);
        store(io + 2 * i, floats, _mm256_mul_ps(load(io + 2 * i, floats), gainAt(i)));
    }
}

void powerToDecibels(const float* power, float* decibels, std::size_t count) noexcept {
    std::size_t i = 0;
    for (; i + kFloats <= count; i += kFloats) {
        _mm256_storeu_ps(decibels + i, decibelsOf(_mm256_loadu_ps(power + i)));
    }
    if (i < count) {
        storePartial(decibels + i, count - i, decibelsOf(loadPartial(power + i, count - i)));
    }
}

void complexPowerToDecibels(const float* in, float* decibels, std::size_t count) noexcept {
    complexToReal(in, decibels, count, [](__m256 re, __m256 im) noexcept {
        return decibelsOf(magnitudeSquaredOf(re, im));
    });
}

/*
 * Complex-tap FIR, 4 output samples per vector.  For every tap j, multiply the input window
 * starting at sample k+j by the broadcast tap, keeping two accumulators: acc = sum x*h.re and
 * accIm = sum x*h.im.  The complex product is then assembled once at the end:
 *     y.re = acc.re - accIm.im,   y.im = acc.im + accIm.re
 * which costs just 2 FMAs per tap per vector in the inner loop.
 */
inline __m256 firCombine(__m256 acc, __m256 accIm) noexcept {
    return _mm256_addsub_ps(acc, _mm256_permute_ps(accIm, 0xB1));
}

void complexFir(const float* in,
                const float* tapsReal,
                const float* tapsImag,
                std::size_t  numTaps,
                float*       out,
                std::size_t  count) noexcept {
    std::size_t k = 0; // output sample index

    // 16 outputs at a time: 8 independent accumulators hide FMA latency.
    for (; k + 16 <= count; k += 16) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        __m256 b0 = a0, b1 = a0, b2 = a0, b3 = a0;

        const float* x = in + 2 * k;
        for (std::size_t j = 0; j < numTaps; ++j) {
            const __m256 hr = _mm256_broadcast_ss(tapsReal + j);
            const __m256 hi = _mm256_broadcast_ss(tapsImag + j);
            const __m256 x0 = _mm256_loadu_ps(x + 2 * j);
            const __m256 x1 = _mm256_loadu_ps(x + 2 * j + 8);
            const __m256 x2 = _mm256_loadu_ps(x + 2 * j + 16);
            const __m256 x3 = _mm256_loadu_ps(x + 2 * j + 24);

            a0 = _mm256_fmadd_ps(x0, hr, a0);
            b0 = _mm256_fmadd_ps(x0, hi, b0);
            a1 = _mm256_fmadd_ps(x1, hr, a1);
            b1 = _mm256_fmadd_ps(x1, hi, b1);
            a2 = _mm256_fmadd_ps(x2, hr, a2);
            b2 = _mm256_fmadd_ps(x2, hi, b2);
            a3 = _mm256_fmadd_ps(x3, hr, a3);
            b3 = _mm256_fmadd_ps(x3, hi, b3);
        }

        float* y = out + 2 * k;
        _mm256_storeu_ps(y, firCombine(a0, b0));
        _mm256_storeu_ps(y + 8, firCombine(a1, b1));
        _mm256_storeu_ps(y + 16, firCombine(a2, b2));
        _mm256_storeu_ps(y + 24, firCombine(a3, b3));
    }

    // Remaining outputs, 4 at a time; the last block is masked.
    for (; k < count; k += 4) {
        const std::size_t floats = 2 * blockSize(count, k, 4);
        const float*      x      = in + 2 * k;

        __m256 a = _mm256_setzero_ps();
        __m256 b = a;
        for (std::size_t j = 0; j < numTaps; ++j) {
            const __m256 xj = loadPartial(x + 2 * j, floats);

            a = _mm256_fmadd_ps(xj, _mm256_broadcast_ss(tapsReal + j), a);
            b = _mm256_fmadd_ps(xj, _mm256_broadcast_ss(tapsImag + j), b);
        }
        storePartial(out + 2 * k, floats, firCombine(a, b));
    }
}

/// Real-tap FIR: one FMA per tap per vector, half the work of complexFir.
void complexFirRealTaps(const float* in,
                        const float* taps,
                        std::size_t  numTaps,
                        float*       out,
                        std::size_t  count) noexcept {
    std::size_t k = 0; // output sample index

    // 32 outputs at a time: 8 independent accumulators hide FMA latency.
    for (; k + 32 <= count; k += 32) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        __m256 a4 = a0, a5 = a0, a6 = a0, a7 = a0;

        const float* x = in + 2 * k;
        for (std::size_t j = 0; j < numTaps; ++j) {
            const __m256 h  = _mm256_broadcast_ss(taps + j);
            const float* xj = x + 2 * j;

            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xj), h, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xj + 8), h, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(xj + 16), h, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(xj + 24), h, a3);
            a4 = _mm256_fmadd_ps(_mm256_loadu_ps(xj + 32), h, a4);
            a5 = _mm256_fmadd_ps(_mm256_loadu_ps(xj + 40), h, a5);
            a6 = _mm256_fmadd_ps(_mm256_loadu_ps(xj + 48), h, a6);
            a7 = _mm256_fmadd_ps(_mm256_loadu_ps(xj + 56), h, a7);
        }

        float* y = out + 2 * k;
        _mm256_storeu_ps(y, a0);
        _mm256_storeu_ps(y + 8, a1);
        _mm256_storeu_ps(y + 16, a2);
        _mm256_storeu_ps(y + 24, a3);
        _mm256_storeu_ps(y + 32, a4);
        _mm256_storeu_ps(y + 40, a5);
        _mm256_storeu_ps(y + 48, a6);
        _mm256_storeu_ps(y + 56, a7);
    }

    // Remaining outputs, 4 at a time; the last block is masked.
    for (; k < count; k += 4) {
        const std::size_t floats = 2 * blockSize(count, k, 4);
        const float*      x      = in + 2 * k;

        __m256 a = _mm256_setzero_ps();
        for (std::size_t j = 0; j < numTaps; ++j) {
            a = _mm256_fmadd_ps(loadPartial(x + 2 * j, floats), _mm256_broadcast_ss(taps + j), a);
        }
        storePartial(out + 2 * k, floats, a);
    }
}

} // namespace

extern const KernelTable kAvx2Kernels = {
    .isa                    = Isa::Avx2,
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
