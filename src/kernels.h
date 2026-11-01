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

/*
 * Internal: the per-instruction-set kernel tables behind the public API.
 *
 * Each kernels_<isa>.cpp is compiled with that instruction set enabled (-mavx2, -mavx512f, ...)
 * and fills in one KernelTable.  cpu.cpp picks a table at runtime, and the public functions
 * (vectorized.cpp, complex_fir_filter.cpp) call through it.  Table entries are named after the
 * public function they implement.
 *
 * RULES FOR ISA TRANSLATION UNITS (kernels_avx2.cpp, kernels_avx512.cpp)
 * ---------------------------------------------------------------------
 * Any inline or template function that such a TU fails to inline gets emitted as a weak symbol
 * compiled for that ISA.  If another TU emits the same symbol for a lower ISA, the linker keeps
 * one copy at random, and the program can crash with SIGILL on CPUs that lack the instruction
 * set (the classic multi-versioning ODR trap).  So in ISA TUs:
 *   - only use intrinsics, plain arithmetic and loops, and helpers in an anonymous namespace
 *     (internal linkage, so never shared);
 *   - do not call std:: functions or templates (std::min, std::sqrt, std::complex, containers,
 *     ...).  Use intrinsics or write the expression out;
 *   - only include <immintrin.h>, <cstddef>, <cstdint>, this header, and the mathfun header for
 *     that ISA (its functions live in a per-ISA namespace, so they cannot collide).
 * This header must therefore stay free of inline functions and templates itself.  The
 * avx_optimized_isa_object_symbols test checks the compiled objects for violations.
 *
 * Kernel conventions: complex data is interleaved float {re, im}, and `count` is in complex
 * samples for complex kernels (as in the public API).  Kernels must handle any count, including 0,
 * and must never form a pointer outside the arrays they were given.
 */

#include <cstddef>
#include <cstdint>

#include "avx_optimized/cpu.h"

namespace avx_optimized::detail {

struct KernelTable {
    Isa isa;

    // Conversions.  int16ToFloat: out = in * scale.  floatToInt16: out = round(saturate(in * scale)).
    void (*int16ToFloat)(const std::int16_t* in, float* out, std::size_t count, float scale) noexcept;
    void (*floatToInt16)(const float* in, std::int16_t* out, std::size_t count, float scale) noexcept;

    // Element-wise operations on floats.  The complex overloads of add() and scale() call these
    // with 2 * count.
    void (*add)(float* io, const float* other, std::size_t count) noexcept;
    void (*addSaturated)(std::int16_t* io, const std::int16_t* other, std::size_t count) noexcept;
    void (*scale)(float* io, float factor, std::size_t count) noexcept;
    void (*reverse)(const float* in, float* out, std::size_t count) noexcept;

    // Complex operations; count is in complex samples.
    void (*scaleIQ)(float* io, float iFactor, float qFactor, std::size_t count) noexcept;
    void (*complexMultiply)(float* io, const float* other, std::size_t count) noexcept;
    void (*magnitudeSquared)(const float* in, float* out, std::size_t count) noexcept;
    void (*magnitude)(const float* in, float* out, std::size_t count) noexcept;
    void (*toPolar)(const float* in, float* magnitude, float* phase, std::size_t count) noexcept;

    /// io[k] *= start + float(k) * step.  Callers guarantee count < 2^31.
    void (*applyLinearGainRamp)(float* io, float start, float step, std::size_t count) noexcept;

    /// decibels = 10 * log10(power), with the special cases documented in vectorized.h.
    void (*powerToDecibels)(const float* power, float* decibels, std::size_t count) noexcept;
    /// decibels = 10 * log10(|in|^2) for complex `in`.
    void (*complexPowerToDecibels)(const float* in, float* decibels, std::size_t count) noexcept;

    /**
     * Direct-form complex FIR over a prepared buffer:
     *     out[k] = sum_{j=0}^{numTaps-1} (tapsReal[j] + i*tapsImag[j]) * in[k + j]
     * where `in` holds count + numTaps - 1 complex samples and the taps are already reversed.
     */
    void (*complexFir)(const float* in,
                       const float* tapsReal,
                       const float* tapsImag,
                       std::size_t  numTaps,
                       float*       out,
                       std::size_t  count) noexcept;

    /// As complexFir, with real taps: out[k] = sum_j taps[j] * in[k + j].
    void (*complexFirRealTaps)(const float* in,
                               const float* taps,
                               std::size_t  numTaps,
                               float*       out,
                               std::size_t  count) noexcept;
};

extern const KernelTable kScalarKernels;
#if defined(AVX_OPTIMIZED_HAVE_X86_KERNELS)
extern const KernelTable kAvx2Kernels;
extern const KernelTable kAvx512Kernels;
#endif

/// The table for the currently active ISA (chosen on first use; see cpu.cpp).
const KernelTable& activeKernels() noexcept;

} // namespace avx_optimized::detail
