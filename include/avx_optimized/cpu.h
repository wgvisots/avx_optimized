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

#include "avx_optimized/export.h"

/**
 * Runtime instruction-set dispatch.
 *
 * Every function in this library is compiled several times, once per instruction set, and the
 * best version the running CPU supports is picked automatically the first time any of them is
 * called.  You normally never need anything in this header.  It exists for:
 *  - finding out which implementation is in use (logging, bug reports),
 *  - pinning a lower instruction set, e.g. to avoid AVX-512 frequency throttling on older Xeons,
 *    or to compare implementations in tests and benchmarks.
 *
 * The initial choice can also be capped without code changes by setting the environment variable
 * AVX_OPTIMIZED_ISA to "scalar", "avx2" or "avx512" (case-insensitive) before the first call.
 * Asking for more than the CPU supports falls back to the best supported level.  An unrecognised
 * value is ignored, with a one-line warning on stderr so a typo doesn't go unnoticed.  The variable
 * only sets the starting point: setActiveIsa() can still select any supported level.
 */
namespace avx_optimized {

/// Instruction-set levels, ordered from least to most capable.
enum class Isa : int {
    Scalar = 0, ///< Portable C++; runs anywhere (and is the only level on non-x86 builds).
    Avx2   = 1, ///< AVX2 + FMA (Intel Haswell / AMD Zen and newer).
    Avx512 = 2, ///< AVX-512 F, BW, DQ and VL (Intel Skylake-SP / AMD Zen 4 and newer).
};

/// True if this build contains `isa` and the running CPU (and OS) can execute it.
AVX_OPTIMIZED_EXPORT bool isSupported(Isa isa) noexcept;

/// The most capable instruction set that isSupported().
AVX_OPTIMIZED_EXPORT Isa bestSupportedIsa() noexcept;

/// The instruction set currently used by every function in this library.
AVX_OPTIMIZED_EXPORT Isa activeIsa() noexcept;

/**
 * Switch every function in this library to `isa`.  Overrides the AVX_OPTIMIZED_ISA environment
 * variable.
 *
 * Takes effect for calls that start after it returns.  It is safe to call concurrently with other
 * library calls, but calls already in progress finish with the previous implementation.
 *
 * @throws std::invalid_argument if `isa` is not supported (see isSupported()).
 */
AVX_OPTIMIZED_EXPORT void setActiveIsa(Isa isa);

/// "scalar", "avx2" or "avx512" (the names AVX_OPTIMIZED_ISA accepts); "unknown" for any other value.
AVX_OPTIMIZED_EXPORT const char* isaName(Isa isa) noexcept;

} // namespace avx_optimized
