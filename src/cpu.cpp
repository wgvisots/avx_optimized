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

#include "avx_optimized/cpu.h"

#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kernels.h"

namespace avx_optimized {
namespace {

bool cpuSupports(Isa isa) noexcept {
    switch (isa) {
    case Isa::Scalar: return true;
#if defined(AVX_OPTIMIZED_HAVE_X86_KERNELS)
    // __builtin_cpu_supports also checks that the OS saves the wider register state (XCR0), so a
    // CPU with AVX-512 under an OS or hypervisor that disables it is correctly rejected.
    case Isa::Avx2: __builtin_cpu_init(); return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    case Isa::Avx512:
        __builtin_cpu_init();
        return cpuSupports(Isa::Avx2) && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
            && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
#endif
    default: return false;
    }
}

const avx_optimized::detail::KernelTable& tableFor(Isa isa) noexcept {
    switch (isa) {
#if defined(AVX_OPTIMIZED_HAVE_X86_KERNELS)
    case Isa::Avx512: return avx_optimized::detail::kAvx512Kernels;
    case Isa::Avx2:   return avx_optimized::detail::kAvx2Kernels;
#endif
    default: return avx_optimized::detail::kScalarKernels;
    }
}

/// Case-insensitive comparison; avoids std::string so that dispatch never allocates.
bool equalsIgnoringCase(const char* text, const char* lowercase) noexcept {
    for (; *text != '\0' && *lowercase != '\0'; ++text, ++lowercase) {
        if (std::tolower(static_cast<unsigned char>(*text)) != *lowercase) {
            return false;
        }
    }
    return *text == '\0' && *lowercase == '\0';
}

bool parseIsa(const char* text, Isa& isa) noexcept {
    for (Isa candidate : {Isa::Scalar, Isa::Avx2, Isa::Avx512}) {
        if (equalsIgnoringCase(text, isaName(candidate))) {
            isa = candidate;
            return true;
        }
    }
    return false;
}

/// Longest part of an unrecognised AVX_OPTIMIZED_ISA value that the warning shows.
constexpr std::size_t kMaxShownLength = 32;

/// Text that is safe to print to a terminal, in a fixed-size buffer (dispatch never allocates).
struct PrintableText {
    char text[kMaxShownLength + 4]; ///< Up to kMaxShownLength characters, "..." and the terminator.
};

/// `text` made safe to print: every character outside printable ASCII becomes '?', so a value
/// can't send escape sequences to the terminal, and a value longer than kMaxShownLength is cut
/// short and ends in "...".
PrintableText printable(const char* text) noexcept {
    PrintableText result{};
    std::size_t   length = 0;
    for (; text[length] != '\0' && length < kMaxShownLength; ++length) {
        const unsigned char c = static_cast<unsigned char>(text[length]);
        result.text[length]   = (c >= 0x20 && c < 0x7F) ? static_cast<char>(c) : '?';
    }
    if (text[length] != '\0') {
        result.text[length]     = '.';
        result.text[length + 1] = '.';
        result.text[length + 2] = '.';
    }
    return result;
}

/// The best supported ISA, capped by the AVX_OPTIMIZED_ISA environment variable if it is set.
Isa computeInitialIsa() noexcept {
    Isa         isa       = bestSupportedIsa();
    const char* requested = std::getenv("AVX_OPTIMIZED_ISA");
    if (requested == nullptr) {
        return isa;
    }

    Isa cap{};
    if (!parseIsa(requested, cap)) {
        std::fprintf(stderr,
                     "avx_optimized: ignoring unrecognised AVX_OPTIMIZED_ISA value '%s' "
                     "(expected scalar, avx2 or avx512); using %s\n",
                     printable(requested).text,
                     isaName(isa));
        return isa;
    }
    return static_cast<int>(cap) < static_cast<int>(isa) ? cap : isa;
}

/// computeInitialIsa(), evaluated once (so any warning is printed once, even with several threads).
Isa initialIsa() noexcept {
    static const Isa initial = computeInitialIsa();
    return initial;
}

// Null until the first call.  Tables are immutable, so publishing a pointer is all the
// synchronization needed.
std::atomic<const avx_optimized::detail::KernelTable*> activeTable{nullptr};

} // namespace

namespace detail {

const avx_optimized::detail::KernelTable& activeKernels() noexcept {
    const avx_optimized::detail::KernelTable* table = activeTable.load(std::memory_order_acquire);
    if (table == nullptr) {
        const avx_optimized::detail::KernelTable* initial = &tableFor(initialIsa());
        // If another thread got here first (or setActiveIsa() ran), keep its choice.
        if (activeTable.compare_exchange_strong(table, initial, std::memory_order_acq_rel)) {
            table = initial;
        }
    }
    return *table;
}

} // namespace detail

bool isSupported(Isa isa) noexcept {
    return cpuSupports(isa);
}

Isa bestSupportedIsa() noexcept {
    if (cpuSupports(Isa::Avx512)) {
        return Isa::Avx512;
    }
    if (cpuSupports(Isa::Avx2)) {
        return Isa::Avx2;
    }
    return Isa::Scalar;
}

Isa activeIsa() noexcept {
    return avx_optimized::detail::activeKernels().isa;
}

void setActiveIsa(Isa isa) {
    if (!isSupported(isa)) {
        throw std::invalid_argument(std::string("avx_optimized: instruction set '") + isaName(isa)
                                    + "' is not supported by this CPU or build");
    }
    activeTable.store(&tableFor(isa), std::memory_order_release);
}

const char* isaName(Isa isa) noexcept {
    switch (isa) {
    case Isa::Scalar: return "scalar";
    case Isa::Avx2:   return "avx2";
    case Isa::Avx512: return "avx512";
    }
    return "unknown";
}

} // namespace avx_optimized
