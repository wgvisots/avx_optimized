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

// A small I/Q processing chain: int16 samples from a radio -> complex float -> low-pass filter ->
// power in dB.  Exits non-zero if the result looks wrong, so it doubles as a smoke test.

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include <avx_optimized/avx_optimized.h>

int main() {
    std::printf("avx_optimized %s, using the %s kernels\n",
                avx_optimized::versionString(),
                avx_optimized::isaName(avx_optimized::activeIsa()));

    // 1. Pretend these interleaved I/Q int16 samples came from an ADC: a weak tone near DC plus a
    //    strong one far away that we want to filter out.
    constexpr std::size_t kSamples = 8192;
    constexpr double      kPi      = 3.14159265358979323846;

    std::vector<std::int16_t> raw(2 * kSamples);
    for (std::size_t k = 0; k < kSamples; ++k) {
        const double wanted   = 0.01 * 2 * kPi * static_cast<double>(k);
        const double unwanted = 0.4 * 2 * kPi * static_cast<double>(k);

        raw[2 * k]     = static_cast<std::int16_t>(4000 * std::cos(wanted) + 20000 * std::cos(unwanted));
        raw[2 * k + 1] = static_cast<std::int16_t>(4000 * std::sin(wanted) + 20000 * std::sin(unwanted));
    }

    // 2. Convert to complex samples in [-1, 1) in one pass.  Containers are passed directly (as
    //    spans, with sizes checked).  AlignedVector is optional but faster.
    avx_optimized::AlignedVector<std::complex<float>> iq(kSamples);
    avx_optimized::int16ToComplex(raw, iq, 1.0F / 32768.0F);

    // 3. Low-pass filter (31-tap moving average, real taps), in two chunks as if streaming.
    const std::vector<float>             taps(31, 1.0F / 31.0F);
    avx_optimized::ComplexFirFilter      lowPass(taps);
    const std::span<std::complex<float>> all = iq;
    lowPass.filter(all.first(kSamples / 2)); // in place
    lowPass.filter(all.last(kSamples / 2));

    // 4. Power of each sample, in dB.
    avx_optimized::AlignedVector<float> powerDb(kSamples);
    avx_optimized::powerToDecibels(iq, powerDb);

    double meanDb = 0.0;
    for (std::size_t k = taps.size(); k < kSamples; ++k) { // skip the filter's start-up transient
        meanDb += powerDb[k];
    }
    meanDb /= static_cast<double>(kSamples - taps.size());

    // The wanted tone has power (4000/32768)^2 ~ 0.0149, times the moving average's power gain of
    // ~0.72 at that frequency = ~0.0108, i.e. about -19.7 dB.  The unwanted tone (power ~0.37) is
    // attenuated ~30 dB, so what leaks through moves the per-sample power only slightly.
    std::printf("mean power after filtering: %.2f dB (expected about -19.7 dB)\n", meanDb);
    return (meanDb > -21.0 && meanDb < -18.5) ? 0 : 1;
}
