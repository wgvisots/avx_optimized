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

// Times every function on every supported instruction set and prints throughput with the speedup
// over the scalar kernels.  Deliberately dependency-free; for publication-quality numbers, pin the
// process to one core (taskset -c N), fix the CPU frequency and run on an otherwise idle machine.
//
//   avx_optimized_benchmark [count]      (default 4096 elements: data stays in L1/L2 cache)

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "avx_optimized/avx_optimized.h"

namespace {

/// Best-of-15 time per call in nanoseconds; each sample runs the function enough times to take ~2 ms.
double nanosecondsPerCall(const std::function<void()>& function) {
    function(); // warm up caches and the dispatch table

    std::size_t repetitions = 1;
    for (;;) {
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        for (std::size_t r = 0; r < repetitions; ++r) {
            function();
        }
        if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(2)) {
            break;
        }
        repetitions *= 2;
    }

    double best = 1e300;
    for (int sample = 0; sample < 15; ++sample) {
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        for (std::size_t r = 0; r < repetitions; ++r) {
            function();
        }
        const std::chrono::duration<double, std::nano> elapsed = std::chrono::steady_clock::now() - start;
        best = std::min(best, elapsed.count() / static_cast<double>(repetitions));
    }
    return best;
}

struct BenchmarkCase {
    std::string           name;
    std::function<void()> run;
};

} // namespace

int main(int argc, char** argv) {
    const std::size_t count = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 4096;

    std::mt19937                          engine(1);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);

    avx_optimized::AlignedVector<float>               floats1(2 * count);
    avx_optimized::AlignedVector<float>               floats2(2 * count);
    avx_optimized::AlignedVector<float>               floats3(2 * count);
    avx_optimized::AlignedVector<std::complex<float>> samples1(count);
    avx_optimized::AlignedVector<std::complex<float>> phasors(count);
    avx_optimized::AlignedVector<std::complex<float>> work(count);
    avx_optimized::AlignedVector<std::int16_t>        shorts1(2 * count);
    avx_optimized::AlignedVector<std::int16_t>        shorts2(2 * count);
    for (std::size_t i = 0; i < 2 * count; ++i) {
        floats1[i] = uniform(engine);
        floats2[i] = uniform(engine);
        shorts1[i] = static_cast<std::int16_t>(uniform(engine) * 30000);
        shorts2[i] = static_cast<std::int16_t>(uniform(engine) * 30000);
    }
    for (std::size_t i = 0; i < count; ++i) {
        samples1[i] = {uniform(engine), uniform(engine)};
        phasors[i]  = std::polar(1.0F, 3.14159265F * uniform(engine));
    }

    // In-place functions run repeatedly on the same buffer, so they use gains of magnitude 1:
    // values stay bounded and never decay into (very slow) subnormals.
    const std::vector<std::complex<float>> complexTaps16(16, std::complex<float>(0.0625F, 0.01F));
    const std::vector<std::complex<float>> complexTaps64(64, std::complex<float>(0.015F, 0.002F));
    const std::vector<std::complex<float>> complexTaps256(256, std::complex<float>(0.004F, 0.0005F));
    const std::vector<float>               realTaps16(16, 0.0625F);
    const std::vector<float>               realTaps64(64, 0.015F);
    const std::vector<float>               realTaps256(256, 0.004F);

    avx_optimized::ComplexFirFilter complexFir16(complexTaps16);
    avx_optimized::ComplexFirFilter complexFir64(complexTaps64);
    avx_optimized::ComplexFirFilter complexFir256(complexTaps256);
    avx_optimized::ComplexFirFilter realFir16(realTaps16);
    avx_optimized::ComplexFirFilter realFir64(realTaps64);
    avx_optimized::ComplexFirFilter realFir256(realTaps256);

    // clang-format off
    const std::vector<BenchmarkCase> cases = {
        {"int16ToFloat",               [&] { avx_optimized::int16ToFloat(shorts1.data(), floats3.data(), count); }},
        {"floatToInt16",               [&] { avx_optimized::floatToInt16(floats1.data(), shorts2.data(), count); }},
        {"int16ToComplex (scaled)",    [&] { avx_optimized::int16ToComplex(shorts1.data(), work.data(), count, 1.0F / 32768.0F); }},
        {"complexToInt16 (scaled)",    [&] { avx_optimized::complexToInt16(samples1.data(), shorts2.data(), count, 32767.0F); }},
        {"add",                        [&] { avx_optimized::add(floats3.data(), floats2.data(), count); }},
        {"addSaturated",               [&] { avx_optimized::addSaturated(shorts2.data(), shorts1.data(), count); }},
        {"scale",                      [&] { avx_optimized::scale(floats3.data(), 1.0F, count); }},
        {"reverse",                    [&] { avx_optimized::reverse(floats1.data(), floats3.data(), count); }},
        {"scaleIQ",                    [&] { avx_optimized::scaleIQ(work.data(), 1.0F, -1.0F, count); }},
        {"complexMultiply",            [&] { avx_optimized::complexMultiply(work.data(), phasors.data(), count); }},
        {"applyLinearGainRamp",        [&] { avx_optimized::applyLinearGainRamp(work.data(), 1.0F, 1.0F, count); }},
        {"magnitudeSquared",           [&] { avx_optimized::magnitudeSquared(samples1.data(), floats3.data(), count); }},
        {"magnitude",                  [&] { avx_optimized::magnitude(samples1.data(), floats3.data(), count); }},
        {"toPolar",                    [&] { avx_optimized::toPolar(samples1.data(), floats2.data(), floats3.data(), count); }},
        {"powerToDecibels (complex)",  [&] { avx_optimized::powerToDecibels(samples1.data(), floats3.data(), count); }},
        {"ComplexFirFilter 16 complex",  [&] { complexFir16.filter(samples1.data(), work.data(), count); }},
        {"ComplexFirFilter 16 real",     [&] { realFir16.filter(samples1.data(), work.data(), count); }},
        {"ComplexFirFilter 64 complex",  [&] { complexFir64.filter(samples1.data(), work.data(), count); }},
        {"ComplexFirFilter 64 real",     [&] { realFir64.filter(samples1.data(), work.data(), count); }},
        {"ComplexFirFilter 256 complex", [&] { complexFir256.filter(samples1.data(), work.data(), count); }},
        {"ComplexFirFilter 256 real",    [&] { realFir256.filter(samples1.data(), work.data(), count); }},
    };
    // clang-format on

    std::vector<avx_optimized::Isa> isas;
    for (const avx_optimized::Isa isa :
         {avx_optimized::Isa::Scalar, avx_optimized::Isa::Avx2, avx_optimized::Isa::Avx512}) {
        if (avx_optimized::isSupported(isa)) {
            isas.push_back(isa);
        }
    }

    std::printf("avx_optimized %s benchmark, %zu elements per call\n", avx_optimized::versionString(), count);
    std::printf("Times are nanoseconds per element (lower is better); x = speedup over scalar.\n\n");
    std::printf("%-30s", "function");
    for (const avx_optimized::Isa isa : isas) {
        std::printf("%18s", avx_optimized::isaName(isa));
    }
    std::printf("\n");

    for (const BenchmarkCase& benchmarkCase : cases) {
        std::printf("%-30s", benchmarkCase.name.c_str());
        double scalarNanoseconds = 0.0;
        for (const avx_optimized::Isa isa : isas) {
            avx_optimized::setActiveIsa(isa);
            std::fill(work.begin(), work.end(), std::complex<float>(0.5F, -0.25F));
            const double perElement = nanosecondsPerCall(benchmarkCase.run) / static_cast<double>(count);
            if (isa == avx_optimized::Isa::Scalar) {
                scalarNanoseconds = perElement;
                std::printf("%18.3f", perElement);
            } else {
                std::printf("%10.3f (%4.1fx)", perElement, scalarNanoseconds / perElement);
            }
        }
        std::printf("\n");
    }
    avx_optimized::setActiveIsa(avx_optimized::bestSupportedIsa());
    return 0;
}
