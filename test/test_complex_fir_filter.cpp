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

#include <algorithm>
#include <cmath>
#include <complex>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "avx_optimized/aligned.h"
#include "avx_optimized/complex_fir_filter.h"
#include "test_helpers.h"

namespace {

/// y[k] = sum_m h[m] x[k-m] with zero history, in double precision.  Also returns, per output, the
/// sum of |h[m]| |x[k-m]|, which bounds the rounding error of a float evaluation.
void referenceFir(const std::vector<std::complex<float>>& taps,
                  const std::vector<std::complex<float>>& x,
                  std::vector<std::complex<double>>&      y,
                  std::vector<double>&                    magnitudeSum) {
    y.assign(x.size(), std::complex<double>{});
    magnitudeSum.assign(x.size(), 0.0);
    for (std::size_t k = 0; k < x.size(); ++k) {
        for (std::size_t m = 0; m < taps.size() && m <= k; ++m) {
            y[k]            += std::complex<double>(taps[m]) * std::complex<double>(x[k - m]);
            magnitudeSum[k] += std::abs(std::complex<double>(taps[m])) * std::abs(std::complex<double>(x[k - m]));
        }
    }
}

::testing::AssertionResult matchesReference(const std::vector<std::complex<float>>& taps,
                                            const std::vector<std::complex<float>>& x,
                                            const std::complex<float>*              actual) {
    std::vector<std::complex<double>> expected;
    std::vector<double>               magnitudeSum;
    referenceFir(taps, x, expected, magnitudeSum);

    // Worst-case float accumulation error grows with the number of taps.
    const double relativeTolerance = 1.2e-7 * static_cast<double>(taps.size() + 4);
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double tolerance = relativeTolerance * magnitudeSum[k] + 1e-30;
        if (std::abs(std::complex<double>(actual[k]) - expected[k]) > tolerance) {
            return ::testing::AssertionFailure() << "output " << k << " of " << x.size() << ": got " << actual[k]
                                                 << ", expected " << expected[k] << " (tolerance " << tolerance << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

std::vector<std::complex<float>> asComplexTaps(const std::vector<float>& realTaps) {
    return std::vector<std::complex<float>>(realTaps.begin(), realTaps.end());
}

std::size_t indexOfLargest(const std::vector<std::complex<float>>& y) {
    std::size_t peak = 0;
    for (std::size_t k = 1; k < y.size(); ++k) {
        if (std::abs(y[k]) > std::abs(y[peak])) {
            peak = k;
        }
    }
    return peak;
}

class ComplexFir : public test_helpers::IsaTest {};

} // namespace

// ---------------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------------

TEST(ComplexFirFilterConstruction, RejectsEmptyTaps) {
    const std::complex<float> complexTap{1.0F, 0.0F};
    const float               realTap = 1.0F;
    EXPECT_THROW(avx_optimized::ComplexFirFilter(static_cast<const std::complex<float>*>(nullptr), 3),
                 std::invalid_argument);
    EXPECT_THROW(avx_optimized::ComplexFirFilter(static_cast<const float*>(nullptr), 3), std::invalid_argument);
    EXPECT_THROW(avx_optimized::ComplexFirFilter(&complexTap, 0), std::invalid_argument);
    EXPECT_THROW(avx_optimized::ComplexFirFilter(&realTap, 0), std::invalid_argument);
    EXPECT_THROW(avx_optimized::ComplexFirFilter(std::vector<std::complex<float>>{}), std::invalid_argument);
    EXPECT_THROW(avx_optimized::ComplexFirFilter(std::vector<float>{}), std::invalid_argument);
}

TEST(ComplexFirFilterConstruction, KeepsTapsUnnormalized) {
    const std::vector<std::complex<float>> taps = {
        {2.0F, 1.0F },
        {0.0F, -3.0F},
        {4.0F, 0.5F }
    };
    const avx_optimized::ComplexFirFilter filter(taps);
    EXPECT_EQ(filter.numTaps(), 3U);
    EXPECT_EQ(filter.delay(), 2U);
    EXPECT_EQ(filter.taps(), taps);
    EXPECT_FALSE(filter.hasRealTaps());
}

TEST(ComplexFirFilterConstruction, DetectsRealTaps) {
    const std::vector<float>               realTaps = {0.25F, 0.5F, 0.25F};
    const std::vector<std::complex<float>> zeroImag = asComplexTaps(realTaps);

    const avx_optimized::ComplexFirFilter fromReal(realTaps);
    const avx_optimized::ComplexFirFilter fromComplex(zeroImag);
    EXPECT_TRUE(fromReal.hasRealTaps());
    EXPECT_TRUE(fromComplex.hasRealTaps());
    EXPECT_EQ(fromReal.taps(), zeroImag);
}

TEST(ComplexFirFilterConstruction, AcceptsAlignedVectors) {
    const avx_optimized::AlignedVector<std::complex<float>> complexTaps = {
        {1.0F, 1.0F},
        {0.5F, 0.0F}
    };
    const avx_optimized::AlignedVector<float> realTaps = {1.0F, 0.5F};

    const avx_optimized::ComplexFirFilter fromComplex(complexTaps);
    const avx_optimized::ComplexFirFilter fromReal(realTaps);
    const avx_optimized::ComplexFirFilter matched =
        avx_optimized::ComplexFirFilter::matchedFilter(complexTaps, avx_optimized::TemplateScaling::AsGiven);
    EXPECT_EQ(fromComplex.numTaps(), 2U);
    EXPECT_EQ(fromReal.numTaps(), 2U);
    EXPECT_EQ(matched.numTaps(), 2U);
}

// ---------------------------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------------------------

TEST_P(ComplexFir, ImpulseResponseIsTheTaps) {
    for (std::size_t numTaps : {1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 33}) {
        SCOPED_TRACE("numTaps=" + std::to_string(numTaps));
        const std::vector<std::complex<float>> taps = test_helpers::randomComplex(numTaps);
        avx_optimized::ComplexFirFilter        filter(taps);

        std::vector<std::complex<float>> x(numTaps + 40, std::complex<float>{});
        std::vector<std::complex<float>> y(x.size());
        x[0] = {1.0F, 0.0F};

        filter.filter(x.data(), y.data(), x.size());

        for (std::size_t k = 0; k < y.size(); ++k) {
            const std::complex<float> expected = k < numTaps ? taps[k] : std::complex<float>{};
            ASSERT_EQ(y[k], expected) << "k=" << k;
        }
    }
}

TEST_P(ComplexFir, MatchesReferenceForManySizes) {
    for (std::size_t numTaps : {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 32, 63, 129}) {
        const std::vector<std::complex<float>> complexTaps = test_helpers::randomComplex(numTaps);
        const std::vector<float>               realTaps    = test_helpers::randomFloats(numTaps);
        for (std::size_t count : test_helpers::testSizes()) {
            for (std::size_t offset : {0, 1}) {
                SCOPED_TRACE("numTaps=" + std::to_string(numTaps) + " count=" + std::to_string(count)
                             + " offset=" + std::to_string(offset));
                const std::vector<std::complex<float>>           x = test_helpers::randomComplex(count);
                test_helpers::GuardedBuffer<std::complex<float>> in(count, offset);
                test_helpers::GuardedBuffer<std::complex<float>> out(count, offset);
                in.fill(x);

                avx_optimized::ComplexFirFilter complexFilter(complexTaps);
                complexFilter.filter(in.data(), out.data(), count);
                ASSERT_TRUE(out.intact());
                ASSERT_TRUE(matchesReference(complexTaps, x, out.data()));

                avx_optimized::ComplexFirFilter realFilter(realTaps);
                realFilter.filter(in.data(), out.data(), count);
                ASSERT_TRUE(out.intact());
                ASSERT_TRUE(matchesReference(asComplexTaps(realTaps), x, out.data()));
            }
        }
    }
}

TEST_P(ComplexFir, RealTapKernelAgreesWithComplexTapKernel) {
    const std::vector<float>               realTaps = test_helpers::randomFloats(23);
    const std::vector<std::complex<float>> x        = test_helpers::randomComplex(777);

    avx_optimized::ComplexFirFilter  realFilter(realTaps);
    std::vector<std::complex<float>> realOut(x.size());
    realFilter.filter(x.data(), realOut.data(), x.size());

    // A negligible imaginary part forces the general complex-tap kernel.
    std::vector<std::complex<float>> complexTaps = asComplexTaps(realTaps);
    complexTaps[5].imag(1e-30F);
    avx_optimized::ComplexFirFilter complexFilter(complexTaps);
    ASSERT_FALSE(complexFilter.hasRealTaps());
    std::vector<std::complex<float>> complexOut(x.size());
    complexFilter.filter(x.data(), complexOut.data(), x.size());

    ASSERT_TRUE(matchesReference(asComplexTaps(realTaps), x, realOut.data()));
    for (std::size_t k = 0; k < x.size(); ++k) {
        ASSERT_NEAR(std::abs(realOut[k] - complexOut[k]), 0.0F, 1e-6F) << "k=" << k;
    }
}

TEST_P(ComplexFir, StreamingInChunksEqualsOneShot) {
    for (const bool realTaps : {false, true}) {
        SCOPED_TRACE(realTaps ? "real taps" : "complex taps");
        const std::vector<std::complex<float>> taps =
            realTaps ? asComplexTaps(test_helpers::randomFloats(37)) : test_helpers::randomComplex(37);
        const std::vector<std::complex<float>> x = test_helpers::randomComplex(10000); // > internal block size

        avx_optimized::ComplexFirFilter  oneShot(taps);
        std::vector<std::complex<float>> expected(x.size());
        oneShot.filter(x.data(), expected.data(), x.size());
        ASSERT_TRUE(matchesReference(taps, x, expected.data()));

        // Chunk sizes chosen to land on and around every vector width and the internal block size.
        const std::size_t chunks[] = {0, 1, 2, 3, 5, 7, 8, 13, 16, 31, 32, 33, 64, 100, 4095, 4096, 4097};

        avx_optimized::ComplexFirFilter  streaming(taps);
        std::vector<std::complex<float>> actual(x.size());
        std::size_t                      position = 0;
        for (std::size_t i = 0; position < x.size(); ++i) {
            const std::size_t count = std::min(chunks[i % std::size(chunks)], x.size() - position);
            streaming.filter(x.data() + position, actual.data() + position, count);
            position += count;
        }

        // Each output goes through identical arithmetic whatever the chunking, so this is exact.
        for (std::size_t k = 0; k < x.size(); ++k) {
            ASSERT_EQ(actual[k], expected[k]) << "k=" << k;
        }
    }
}

TEST_P(ComplexFir, InPlaceFilteringWorks) {
    const std::vector<std::complex<float>> taps = test_helpers::randomComplex(20);
    const std::vector<std::complex<float>> x    = test_helpers::randomComplex(9000);

    avx_optimized::ComplexFirFilter  reference(taps);
    std::vector<std::complex<float>> expected(x.size());
    reference.filter(x.data(), expected.data(), x.size());

    avx_optimized::ComplexFirFilter  inPlace(taps);
    std::vector<std::complex<float>> data = x;
    inPlace.filter(data.data(), data.data(), data.size());

    EXPECT_EQ(data, expected);
}

TEST_P(ComplexFir, FlushWritesTheRemainingOutputsAndResets) {
    const std::vector<std::complex<float>> taps = test_helpers::randomComplex(19);
    const std::vector<std::complex<float>> x    = test_helpers::randomComplex(500);

    // Reference: the stream followed by delay() zeros, in one call.
    std::vector<std::complex<float>> padded = x;
    padded.resize(x.size() + taps.size() - 1, std::complex<float>{});
    avx_optimized::ComplexFirFilter  reference(taps);
    std::vector<std::complex<float>> expected(padded.size());
    reference.filter(padded.data(), expected.data(), padded.size());

    avx_optimized::ComplexFirFilter  filter(taps);
    std::vector<std::complex<float>> actual(padded.size());
    filter.filter(x.data(), actual.data(), x.size());
    filter.flush(actual.data() + x.size());
    EXPECT_EQ(actual, expected);

    // After flush() the filter starts a fresh stream.
    std::vector<std::complex<float>> again(x.size());
    filter.filter(x.data(), again.data(), x.size());
    EXPECT_TRUE(std::equal(again.begin(), again.end(), expected.begin()));
}

TEST_P(ComplexFir, ResetForgetsHistory) {
    const std::vector<std::complex<float>> taps = test_helpers::randomComplex(12);
    const std::vector<std::complex<float>> x    = test_helpers::randomComplex(300);
    avx_optimized::ComplexFirFilter        filter(taps);

    std::vector<std::complex<float>> first(x.size());
    filter.filter(x.data(), first.data(), x.size());

    std::vector<std::complex<float>> withHistory(x.size());
    filter.filter(x.data(), withHistory.data(), x.size());
    EXPECT_NE(withHistory, first); // the second pass sees the first pass's tail

    filter.reset();
    std::vector<std::complex<float>> afterReset(x.size());
    filter.filter(x.data(), afterReset.data(), x.size());
    EXPECT_EQ(afterReset, first);
}

TEST_P(ComplexFir, LowPassFilterPassesDcAndRejectsHighFrequency) {
    // Windowed-sinc low-pass with real taps, cutoff 0.1 cycles/sample, unity gain at DC.
    constexpr std::size_t kTaps = 63;
    constexpr double      kPi   = 3.14159265358979323846;

    std::vector<float> taps(kTaps);
    double             sum = 0.0;
    for (std::size_t m = 0; m < kTaps; ++m) {
        const double t       = static_cast<double>(m) - (kTaps - 1) / 2.0;
        const double sinc    = t == 0.0 ? 2 * 0.1 : std::sin(2 * kPi * 0.1 * t) / (kPi * t);
        const double hamming = 0.54 - 0.46 * std::cos(2 * kPi * static_cast<double>(m) / (kTaps - 1));

        taps[m]  = static_cast<float>(sinc * hamming);
        sum     += sinc * hamming;
    }
    for (float& tap : taps) {
        tap /= static_cast<float>(sum);
    }

    const auto toneRms = [&](double cyclesPerSample) {
        avx_optimized::ComplexFirFilter  filter(taps);
        std::vector<std::complex<float>> x(2048);
        for (std::size_t k = 0; k < x.size(); ++k) {
            const double phase = 2 * kPi * cyclesPerSample * static_cast<double>(k);
            x[k]               = {static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};
        }
        std::vector<std::complex<float>> y(x.size());
        filter.filter(x.data(), y.data(), x.size());

        double power = 0.0;
        for (std::size_t k = kTaps; k < y.size(); ++k) { // skip the start-up transient
            power += std::norm(std::complex<double>(y[k]));
        }
        return std::sqrt(power / static_cast<double>(y.size() - kTaps));
    };

    EXPECT_NEAR(toneRms(0.0), 1.0, 1e-4); // DC passes at unity gain
    EXPECT_LT(toneRms(0.35), 0.01);       // well into the stopband: >40 dB down
    EXPECT_LT(toneRms(-0.35), 0.01);      // negative frequencies too (complex data)
}

// ---------------------------------------------------------------------------------------------
// Matched filter
// ---------------------------------------------------------------------------------------------

TEST(MatchedFilter, TapsAreTheConjugatedReversedTemplate) {
    const std::vector<std::complex<float>> templ = test_helpers::randomComplex(23);
    const avx_optimized::ComplexFirFilter  filter =
        avx_optimized::ComplexFirFilter::matchedFilter(templ, avx_optimized::TemplateScaling::AsGiven);
    ASSERT_EQ(filter.numTaps(), templ.size());
    EXPECT_EQ(filter.delay(), templ.size() - 1);
    for (std::size_t m = 0; m < templ.size(); ++m) {
        EXPECT_EQ(filter.taps()[m], std::conj(templ[templ.size() - 1 - m])) << "m=" << m;
    }
}

TEST(MatchedFilter, UnitEnergyScalesTapsToEnergyOne) {
    std::vector<std::complex<float>> templ = test_helpers::randomComplex(40);
    for (std::complex<float>& sample : templ) {
        sample *= 37.0F; // arbitrary observation amplitude
    }
    const avx_optimized::ComplexFirFilter filter =
        avx_optimized::ComplexFirFilter::matchedFilter(templ, avx_optimized::TemplateScaling::UnitEnergy);

    double energy = 0.0;
    for (const std::complex<float>& tap : filter.taps()) {
        energy += std::norm(std::complex<double>(tap));
    }
    EXPECT_NEAR(energy, 1.0, 1e-6);
}

TEST(MatchedFilter, RejectsBadTemplates) {
    const avx_optimized::TemplateScaling asGiven    = avx_optimized::TemplateScaling::AsGiven;
    const avx_optimized::TemplateScaling unitEnergy = avx_optimized::TemplateScaling::UnitEnergy;

    EXPECT_THROW((void)avx_optimized::ComplexFirFilter::matchedFilter(std::vector<std::complex<float>>{}, asGiven),
                 std::invalid_argument);
    EXPECT_THROW((void)avx_optimized::ComplexFirFilter::matchedFilter(nullptr, 4, asGiven), std::invalid_argument);

    const std::vector<std::complex<float>> silent(8, std::complex<float>{});
    EXPECT_THROW((void)avx_optimized::ComplexFirFilter::matchedFilter(silent, unitEnergy), std::invalid_argument);
    EXPECT_NO_THROW((void)avx_optimized::ComplexFirFilter::matchedFilter(silent, asGiven)); // still a (zero) filter

    std::vector<std::complex<float>> infinite(8, std::complex<float>{1.0F, 0.0F});
    infinite[3] = {std::numeric_limits<float>::infinity(), 0.0F};
    EXPECT_THROW((void)avx_optimized::ComplexFirFilter::matchedFilter(infinite, unitEnergy), std::invalid_argument);
}

TEST_P(ComplexFir, MatchedFilterPeaksAtPulseEndWithAmplitudeAndPhase) {
    const std::vector<std::complex<float>> templ  = test_helpers::randomComplex(50);
    double                                 energy = 0.0;
    for (const std::complex<float>& sample : templ) {
        energy += std::norm(std::complex<double>(sample));
    }
    const std::complex<float> amplitude  = std::polar(0.7F, 1.1F);
    const std::size_t         pulseStart = 1000;

    std::vector<std::complex<float>> x(3000, std::complex<float>{});
    for (std::size_t n = 0; n < templ.size(); ++n) {
        x[pulseStart + n] = amplitude * templ[n];
    }

    for (const avx_optimized::TemplateScaling scaling :
         {avx_optimized::TemplateScaling::AsGiven, avx_optimized::TemplateScaling::UnitEnergy}) {
        avx_optimized::ComplexFirFilter  filter = avx_optimized::ComplexFirFilter::matchedFilter(templ, scaling);
        std::vector<std::complex<float>> y(x.size());
        filter.filter(x.data(), y.data(), x.size());

        const std::size_t peak = indexOfLargest(y);
        EXPECT_EQ(peak, pulseStart + filter.delay());

        const double gain = scaling == avx_optimized::TemplateScaling::AsGiven ? energy : std::sqrt(energy);
        const std::complex<double> expected = std::complex<double>(amplitude) * gain;
        EXPECT_NEAR(std::abs(std::complex<double>(y[peak]) - expected), 0.0, 1e-5 * gain);
        EXPECT_NEAR(std::arg(y[peak]), 1.1, 1e-5); // phase of the pulse relative to the template
    }
}

TEST_P(ComplexFir, MatchedFilterWithLearnedTemplateFindsPulsesAcrossChunks) {
    // The waveform is only known through a noisy observation; that observation is the template.
    std::normal_distribution<float>        noise(0.0F, 0.3F);
    const std::vector<std::complex<float>> truth = test_helpers::randomComplex(64);
    std::vector<std::complex<float>>       learned(truth.size());
    for (std::size_t n = 0; n < truth.size(); ++n) {
        learned[n] = 5.0F * truth[n] + std::complex<float>(noise(test_helpers::rng()), noise(test_helpers::rng()));
    }

    // A noisy stream with two weaker copies of the true waveform.
    std::vector<std::complex<float>> x(6000);
    for (std::complex<float>& sample : x) {
        sample = {noise(test_helpers::rng()), noise(test_helpers::rng())};
    }
    const std::size_t pulses[] = {1234, 4321};
    for (const std::size_t pulseStart : pulses) {
        for (std::size_t n = 0; n < truth.size(); ++n) {
            x[pulseStart + n] += truth[n];
        }
    }

    // Stream it in awkward chunk sizes so pulses straddle calls.
    avx_optimized::ComplexFirFilter filter =
        avx_optimized::ComplexFirFilter::matchedFilter(learned, avx_optimized::TemplateScaling::UnitEnergy);
    std::vector<std::complex<float>> y(x.size());
    for (std::size_t position = 0, i = 0; position < x.size(); ++i) {
        const std::size_t count = std::min<std::size_t>(i % 2 == 0 ? 37 : 1001, x.size() - position);
        filter.filter(x.data() + position, y.data() + position, count);
        position += count;
    }

    // The two largest outputs, at least a template length apart, are the pulse ends.
    std::vector<std::size_t> order(y.size());
    for (std::size_t k = 0; k < y.size(); ++k) {
        order[k] = k;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::abs(y[a]) > std::abs(y[b]);
    });
    std::vector<std::size_t> peaks;
    for (const std::size_t k : order) {
        bool separate = true;
        for (const std::size_t p : peaks) {
            separate = separate && (k > p ? k - p : p - k) > truth.size();
        }
        if (separate) {
            peaks.push_back(k);
        }
        if (peaks.size() == 2) {
            break;
        }
    }
    std::sort(peaks.begin(), peaks.end());
    EXPECT_EQ(peaks[0], pulses[0] + filter.delay());
    EXPECT_EQ(peaks[1], pulses[1] + filter.delay());
}

TEST_P(ComplexFir, MatchedFilterFlushFindsAPulseAtTheVeryEnd) {
    const std::vector<std::complex<float>> templ = test_helpers::randomComplex(40);

    // The pulse occupies the last 40 samples, so its peak only appears once flush() runs.
    std::vector<std::complex<float>> x(500, std::complex<float>{});
    const std::size_t                pulseStart = x.size() - templ.size();
    std::copy(templ.begin(), templ.end(), x.begin() + static_cast<std::ptrdiff_t>(pulseStart));

    avx_optimized::ComplexFirFilter filter =
        avx_optimized::ComplexFirFilter::matchedFilter(templ, avx_optimized::TemplateScaling::UnitEnergy);
    std::vector<std::complex<float>> y(x.size() + filter.delay());
    filter.filter(x.data(), y.data(), x.size());
    filter.flush(y.data() + x.size());

    EXPECT_EQ(indexOfLargest(y), pulseStart + filter.delay());
}

TEST_P(ComplexFir, UnitEnergyMatchedFilterPreservesWhiteNoisePower) {
    // Own generator, so the inputs don't change when other tests are added.  With 200k samples
    // (outputs are correlated over 33 samples) the estimate's standard deviation is about 0.026.
    std::mt19937                     engine(2024);
    std::normal_distribution<float>  noise(0.0F, 1.0F);
    std::vector<std::complex<float>> x(200000);
    for (std::complex<float>& sample : x) {
        sample = {noise(engine), noise(engine)}; // E|sample|^2 = 2
    }
    avx_optimized::ComplexFirFilter filter =
        avx_optimized::ComplexFirFilter::matchedFilter(test_helpers::randomComplex(33),
                                                       avx_optimized::TemplateScaling::UnitEnergy);
    std::vector<std::complex<float>> y(x.size());
    filter.filter(x.data(), y.data(), x.size());

    double power = 0.0;
    for (std::size_t k = filter.delay(); k < y.size(); ++k) {
        power += std::norm(std::complex<double>(y[k]));
    }
    power /= static_cast<double>(y.size() - filter.delay());
    EXPECT_NEAR(power, 2.0, 0.1);
}

TEST(ComplexFirFilterIsas, AllIsasAgree) {
    const std::vector<std::complex<float>> taps = test_helpers::randomComplex(45);
    const std::vector<std::complex<float>> x    = test_helpers::randomComplex(5000);
    for (const avx_optimized::Isa isa : test_helpers::supportedIsas()) {
        test_helpers::ScopedIsa          scoped(isa);
        avx_optimized::ComplexFirFilter  filter(taps);
        std::vector<std::complex<float>> y(x.size());
        filter.filter(x.data(), y.data(), x.size());
        EXPECT_TRUE(matchesReference(taps, x, y.data())) << avx_optimized::isaName(isa);
    }
}

INSTANTIATE_TEST_SUITE_P(AllIsas,
                         ComplexFir,
                         ::testing::ValuesIn(test_helpers::supportedIsas()),
                         test_helpers::isaTestName);
