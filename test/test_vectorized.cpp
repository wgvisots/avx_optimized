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

// Every function is checked against a straightforward double-precision reference, for every
// supported instruction set, every size in testSizes() and several misalignments, with guard
// zones around each output to catch out-of-bounds writes.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "avx_optimized/vectorized.h"
#include "test_helpers.h"

namespace {

std::string where(std::size_t count, std::size_t offset) {
    return "count=" + std::to_string(count) + " offset=" + std::to_string(offset);
}

constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr float kNaN      = std::numeric_limits<float>::quiet_NaN();

/// |actual - expected| <= relativeTolerance * scale, where scale defaults to |expected|.
::testing::AssertionResult
closeRelative(double actual, double expected, double relativeTolerance, double scale = -1.0) {
    if (scale < 0.0) {
        scale = std::fabs(expected);
    }
    const double tolerance = relativeTolerance * scale + 1e-30;
    if (std::fabs(actual - expected) <= tolerance) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "actual " << actual << " expected " << expected << " (difference "
                                         << std::fabs(actual - expected) << " > tolerance " << tolerance << ")";
}

class Vectorized : public test_helpers::IsaTest {};

} // namespace

// ---------------------------------------------------------------------------------------------
// Interleaved buffer helpers
// ---------------------------------------------------------------------------------------------

TEST(InterleavedHelpers, AsComplexAndAsFloatsViewTheSameMemory) {
    std::vector<float> interleaved = {1.0F, 2.0F, 3.0F, 4.0F};

    std::complex<float>* samples = avx_optimized::asComplex(interleaved.data());
    EXPECT_EQ(samples[0], std::complex<float>(1.0F, 2.0F));
    EXPECT_EQ(samples[1], std::complex<float>(3.0F, 4.0F));
    EXPECT_EQ(avx_optimized::asFloats(samples), interleaved.data());

    const std::vector<float>&  constant     = interleaved;
    const std::complex<float>* constSamples = avx_optimized::asComplex(constant.data());
    const float*               constFloats  = avx_optimized::asFloats(constSamples);
    EXPECT_EQ(constFloats, constant.data());
}

// ---------------------------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------------------------

TEST_P(Vectorized, Int16ToFloatScalesExactly) {
    for (const float scale : {1.0F, 1.0F / 32768.0F, -0.37F}) {
        for (std::size_t count : test_helpers::testSizes()) {
            for (std::size_t offset : test_helpers::kOffsets) {
                SCOPED_TRACE(where(count, offset) + " scale=" + std::to_string(scale));
                std::vector<std::int16_t> source = test_helpers::randomInt16(count);
                if (count > 1) {
                    source[0]         = -32768;
                    source[count - 1] = 32767;
                }
                test_helpers::GuardedBuffer<std::int16_t> in(count, offset);
                in.fill(source);
                test_helpers::GuardedBuffer<float> out(count, offset);

                avx_optimized::int16ToFloat(in.data(), out.data(), count, scale);

                ASSERT_TRUE(out.intact());
                for (std::size_t i = 0; i < count; ++i) {
                    ASSERT_EQ(out[i], static_cast<float>(source[i]) * scale) << "i=" << i;
                }
            }
        }
    }
}

namespace {

std::int16_t referenceToInt16(float x, float scale) {
    const float scaled = x * scale;
    if (std::isnan(scaled)) {
        return 0;
    }
    return static_cast<std::int16_t>(std::nearbyint(std::clamp(static_cast<double>(scaled), -32768.0, 32767.0)));
}

} // namespace

TEST_P(Vectorized, FloatToInt16RoundsSaturatesAndIsExact) {
    const std::vector<float> specials = {0.0F,      -0.0F,    0.5F,     1.5F,      2.5F,       -2.5F,     3.5F,
                                         -3.5F,     32766.5F, 32767.4F, 32767.5F,  32768.0F,   -32767.5F, -32768.5F,
                                         -32769.0F, 1e10F,    -1e10F,   kInfinity, -kInfinity, kNaN,      1e-30F};
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            std::vector<float> source = test_helpers::randomFloats(count, -40000.0F, 40000.0F);
            for (std::size_t i = 0; i < count; i += 3) { // sprinkle edge cases through body and tail
                source[i] = specials[(i / 3) % specials.size()];
            }
            test_helpers::GuardedBuffer<float> in(count, offset);
            in.fill(source);
            test_helpers::GuardedBuffer<std::int16_t> out(count, offset);

            avx_optimized::floatToInt16(in.data(), out.data(), count);

            ASSERT_TRUE(out.intact());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_EQ(out[i], referenceToInt16(source[i], 1.0F)) << "i=" << i << " input=" << source[i];
            }
        }
    }
}

TEST_P(Vectorized, FloatToInt16AppliesScaleBeforeRounding) {
    for (std::size_t count : test_helpers::testSizes()) {
        SCOPED_TRACE(where(count, 1));
        const std::vector<float>                  source = test_helpers::randomFloats(count, -1.2F, 1.2F);
        test_helpers::GuardedBuffer<float>        in(count, 1);
        test_helpers::GuardedBuffer<std::int16_t> out(count, 1);
        in.fill(source);

        avx_optimized::floatToInt16(in.data(), out.data(), count, 32767.0F);

        ASSERT_TRUE(out.intact());
        for (std::size_t i = 0; i < count; ++i) {
            ASSERT_EQ(out[i], referenceToInt16(source[i], 32767.0F)) << "i=" << i << " input=" << source[i];
        }
    }
}

TEST_P(Vectorized, Int16ToComplexCountsSamples) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<std::int16_t>                  source = test_helpers::randomInt16(2 * count);
            test_helpers::GuardedBuffer<std::int16_t>        in(2 * count, offset);
            test_helpers::GuardedBuffer<std::complex<float>> out(count, offset);
            in.fill(source);

            avx_optimized::int16ToComplex(in.data(), out.data(), count, 1.0F / 32768.0F);

            ASSERT_TRUE(out.intact());
            for (std::size_t k = 0; k < count; ++k) {
                ASSERT_EQ(out[k].real(), static_cast<float>(source[2 * k]) / 32768.0F) << "k=" << k;
                ASSERT_EQ(out[k].imag(), static_cast<float>(source[2 * k + 1]) / 32768.0F) << "k=" << k;
            }
        }
    }
}

TEST_P(Vectorized, ComplexToInt16CountsSamplesAndRoundTrips) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<std::int16_t>           original = test_helpers::randomInt16(2 * count);
            std::vector<std::complex<float>>          samples(count);
            test_helpers::GuardedBuffer<std::int16_t> back(2 * count, offset);

            // A power-of-two scale is exact both ways, so this must round-trip bit for bit.
            avx_optimized::int16ToComplex(original.data(), samples.data(), count, 1.0F / 32768.0F);
            avx_optimized::complexToInt16(samples.data(), back.data(), count, 32768.0F);

            ASSERT_TRUE(back.intact());
            for (std::size_t i = 0; i < 2 * count; ++i) {
                ASSERT_EQ(back[i], original[i]) << "i=" << i;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Element-wise arithmetic
// ---------------------------------------------------------------------------------------------

TEST_P(Vectorized, AddIsExact) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<float>           a = test_helpers::randomFloats(count, -100.0F, 100.0F);
            const std::vector<float>           b = test_helpers::randomFloats(count, -100.0F, 100.0F);
            test_helpers::GuardedBuffer<float> io(count, offset);
            test_helpers::GuardedBuffer<float> other(count, (offset + 1) % 4); // different misalignment from io
            io.fill(a);
            other.fill(b);

            avx_optimized::add(io.data(), other.data(), count);

            ASSERT_TRUE(io.intact());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_EQ(io[i], a[i] + b[i]) << "i=" << i;
            }
        }
    }
}

TEST_P(Vectorized, AddComplexIsExact) {
    for (std::size_t count : test_helpers::testSizes()) {
        SCOPED_TRACE(where(count, 3));
        const std::vector<std::complex<float>>           a = test_helpers::randomComplex(count, -100.0F, 100.0F);
        const std::vector<std::complex<float>>           b = test_helpers::randomComplex(count, -100.0F, 100.0F);
        test_helpers::GuardedBuffer<std::complex<float>> io(count, 3);
        io.fill(a);

        avx_optimized::add(io.data(), b.data(), count);

        ASSERT_TRUE(io.intact());
        for (std::size_t i = 0; i < count; ++i) {
            ASSERT_EQ(io[i], a[i] + b[i]) << "i=" << i;
        }
    }
}

TEST_P(Vectorized, AddAllowsOtherToBeIo) {
    std::vector<float> a        = test_helpers::randomFloats(1001);
    std::vector<float> expected = a;
    for (float& x : expected) {
        x += x;
    }
    avx_optimized::add(a.data(), a.data(), a.size());
    EXPECT_EQ(a, expected);
}

TEST_P(Vectorized, AddSaturatedIsExact) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<std::int16_t> a = test_helpers::randomInt16(count); // full range: many sums saturate
            const std::vector<std::int16_t> b = test_helpers::randomInt16(count);
            test_helpers::GuardedBuffer<std::int16_t> io(count, offset);
            test_helpers::GuardedBuffer<std::int16_t> other(count, offset);
            io.fill(a);
            other.fill(b);

            avx_optimized::addSaturated(io.data(), other.data(), count);

            ASSERT_TRUE(io.intact());
            for (std::size_t i = 0; i < count; ++i) {
                const int expected = std::clamp(int{a[i]} + int{b[i]}, -32768, 32767);
                ASSERT_EQ(io[i], expected) << "i=" << i << " a=" << a[i] << " b=" << b[i];
            }
        }
    }
}

TEST_P(Vectorized, ScaleIsExact) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<float>           a = test_helpers::randomFloats(count, -100.0F, 100.0F);
            test_helpers::GuardedBuffer<float> io(count, offset);
            io.fill(a);

            avx_optimized::scale(io.data(), -1.75F, count);

            ASSERT_TRUE(io.intact());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_EQ(io[i], a[i] * -1.75F) << "i=" << i;
            }
        }
    }
}

TEST_P(Vectorized, ScaleComplexIsExact) {
    for (std::size_t count : test_helpers::testSizes()) {
        SCOPED_TRACE(where(count, 1));
        const std::vector<std::complex<float>>           a = test_helpers::randomComplex(count, -100.0F, 100.0F);
        test_helpers::GuardedBuffer<std::complex<float>> io(count, 1);
        io.fill(a);

        avx_optimized::scale(io.data(), 0.3F, count);

        ASSERT_TRUE(io.intact());
        for (std::size_t i = 0; i < count; ++i) {
            ASSERT_EQ(io[i].real(), a[i].real() * 0.3F) << "i=" << i;
            ASSERT_EQ(io[i].imag(), a[i].imag() * 0.3F) << "i=" << i;
        }
    }
}

TEST_P(Vectorized, ReverseIsExact) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<float>           a = test_helpers::randomFloats(count);
            test_helpers::GuardedBuffer<float> in(count, offset);
            test_helpers::GuardedBuffer<float> out(count, (offset + 2) % 4);
            in.fill(a);

            avx_optimized::reverse(in.data(), out.data(), count);

            ASSERT_TRUE(out.intact());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_EQ(out[count - 1 - i], a[i]) << "i=" << i;
            }
        }
    }
}

TEST_P(Vectorized, ScaleIQIsExact) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<std::complex<float>>           a = test_helpers::randomComplex(count, -10.0F, 10.0F);
            test_helpers::GuardedBuffer<std::complex<float>> io(count, offset);
            io.fill(a);

            avx_optimized::scaleIQ(io.data(), 0.25F, -3.0F, count);

            ASSERT_TRUE(io.intact());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_EQ(io[i].real(), a[i].real() * 0.25F) << "i=" << i;
                ASSERT_EQ(io[i].imag(), a[i].imag() * -3.0F) << "i=" << i;
            }
        }
    }
}

TEST_P(Vectorized, ComplexMultiplyMatchesReference) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<std::complex<float>>           a = test_helpers::randomComplex(count, -10.0F, 10.0F);
            const std::vector<std::complex<float>>           b = test_helpers::randomComplex(count, -10.0F, 10.0F);
            test_helpers::GuardedBuffer<std::complex<float>> io(count, offset);
            test_helpers::GuardedBuffer<std::complex<float>> other(count, offset);
            io.fill(a);
            other.fill(b);

            avx_optimized::complexMultiply(io.data(), other.data(), count);

            ASSERT_TRUE(io.intact());
            for (std::size_t i = 0; i < count; ++i) {
                const std::complex<double> expected = std::complex<double>(a[i]) * std::complex<double>(b[i]);
                const double bound = std::abs(std::complex<double>(a[i])) * std::abs(std::complex<double>(b[i]));
                ASSERT_TRUE(closeRelative(io[i].real(), expected.real(), 4e-7, bound)) << "i=" << i;
                ASSERT_TRUE(closeRelative(io[i].imag(), expected.imag(), 4e-7, bound)) << "i=" << i;
            }
        }
    }
}

TEST_P(Vectorized, ComplexMultiplyAllowsOtherToBeIo) {
    std::vector<std::complex<float>>       a        = test_helpers::randomComplex(999);
    const std::vector<std::complex<float>> original = a;
    avx_optimized::complexMultiply(a.data(), a.data(), a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::complex<double> expected = std::complex<double>(original[i]) * std::complex<double>(original[i]);
        const double               bound    = std::norm(std::complex<double>(original[i]));
        EXPECT_TRUE(closeRelative(a[i].real(), expected.real(), 4e-7, bound));
        EXPECT_TRUE(closeRelative(a[i].imag(), expected.imag(), 4e-7, bound));
    }
}

TEST_P(Vectorized, LinearGainRampMatchesReference) {
    const std::pair<float, float> ramps[] = {
        {0.5F, 2.0F },
        {3.0F, -1.0F},
        {1.0F, 1.0F }
    };
    for (const auto& [startGain, endGain] : ramps) {
        for (std::size_t count : test_helpers::testSizes()) {
            for (std::size_t offset : test_helpers::kOffsets) {
                SCOPED_TRACE(where(count, offset) + " ramp " + std::to_string(startGain) + "->"
                             + std::to_string(endGain));
                const std::vector<std::complex<float>>           a = test_helpers::randomComplex(count, -10.0F, 10.0F);
                test_helpers::GuardedBuffer<std::complex<float>> io(count, offset);
                io.fill(a);

                avx_optimized::applyLinearGainRamp(io.data(), startGain, endGain, count);

                ASSERT_TRUE(io.intact());
                const double gainScale = std::max(std::fabs(startGain), std::fabs(endGain));
                for (std::size_t k = 0; k < count; ++k) {
                    const double t     = count > 1 ? static_cast<double>(k) / static_cast<double>(count - 1) : 0.0;
                    const double gain  = startGain + (endGain - startGain) * t;
                    const double bound = gainScale * std::abs(std::complex<double>(a[k]));
                    ASSERT_TRUE(closeRelative(io[k].real(), a[k].real() * gain, 1e-6, bound)) << "k=" << k;
                    ASSERT_TRUE(closeRelative(io[k].imag(), a[k].imag() * gain, 1e-6, bound)) << "k=" << k;
                }
                if (count > 0) { // the first sample gets exactly startGain
                    ASSERT_EQ(io[0], a[0] * startGain);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Magnitude, phase and power
// ---------------------------------------------------------------------------------------------

TEST_P(Vectorized, MagnitudeSquaredMatchesReference) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<std::complex<float>>           a = test_helpers::randomComplex(count, -100.0F, 100.0F);
            test_helpers::GuardedBuffer<std::complex<float>> in(count, offset);
            test_helpers::GuardedBuffer<float>               out(count, offset);
            in.fill(a);

            avx_optimized::magnitudeSquared(in.data(), out.data(), count);

            ASSERT_TRUE(out.intact());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_TRUE(closeRelative(out[i], std::norm(std::complex<double>(a[i])), 3e-7)) << "i=" << i;
            }
        }
    }
}

TEST_P(Vectorized, MagnitudeMatchesReference) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<std::complex<float>>           a = test_helpers::randomComplex(count, -100.0F, 100.0F);
            test_helpers::GuardedBuffer<std::complex<float>> in(count, offset);
            test_helpers::GuardedBuffer<float>               out(count, offset);
            in.fill(a);

            avx_optimized::magnitude(in.data(), out.data(), count);

            ASSERT_TRUE(out.intact());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_TRUE(closeRelative(out[i], std::abs(std::complex<double>(a[i])), 3e-7)) << "i=" << i;
            }
        }
    }
}

namespace {
// Phase tolerance in radians: a few float ulps at pi (ulp(pi) ~ 2.4e-7).
constexpr double kPhaseTolerance = 1e-6;
} // namespace

TEST_P(Vectorized, ToPolarMatchesReference) {
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            const std::vector<std::complex<float>>           a = test_helpers::randomComplex(count, -100.0F, 100.0F);
            test_helpers::GuardedBuffer<std::complex<float>> in(count, offset);
            test_helpers::GuardedBuffer<float>               magnitude(count, offset);
            test_helpers::GuardedBuffer<float>               phase(count, (offset + 1) % 4);
            in.fill(a);

            avx_optimized::toPolar(in.data(), magnitude.data(), phase.data(), count);

            ASSERT_TRUE(magnitude.intact());
            ASSERT_TRUE(phase.intact());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_TRUE(closeRelative(magnitude[i], std::abs(std::complex<double>(a[i])), 3e-7)) << "i=" << i;
                ASSERT_NEAR(phase[i], std::arg(std::complex<double>(a[i])), kPhaseTolerance) << "i=" << i;
            }
        }
    }
}

TEST_P(Vectorized, ToPolarSpecialValuesFollowAtan2) {
    // Axes, signed zeros (which pick +pi vs -pi on the negative real axis), infinities and tiny values.
    const std::vector<std::complex<float>> values = {
        {0.0F,       0.0F      },
        {-0.0F,      0.0F      },
        {0.0F,       -0.0F     },
        {-0.0F,      -0.0F     },
        {1.0F,       0.0F      },
        {-1.0F,      0.0F      },
        {-1.0F,      -0.0F     },
        {0.0F,       1.0F      },
        {0.0F,       -1.0F     },
        {kInfinity,  1.0F      },
        {-kInfinity, 1.0F      },
        {1.0F,       kInfinity },
        {kInfinity,  kInfinity },
        {-kInfinity, -kInfinity},
        {1e-38F,     1e-38F    },
        {-3.0F,      4.0F      },
        {3.0F,       -4.0F     },
        {-1e30F,     1e-30F    }
    };
    std::vector<float> magnitude(values.size());
    std::vector<float> phase(values.size());

    avx_optimized::toPolar(values.data(), magnitude.data(), phase.data(), values.size());

    for (std::size_t i = 0; i < values.size(); ++i) {
        SCOPED_TRACE("z=(" + std::to_string(values[i].real()) + "," + std::to_string(values[i].imag()) + ")");
        const float expected = std::atan2(values[i].imag(), values[i].real());
        EXPECT_NEAR(phase[i], expected, kPhaseTolerance);
        EXPECT_EQ(std::signbit(phase[i]), std::signbit(expected));
    }
    EXPECT_EQ(magnitude[0], 0.0F);
    EXPECT_EQ(magnitude[5], 1.0F);
    EXPECT_EQ(magnitude[9], kInfinity);
    EXPECT_EQ(magnitude[15], 5.0F);
}

namespace {

/// 10 log10(p) in double, with the special cases the API documents.
double referenceDecibels(float power) {
    if (std::isnan(power) || power < 0.0F) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (power == 0.0F) {
        return -std::numeric_limits<double>::infinity();
    }
    return 10.0 * std::log10(static_cast<double>(power));
}

::testing::AssertionResult decibelsMatch(float actual, float power) {
    const double expected = referenceDecibels(power);
    if (std::isnan(expected)) {
        return std::isnan(actual) ? ::testing::AssertionSuccess()
                                  : ::testing::AssertionFailure() << "expected NaN, got " << actual;
    }
    if (std::isinf(expected)) {
        return static_cast<double>(actual) == expected
                 ? ::testing::AssertionSuccess()
                 : ::testing::AssertionFailure() << "expected " << expected << ", got " << actual;
    }
    // About 1e-5 dB near 0 dB, growing with the magnitude of the result (float has ~7 digits).
    return closeRelative(actual, expected, 1e-6, std::fabs(expected) + 20.0);
}

} // namespace

TEST_P(Vectorized, PowerToDecibelsMatchesReference) {
    std::mt19937                          engine(7);
    std::uniform_real_distribution<float> exponent(-45.0F, 38.0F); // includes subnormals
    const std::vector<float>              specials = {0.0F,
                                                      -0.0F,
                                                      1.0F,
                                                      10.0F,
                                                      0.1F,
                                                      kInfinity,
                                                      -kInfinity,
                                                      kNaN,
                                                      -1.0F,
                                                      std::numeric_limits<float>::denorm_min(),
                                                      std::numeric_limits<float>::min(),
                                                      std::numeric_limits<float>::max(),
                                                      1e-40F};
    for (std::size_t count : test_helpers::testSizes()) {
        for (std::size_t offset : test_helpers::kOffsets) {
            SCOPED_TRACE(where(count, offset));
            std::vector<float> power(count);
            for (std::size_t i = 0; i < count; ++i) {
                power[i] = i % 4 == 0 ? specials[(i / 4) % specials.size()] : std::pow(10.0F, exponent(engine));
            }
            test_helpers::GuardedBuffer<float> in(count, offset);
            test_helpers::GuardedBuffer<float> out(count, offset);
            in.fill(power);

            avx_optimized::powerToDecibels(in.data(), out.data(), count);

            ASSERT_TRUE(out.intact());
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_TRUE(decibelsMatch(out[i], power[i])) << "i=" << i << " power=" << power[i];
            }
        }
    }
}

TEST_P(Vectorized, PowerToDecibelsOfComplexSamples) {
    for (std::size_t count : test_helpers::testSizes()) {
        SCOPED_TRACE(where(count, 1));
        std::vector<std::complex<float>> samples = test_helpers::randomComplex(count, -3.0F, 3.0F);
        if (count > 2) {
            samples[1] = {0.0F, 0.0F}; // -infinity dB
        }
        test_helpers::GuardedBuffer<float> out(count, 1);

        avx_optimized::powerToDecibels(samples.data(), out.data(), count);

        ASSERT_TRUE(out.intact());
        for (std::size_t i = 0; i < count; ++i) {
            const float power = static_cast<float>(std::norm(std::complex<double>(samples[i])));
            ASSERT_TRUE(decibelsMatch(out[i], power)) << "i=" << i;
        }
    }
}

TEST_P(Vectorized, ZeroCountAcceptsNullPointers) {
    // Must not touch memory at all.
    avx_optimized::int16ToFloat(nullptr, nullptr, 0);
    avx_optimized::floatToInt16(nullptr, nullptr, 0);
    avx_optimized::int16ToComplex(nullptr, nullptr, 0);
    avx_optimized::complexToInt16(nullptr, nullptr, 0);
    avx_optimized::add(static_cast<float*>(nullptr), nullptr, 0);
    avx_optimized::add(static_cast<std::complex<float>*>(nullptr), nullptr, 0);
    avx_optimized::addSaturated(nullptr, nullptr, 0);
    avx_optimized::scale(static_cast<float*>(nullptr), 2.0F, 0);
    avx_optimized::scale(static_cast<std::complex<float>*>(nullptr), 2.0F, 0);
    avx_optimized::reverse(nullptr, nullptr, 0);
    avx_optimized::scaleIQ(nullptr, 1.0F, 2.0F, 0);
    avx_optimized::complexMultiply(nullptr, nullptr, 0);
    avx_optimized::applyLinearGainRamp(nullptr, 1.0F, 2.0F, 0);
    avx_optimized::magnitudeSquared(nullptr, nullptr, 0);
    avx_optimized::magnitude(nullptr, nullptr, 0);
    avx_optimized::toPolar(nullptr, nullptr, nullptr, 0);
    avx_optimized::powerToDecibels(static_cast<const float*>(nullptr), nullptr, 0);
    avx_optimized::powerToDecibels(static_cast<const std::complex<float>*>(nullptr), nullptr, 0);
    SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(AllIsas,
                         Vectorized,
                         ::testing::ValuesIn(test_helpers::supportedIsas()),
                         test_helpers::isaTestName);
