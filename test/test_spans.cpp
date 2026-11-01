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

// The std::span forms: identical results to the pointer forms (which test_vectorized.cpp and
// test_complex_fir_filter.cpp check against references), size checks that throw before anything
// is written, and acceptance of the usual containers.

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "avx_optimized/aligned.h"
#include "avx_optimized/complex_fir_filter.h"
#include "avx_optimized/vectorized.h"
#include "test_helpers.h"

namespace {

constexpr std::size_t kCount = 37; // not a multiple of any vector width

} // namespace

// ---------------------------------------------------------------------------------------------
// Same results as the pointer forms
// ---------------------------------------------------------------------------------------------

TEST(Spans, ConversionsMatchPointerForms) {
    const std::vector<std::int16_t>        shorts  = test_helpers::randomInt16(2 * kCount);
    const std::vector<float>               floats  = test_helpers::randomFloats(kCount, -2.0F, 2.0F);
    const std::vector<std::complex<float>> samples = test_helpers::randomComplex(kCount);

    std::vector<float> spanFloats(2 * kCount);
    std::vector<float> pointerFloats(2 * kCount);
    avx_optimized::int16ToFloat(shorts, spanFloats, 0.5F);
    avx_optimized::int16ToFloat(shorts.data(), pointerFloats.data(), shorts.size(), 0.5F);
    EXPECT_EQ(spanFloats, pointerFloats);

    std::vector<std::int16_t> spanShorts(kCount);
    std::vector<std::int16_t> pointerShorts(kCount);
    avx_optimized::floatToInt16(floats, spanShorts, 1000.0F);
    avx_optimized::floatToInt16(floats.data(), pointerShorts.data(), floats.size(), 1000.0F);
    EXPECT_EQ(spanShorts, pointerShorts);

    std::vector<std::complex<float>> spanSamples(kCount);
    std::vector<std::complex<float>> pointerSamples(kCount);
    avx_optimized::int16ToComplex(shorts, spanSamples, 1.0F / 32768.0F);
    avx_optimized::int16ToComplex(shorts.data(), pointerSamples.data(), kCount, 1.0F / 32768.0F);
    EXPECT_EQ(spanSamples, pointerSamples);

    std::vector<std::int16_t> spanIQ(2 * kCount);
    std::vector<std::int16_t> pointerIQ(2 * kCount);
    avx_optimized::complexToInt16(samples, spanIQ, 32767.0F);
    avx_optimized::complexToInt16(samples.data(), pointerIQ.data(), kCount, 32767.0F);
    EXPECT_EQ(spanIQ, pointerIQ);
}

TEST(Spans, ArithmeticMatchesPointerForms) {
    const std::vector<float>               a  = test_helpers::randomFloats(kCount);
    const std::vector<float>               b  = test_helpers::randomFloats(kCount);
    const std::vector<std::complex<float>> ca = test_helpers::randomComplex(kCount);
    const std::vector<std::complex<float>> cb = test_helpers::randomComplex(kCount);

    std::vector<float> spanResult    = a;
    std::vector<float> pointerResult = a;
    avx_optimized::add(spanResult, b);
    avx_optimized::add(pointerResult.data(), b.data(), kCount);
    EXPECT_EQ(spanResult, pointerResult);

    avx_optimized::scale(spanResult, 3.0F);
    avx_optimized::scale(pointerResult.data(), 3.0F, kCount);
    EXPECT_EQ(spanResult, pointerResult);

    std::vector<float> spanReversed(kCount);
    std::vector<float> pointerReversed(kCount);
    avx_optimized::reverse(a, spanReversed);
    avx_optimized::reverse(a.data(), pointerReversed.data(), kCount);
    EXPECT_EQ(spanReversed, pointerReversed);

    std::vector<std::complex<float>> spanComplex    = ca;
    std::vector<std::complex<float>> pointerComplex = ca;
    avx_optimized::add(spanComplex, cb);
    avx_optimized::add(pointerComplex.data(), cb.data(), kCount);
    avx_optimized::scale(spanComplex, 0.5F);
    avx_optimized::scale(pointerComplex.data(), 0.5F, kCount);
    avx_optimized::scaleIQ(spanComplex, 2.0F, -1.0F);
    avx_optimized::scaleIQ(pointerComplex.data(), 2.0F, -1.0F, kCount);
    avx_optimized::complexMultiply(spanComplex, cb);
    avx_optimized::complexMultiply(pointerComplex.data(), cb.data(), kCount);
    avx_optimized::applyLinearGainRamp(spanComplex, 0.5F, 2.0F);
    avx_optimized::applyLinearGainRamp(pointerComplex.data(), 0.5F, 2.0F, kCount);
    EXPECT_EQ(spanComplex, pointerComplex);

    std::vector<std::int16_t>       spanShorts    = test_helpers::randomInt16(kCount);
    std::vector<std::int16_t>       pointerShorts = spanShorts;
    const std::vector<std::int16_t> other         = test_helpers::randomInt16(kCount);
    avx_optimized::addSaturated(spanShorts, other);
    avx_optimized::addSaturated(pointerShorts.data(), other.data(), kCount);
    EXPECT_EQ(spanShorts, pointerShorts);
}

TEST(Spans, MagnitudePhaseAndPowerMatchPointerForms) {
    const std::vector<std::complex<float>> samples = test_helpers::randomComplex(kCount);
    const std::vector<float>               power   = test_helpers::randomFloats(kCount, 0.0F, 10.0F);

    std::vector<float> spanOut(kCount);
    std::vector<float> pointerOut(kCount);

    avx_optimized::magnitudeSquared(samples, spanOut);
    avx_optimized::magnitudeSquared(samples.data(), pointerOut.data(), kCount);
    EXPECT_EQ(spanOut, pointerOut);

    avx_optimized::magnitude(samples, spanOut);
    avx_optimized::magnitude(samples.data(), pointerOut.data(), kCount);
    EXPECT_EQ(spanOut, pointerOut);

    avx_optimized::powerToDecibels(samples, spanOut);
    avx_optimized::powerToDecibels(samples.data(), pointerOut.data(), kCount);
    EXPECT_EQ(spanOut, pointerOut);

    avx_optimized::powerToDecibels(power, spanOut);
    avx_optimized::powerToDecibels(power.data(), pointerOut.data(), kCount);
    EXPECT_EQ(spanOut, pointerOut);

    std::vector<float> spanPhase(kCount);
    std::vector<float> pointerPhase(kCount);
    avx_optimized::toPolar(samples, spanOut, spanPhase);
    avx_optimized::toPolar(samples.data(), pointerOut.data(), pointerPhase.data(), kCount);
    EXPECT_EQ(spanOut, pointerOut);
    EXPECT_EQ(spanPhase, pointerPhase);
}

// ---------------------------------------------------------------------------------------------
// Size checks
// ---------------------------------------------------------------------------------------------

TEST(Spans, MismatchedSizesThrowWithoutWriting) {
    const std::vector<std::int16_t>        shorts(10, 7);
    const std::vector<float>               floats(10, 1.0F);
    const std::vector<std::complex<float>> samples(10, std::complex<float>(1.0F, 1.0F));

    std::vector<float>               floatsOut(9, -5.0F);
    std::vector<std::int16_t>        shortsOut(9, -5);
    std::vector<std::complex<float>> samplesOut(9, std::complex<float>(-5.0F, -5.0F));

    EXPECT_THROW(avx_optimized::int16ToFloat(shorts, floatsOut), std::invalid_argument);
    EXPECT_THROW(avx_optimized::floatToInt16(floats, shortsOut), std::invalid_argument);
    EXPECT_THROW(avx_optimized::int16ToComplex(shorts, samplesOut), std::invalid_argument); // needs 18 values
    EXPECT_THROW(avx_optimized::complexToInt16(samples, shortsOut), std::invalid_argument); // needs 20 values
    EXPECT_THROW(avx_optimized::add(floatsOut, floats), std::invalid_argument);
    EXPECT_THROW(avx_optimized::add(samplesOut, samples), std::invalid_argument);
    EXPECT_THROW(avx_optimized::addSaturated(shortsOut, shorts), std::invalid_argument);
    EXPECT_THROW(avx_optimized::complexMultiply(samplesOut, samples), std::invalid_argument);
    EXPECT_THROW(avx_optimized::reverse(floats, floatsOut), std::invalid_argument);
    EXPECT_THROW(avx_optimized::magnitudeSquared(samples, floatsOut), std::invalid_argument);
    EXPECT_THROW(avx_optimized::magnitude(samples, floatsOut), std::invalid_argument);
    EXPECT_THROW(avx_optimized::powerToDecibels(samples, floatsOut), std::invalid_argument);
    EXPECT_THROW(avx_optimized::powerToDecibels(floats, floatsOut), std::invalid_argument);

    std::vector<float> phase(10);
    EXPECT_THROW(avx_optimized::toPolar(samples, floatsOut, phase), std::invalid_argument);
    EXPECT_THROW(avx_optimized::toPolar(samples, phase, floatsOut), std::invalid_argument);

    // Nothing was written by any of the failed calls.
    EXPECT_EQ(floatsOut, std::vector<float>(9, -5.0F));
    EXPECT_EQ(shortsOut, std::vector<std::int16_t>(9, -5));
    EXPECT_EQ(samplesOut, std::vector<std::complex<float>>(9, std::complex<float>(-5.0F, -5.0F)));
}

TEST(Spans, ErrorMessagesNameTheFunctionAndSizes) {
    const std::vector<std::complex<float>> samples(10);
    std::vector<float>                     out(9);
    try {
        avx_optimized::magnitude(samples, out);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "avx_optimized::magnitude: out has 9 elements but must have 10");
    }
}

TEST(Spans, InterleavedIQSizesCountTwoValuesPerSample) {
    const std::vector<std::int16_t>  iq(20, 100);
    std::vector<std::complex<float>> samples(10);
    EXPECT_NO_THROW(avx_optimized::int16ToComplex(iq, samples));

    std::vector<std::int16_t> back(20);
    EXPECT_NO_THROW(avx_optimized::complexToInt16(samples, back));
    EXPECT_EQ(back, iq);
}

// ---------------------------------------------------------------------------------------------
// Containers
// ---------------------------------------------------------------------------------------------

TEST(Spans, AcceptCommonContainers) {
    // std::array, C array, AlignedVector and a subspan, all without explicit conversions.
    const std::array<float, 8>          array     = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
    float                               cArray[8] = {};
    avx_optimized::AlignedVector<float> aligned(8);

    avx_optimized::reverse(array, cArray);
    avx_optimized::reverse(cArray, aligned);
    EXPECT_TRUE(std::equal(aligned.begin(), aligned.end(), array.begin()));

    std::span<float> firstHalf = std::span<float>(aligned).first(4);
    avx_optimized::scale(firstHalf, 10.0F);
    EXPECT_EQ(aligned[0], 10.0F);
    EXPECT_EQ(aligned[3], 40.0F);
    EXPECT_EQ(aligned[4], 5.0F);
}

TEST(Spans, EmptySpansAreFine) {
    std::vector<float>               noFloats;
    std::vector<std::complex<float>> noSamples;
    avx_optimized::add(noFloats, noFloats);
    avx_optimized::magnitude(noSamples, noFloats);
    avx_optimized::scale(noSamples, 2.0F);
    SUCCEED();
}

// ---------------------------------------------------------------------------------------------
// ComplexFirFilter
// ---------------------------------------------------------------------------------------------

TEST(Spans, FilterConstructorsAcceptContainers) {
    const std::array<float, 3>                              realArray = {0.25F, 0.5F, 0.25F};
    const avx_optimized::AlignedVector<std::complex<float>> complexTaps(5, std::complex<float>(0.2F, 0.1F));

    const avx_optimized::ComplexFirFilter fromArray(realArray);
    const avx_optimized::ComplexFirFilter fromAligned(complexTaps);
    const avx_optimized::ComplexFirFilter fromSubspan(std::span<const float>(realArray).first(2));
    const avx_optimized::ComplexFirFilter matched =
        avx_optimized::ComplexFirFilter::matchedFilter(complexTaps, avx_optimized::TemplateScaling::UnitEnergy);

    EXPECT_EQ(fromArray.numTaps(), 3U);
    EXPECT_TRUE(fromArray.hasRealTaps());
    EXPECT_EQ(fromAligned.numTaps(), 5U);
    EXPECT_EQ(fromSubspan.numTaps(), 2U);
    EXPECT_EQ(matched.numTaps(), 5U);
}

TEST(Spans, FilterAndFlushMatchPointerForms) {
    const std::vector<std::complex<float>> taps = test_helpers::randomComplex(9);
    const std::vector<std::complex<float>> x    = test_helpers::randomComplex(200);

    avx_optimized::ComplexFirFilter  pointerFilter(taps);
    std::vector<std::complex<float>> expected(x.size() + pointerFilter.delay());
    pointerFilter.filter(x.data(), expected.data(), x.size());
    pointerFilter.flush(expected.data() + x.size());

    // Out of place, then flush into a larger buffer: flush returns exactly the written part.
    avx_optimized::ComplexFirFilter  spanFilter(taps);
    std::vector<std::complex<float>> out(x.size());
    std::vector<std::complex<float>> tailBuffer(64);
    spanFilter.filter(x, out);
    const std::span<std::complex<float>> tail = spanFilter.flush(tailBuffer);
    EXPECT_EQ(tail.size(), spanFilter.delay());
    EXPECT_EQ(tail.data(), tailBuffer.data());
    EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin()));
    EXPECT_TRUE(std::equal(tail.begin(), tail.end(), expected.begin() + static_cast<std::ptrdiff_t>(x.size())));

    // In place.
    avx_optimized::ComplexFirFilter  inPlaceFilter(taps);
    std::vector<std::complex<float>> data = x;
    inPlaceFilter.filter(data);
    EXPECT_EQ(data, out);
}

TEST(Spans, FilterSizeChecks) {
    avx_optimized::ComplexFirFilter        filter(test_helpers::randomComplex(9)); // delay() == 8
    const std::vector<std::complex<float>> in(10);
    std::vector<std::complex<float>>       shortOut(9, std::complex<float>(-5.0F, 0.0F));
    std::vector<std::complex<float>>       tooSmallForFlush(7, std::complex<float>(-5.0F, 0.0F));

    EXPECT_THROW(filter.filter(in, shortOut), std::invalid_argument);
    EXPECT_THROW((void)filter.flush(tooSmallForFlush), std::invalid_argument);
    EXPECT_EQ(shortOut, std::vector<std::complex<float>>(9, std::complex<float>(-5.0F, 0.0F)));
    EXPECT_EQ(tooSmallForFlush, std::vector<std::complex<float>>(7, std::complex<float>(-5.0F, 0.0F)));

    std::vector<std::complex<float>> exactlyDelay(8);
    EXPECT_EQ(filter.flush(exactlyDelay).size(), 8U);
}
