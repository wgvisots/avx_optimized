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

// Tests for the pre-0.2 compatibility facade (interleaved floats, sizes in floats).

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "avx_optimized/convolution.h"
#include "test_helpers.h"

namespace {

std::vector<float> interleave(const std::vector<std::complex<float>>& samples) {
    std::vector<float> floats(2 * samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        floats[2 * i]     = samples[i].real();
        floats[2 * i + 1] = samples[i].imag();
    }
    return floats;
}

class ConvolutionFacade : public test_helpers::IsaTest {};

} // namespace

TEST(ConvolutionFacadeConstruction, RejectsBadKernels) {
    const std::vector<float> kernel(7, 1.0F);
    EXPECT_THROW(avx_optimized::Convolution(kernel.data(), 7), std::invalid_argument); // odd number of floats
    EXPECT_THROW(avx_optimized::Convolution(kernel.data(), 0), std::invalid_argument);
    EXPECT_THROW(avx_optimized::Convolution(nullptr, 4), std::invalid_argument);

    const avx_optimized::Convolution ok(kernel.data(), 6);
    EXPECT_EQ(ok.firFilter().numTaps(), 3U);
    EXPECT_EQ(ok.delay(), 2U);
}

TEST_P(ConvolutionFacade, IsTheSameAsComplexFirFilter) {
    const std::vector<std::complex<float>> taps         = test_helpers::randomComplex(19);
    const std::vector<std::complex<float>> x            = test_helpers::randomComplex(2500);
    const std::vector<float>               kernelFloats = interleave(taps);
    const std::vector<float>               xFloats      = interleave(x);

    avx_optimized::ComplexFirFilter  fir(taps);
    std::vector<std::complex<float>> expected(x.size());
    fir.filter(x.data(), expected.data(), x.size());

    // In two calls of different sizes, as streaming code would.
    avx_optimized::Convolution conv(kernelFloats.data(), kernelFloats.size());
    std::vector<float>         out(xFloats.size());
    conv.convolve(xFloats.data(), out.data(), 2 * 999);
    conv.convolve(xFloats.data() + 2 * 999, out.data() + 2 * 999, xFloats.size() - 2 * 999);

    for (std::size_t k = 0; k < x.size(); ++k) {
        ASSERT_EQ(out[2 * k], expected[k].real()) << "k=" << k;
        ASSERT_EQ(out[2 * k + 1], expected[k].imag()) << "k=" << k;
    }
}

TEST_P(ConvolutionFacade, OldValidModeOutputAppearsDelaySamplesLater) {
    // The old class computed old[k] = sum_m h[m] * x[k + N-1 - m] ("valid" convolution).  The
    // facade's out[k + delay()] must equal that, which is what MIGRATION.md tells porters.
    const std::vector<std::complex<float>> taps         = test_helpers::randomComplex(12);
    const std::vector<std::complex<float>> x            = test_helpers::randomComplex(400);
    const std::size_t                      numTaps      = taps.size();
    const std::vector<float>               kernelFloats = interleave(taps);
    std::vector<float>                     data         = interleave(x);

    avx_optimized::Convolution conv(kernelFloats.data(), kernelFloats.size());
    conv.convolve(data.data(), data.data(), data.size()); // in place, as allowed

    for (std::size_t k = 0; k + numTaps <= x.size(); ++k) {
        std::complex<double> old{};
        double               bound = 0.0;
        for (std::size_t m = 0; m < numTaps; ++m) {
            const std::complex<double> term =
                std::complex<double>(taps[m]) * std::complex<double>(x[k + numTaps - 1 - m]);

            old   += term;
            bound += std::abs(term);
        }
        const std::size_t          j = k + conv.delay();
        const std::complex<double> actual(data[2 * j], data[2 * j + 1]);
        ASSERT_LE(std::abs(actual - old), 2e-6 * bound) << "k=" << k;
    }
}

TEST_P(ConvolutionFacade, FlushWritesTheLastOutputs) {
    const std::vector<std::complex<float>> taps = test_helpers::randomComplex(6);
    const std::vector<std::complex<float>> x    = test_helpers::randomComplex(50);

    avx_optimized::ComplexFirFilter  fir(taps);
    std::vector<std::complex<float>> expected(x.size() + fir.delay());
    fir.filter(x.data(), expected.data(), x.size());
    fir.flush(expected.data() + x.size());

    const std::vector<float>   kernelFloats = interleave(taps);
    const std::vector<float>   xFloats      = interleave(x);
    avx_optimized::Convolution conv(kernelFloats.data(), kernelFloats.size());
    std::vector<float>         out(2 * expected.size());
    conv.convolve(xFloats.data(), out.data(), xFloats.size());
    conv.flush(out.data() + xFloats.size());

    EXPECT_EQ(out, interleave(expected));
}

TEST_P(ConvolutionFacade, OddSizeLeavesTrailingFloatAlone) {
    const std::vector<float>   kernel = {1.0F, 0.0F};
    avx_optimized::Convolution conv(kernel.data(), kernel.size());
    const std::vector<float>   in = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    std::vector<float>         out(5, -9.0F);

    conv.convolve(in.data(), out.data(), 5);

    EXPECT_EQ(out, (std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F, -9.0F}));
}

TEST_P(ConvolutionFacade, ResetForgetsHistory) {
    const std::vector<float>   kernelFloats = interleave(test_helpers::randomComplex(9));
    const std::vector<float>   xFloats      = interleave(test_helpers::randomComplex(100));
    avx_optimized::Convolution conv(kernelFloats.data(), kernelFloats.size());
    std::vector<float>         first(xFloats.size());
    std::vector<float>         again(xFloats.size());

    conv.convolve(xFloats.data(), first.data(), xFloats.size());
    conv.reset();
    conv.convolve(xFloats.data(), again.data(), xFloats.size());

    EXPECT_EQ(first, again);
}

INSTANTIATE_TEST_SUITE_P(AllIsas,
                         ConvolutionFacade,
                         ::testing::ValuesIn(test_helpers::supportedIsas()),
                         test_helpers::isaTestName);
