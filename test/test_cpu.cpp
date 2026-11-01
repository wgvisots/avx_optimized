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

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"

#include "avx_optimized/cpu.h"
#include "avx_optimized/version.h"
#include "test_helpers.h"

namespace {

// Captured during static initialization, before any test can call setActiveIsa().
const avx_optimized::Isa initialIsa = avx_optimized::activeIsa();

} // namespace

TEST(Cpu, ReportsWhatWasDetected) {
    // Not an assertion: makes CI logs show which implementations were exercised.
    std::cout << "[ INFO     ] best supported ISA: " << avx_optimized::isaName(avx_optimized::bestSupportedIsa())
              << ", initial ISA: " << avx_optimized::isaName(initialIsa) << ", library version "
              << AVX_OPTIMIZED_VERSION_STRING << std::endl;
    SUCCEED();
}

TEST(Cpu, ScalarIsAlwaysSupported) {
    EXPECT_TRUE(avx_optimized::isSupported(avx_optimized::Isa::Scalar));
}

TEST(Cpu, SupportIsMonotonic) {
    if (avx_optimized::isSupported(avx_optimized::Isa::Avx512)) {
        EXPECT_TRUE(avx_optimized::isSupported(avx_optimized::Isa::Avx2));
    }
    EXPECT_TRUE(avx_optimized::isSupported(avx_optimized::bestSupportedIsa()));
}

TEST(Cpu, InitialIsaHonorsEnvironmentCap) {
    // CTest also runs this with AVX_OPTIMIZED_ISA set to various values (see test/CMakeLists.txt).
    avx_optimized::Isa expected = avx_optimized::bestSupportedIsa();
    if (const char* env = std::getenv("AVX_OPTIMIZED_ISA")) {
        std::string requested = env;
        for (char& c : requested) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        for (const avx_optimized::Isa cap : {avx_optimized::Isa::Scalar,
                                             avx_optimized::Isa::Avx2,
                                             avx_optimized::Isa::Avx512}) { // unknown values impose no cap
            if (requested == avx_optimized::isaName(cap) && static_cast<int>(cap) < static_cast<int>(expected)) {
                expected = cap;
            }
        }
    }
    EXPECT_EQ(initialIsa, expected);
}

TEST(Cpu, SetActiveIsaSwitchesAndRejectsUnsupported) {
    const avx_optimized::Isa before = avx_optimized::activeIsa();
    for (const avx_optimized::Isa isa :
         {avx_optimized::Isa::Scalar, avx_optimized::Isa::Avx2, avx_optimized::Isa::Avx512}) {
        if (avx_optimized::isSupported(isa)) {
            avx_optimized::setActiveIsa(isa);
            EXPECT_EQ(avx_optimized::activeIsa(), isa);
        } else {
            EXPECT_THROW(avx_optimized::setActiveIsa(isa), std::invalid_argument);
        }
    }
    avx_optimized::setActiveIsa(before);
}

TEST(Cpu, RuntimeVersionMatchesHeaders) {
    EXPECT_STREQ(avx_optimized::versionString(), AVX_OPTIMIZED_VERSION_STRING);
}

TEST(Cpu, IsaNames) {
    EXPECT_STREQ(avx_optimized::isaName(avx_optimized::Isa::Scalar), "scalar");
    EXPECT_STREQ(avx_optimized::isaName(avx_optimized::Isa::Avx2), "avx2");
    EXPECT_STREQ(avx_optimized::isaName(avx_optimized::Isa::Avx512), "avx512");
    EXPECT_STREQ(avx_optimized::isaName(static_cast<avx_optimized::Isa>(42)), "unknown");
}
