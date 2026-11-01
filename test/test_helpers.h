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

#include <complex>
#include <cstddef>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "avx_optimized/aligned.h"
#include "avx_optimized/cpu.h"

namespace avx_optimized {
/// Lets GoogleTest print Isa values by name.
inline void PrintTo(Isa isa, std::ostream* os) {
    *os << isaName(isa);
}
} // namespace avx_optimized

namespace test_helpers {

/// Every instruction set that can run on this machine.
inline std::vector<avx_optimized::Isa> supportedIsas() {
    std::vector<avx_optimized::Isa> result;
    for (auto isa : {avx_optimized::Isa::Scalar, avx_optimized::Isa::Avx2, avx_optimized::Isa::Avx512}) {
        if (avx_optimized::isSupported(isa)) {
            result.push_back(isa);
        }
    }
    return result;
}

inline std::string isaTestName(const testing::TestParamInfo<avx_optimized::Isa>& info) {
    return avx_optimized::isaName(info.param);
}

/// Activates an instruction set for the lifetime of the object.
class ScopedIsa {
public:
    explicit ScopedIsa(avx_optimized::Isa isa)
        : previous_(avx_optimized::activeIsa()) {
        avx_optimized::setActiveIsa(isa);
    }
    ~ScopedIsa() { avx_optimized::setActiveIsa(previous_); }
    ScopedIsa(const ScopedIsa&)            = delete;
    ScopedIsa& operator=(const ScopedIsa&) = delete;

private:
    avx_optimized::Isa previous_;
};

/// Fixture base: runs each test with the parameter's instruction set active.
class IsaTest : public testing::TestWithParam<avx_optimized::Isa> {
protected:
    void SetUp() override { isa_ = std::make_unique<ScopedIsa>(GetParam()); }
    void TearDown() override { isa_.reset(); }

private:
    std::unique_ptr<ScopedIsa> isa_;
};

/// Sizes that cover empty input, every tail length for every vector width, and multi-block runs.
inline std::vector<std::size_t> testSizes() {
    std::vector<std::size_t> sizes;
    for (std::size_t n = 0; n <= 70; ++n) {
        sizes.push_back(n);
    }
    for (std::size_t n : {127, 128, 129, 255, 256, 257, 1000, 1023, 1024, 1025, 4099}) {
        sizes.push_back(n);
    }
    return sizes;
}

/// Element offsets from a 64-byte boundary; nonzero offsets exercise unaligned access.
inline constexpr std::size_t kOffsets[] = {0, 1, 3};

/**
 * A buffer of `count` elements placed `offset` elements past a 64-byte boundary, surrounded by
 * guard zones filled with a sentinel byte pattern.  intact() reports whether anything wrote
 * outside the buffer.
 */
template <typename T>
class GuardedBuffer {
public:
    static constexpr std::size_t   kGuardBytes = 256;
    static constexpr unsigned char kSentinel   = 0xA5;

    GuardedBuffer(std::size_t count, std::size_t offset)
        : count_(count)
        , offsetBytes_(kGuardBytes + offset * sizeof(T))
        , storage_(offsetBytes_ + count * sizeof(T) + kGuardBytes) {
        std::memset(storage_.data(), kSentinel, storage_.size());
    }

    T*          data() { return reinterpret_cast<T*>(storage_.data() + offsetBytes_); }
    T&          operator[](std::size_t i) { return data()[i]; }
    std::size_t size() const { return count_; }

    void fill(const std::vector<T>& values) {
        if (count_ > 0) {
            std::memcpy(data(), values.data(), count_ * sizeof(T));
        }
    }
    std::vector<T> values() { return std::vector<T>(data(), data() + count_); }

    bool intact() const {
        const std::size_t end = offsetBytes_ + count_ * sizeof(T);
        for (std::size_t i = 0; i < storage_.size(); ++i) {
            if ((i < offsetBytes_ || i >= end) && storage_[i] != kSentinel) {
                return false;
            }
        }
        return true;
    }

private:
    std::size_t                                 count_;
    std::size_t                                 offsetBytes_;
    avx_optimized::AlignedVector<unsigned char> storage_;
};

inline std::mt19937& rng() {
    static std::mt19937 engine(12345);
    return engine;
}

inline std::vector<float> randomFloats(std::size_t count, float lo = -1.0F, float hi = 1.0F) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float>                    v(count);
    for (auto& x : v) {
        x = dist(rng());
    }
    return v;
}

inline std::vector<std::complex<float>> randomComplex(std::size_t count, float lo = -1.0F, float hi = 1.0F) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<std::complex<float>>      v(count);
    for (auto& x : v) {
        x = {dist(rng()), dist(rng())};
    }
    return v;
}

inline std::vector<std::int16_t> randomInt16(std::size_t count) {
    std::uniform_int_distribution<int> dist(-32768, 32767);
    std::vector<std::int16_t>          v(count);
    for (auto& x : v) {
        x = static_cast<std::int16_t>(dist(rng()));
    }
    return v;
}

} // namespace test_helpers
