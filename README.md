# avx_optimized

[![CI](https://github.com/wgvisots/avx_optimized/actions/workflows/ci.yml/badge.svg)](https://github.com/wgvisots/avx_optimized/actions/workflows/ci.yml)

Fast, tested building blocks for signal processing on x86-64. It covers I/Q conversions,
element-wise real and complex arithmetic, magnitude, phase and decibels, and streaming FIR and
matched filters. Every function has AVX2 and AVX-512 implementations plus a portable fallback, and
the best one for the running CPU is **chosen at runtime**. One binary runs everywhere and uses
AVX-512 where it exists.

```cpp
#include <avx_optimized/avx_optimized.h>

// int16 I/Q from a radio -> complex samples in [-1, 1) -> low-pass filter -> power in dB
avx_optimized::int16ToComplex(raw, samples, 1.0F / 32768.0F); // vectors, arrays or spans; sizes checked

avx_optimized::ComplexFirFilter lowPass(lowPassTaps); // real or complex taps
lowPass.filter(samples);                              // streaming, in place

avx_optimized::powerToDecibels(samples, powerDb);
```

## Quick reference

Every function takes **spans**: pass `std::vector`, `AlignedVector`, `std::array`, C arrays or
subspans directly, and sizes are checked. Each one also has a **pointer form** for raw buffers,
taking pointers plus a `count` as the last parameter, e.g. `magnitude(in, out, count)`. See
[Conventions](#conventions).

**Conversions**

| Function | Does | Exact¹ |
|---|---|:-:|
| `int16ToFloat(in, out, scale = 1)` | `out = in × scale` | ✓ |
| `floatToInt16(in, out, scale = 1)` | `out = round(in × scale)`, saturating; NaN → 0 | ✓ |
| `int16ToComplex(interleavedIQ, out, scale = 1)` | interleaved int16 I/Q → complex samples | ✓ |
| `complexToInt16(in, interleavedIQ, scale = 1)` | complex samples → interleaved int16 I/Q | ✓ |
| `asComplex(floats)` / `asFloats(samples)` | view interleaved float I/Q (pointers) as complex samples, and back | — |

**Arithmetic**

| Function | Does | Exact¹ |
|---|---|:-:|
| `add(io, other)` | `io += other` (float or complex) | ✓ |
| `addSaturated(io, other)` | int16 add, saturating | ✓ |
| `scale(io, factor)` | `io *= factor` (float or complex) | ✓ |
| `scaleIQ(io, iFactor, qFactor)` | scale I and Q by separate factors | ✓ |
| `complexMultiply(io, other)` | `io *= other` (complex) | |
| `reverse(in, out)` | reverse an array | ✓ |
| `applyLinearGainRamp(io, startGain, endGain)` | gain changing linearly across the array | |

**Magnitude, phase and power**

| Function | Does | Exact¹ |
|---|---|:-:|
| `magnitudeSquared(in, out)` | `|z|²` | |
| `magnitude(in, out)` | `|z|` | |
| `toPolar(in, magnitude, phase)` | `|z|` and `atan2(im, re)` in [-π, π] | |
| `powerToDecibels(in, out)` | `10·log10(p)` for power values, or `10·log10(|z|²)` for complex samples | |

**Filters**

| Class / function | Does |
|---|---|
| `ComplexFirFilter(taps)` | streaming FIR for complex data, with real or complex taps |
| `ComplexFirFilter::matchedFilter(template, scaling)` | streaming matched filter (correlator) built from a template |
| `filter(in, out)` / `filter(io)` | process the next chunk of the stream (in place with one argument) |
| `flush(out)` / `reset()` / `delay()` | end the stream (returns the outputs written) / start over / numTaps − 1 |
| `Convolution(kernel, kernelSize)` | compatibility facade with the pre-0.2 interleaved-float interface |

**Runtime dispatch:** `activeIsa()`, `setActiveIsa(isa)`, `isSupported(isa)`, `bestSupportedIsa()`,
`isaName(isa)` and `versionString()`.

¹ Bit-identical on every instruction set. The others agree to within a few float ulps; the vector
versions use FMA and vectorized `log`/`atan2`.

The full reference is in the headers under [`include/avx_optimized/`](include/avx_optimized/). An
HTML API reference can be generated with Doxygen; see [Building](#building).

## Recipes

**Radio samples in.** Interleaved int16 I/Q → complex samples in [-1, 1), in one pass:

```cpp
std::vector<std::int16_t>                         raw(2 * count); // I, Q, I, Q, ...
avx_optimized::AlignedVector<std::complex<float>> samples(count);
avx_optimized::int16ToComplex(raw, samples, 1.0F / 32768.0F);
```

**Radio samples out.** Complex samples in [-1, 1] → interleaved int16, rounded and saturated:

```cpp
avx_optimized::complexToInt16(samples, raw, 32767.0F); // raw must hold 2 values per sample
```

Scaling by 1/32768 on the way in and 32767 on the way out doesn't round-trip exactly
(32767 → 0.99997 → 32766). Use the same power of two both ways if that matters.

**Low-pass filter** with real taps (from a windowed sinc, `scipy.signal.firwin`, Remez, ...):

```cpp
const std::vector<float>        taps = designLowPass(); // your design
avx_optimized::ComplexFirFilter lowPass(taps);          // real taps: half the work of complex taps
lowPass.filter(samples);                                // in place; call once per chunk, state carries over
```

**Power in dB:**

```cpp
avx_optimized::AlignedVector<float> powerDb(samples.size());
avx_optimized::powerToDecibels(samples, powerDb); // 10·log10(|z|²); 0 → -inf
```

**Matched-filter detection in a stream that arrives in chunks:**

```cpp
avx_optimized::ComplexFirFilter matched =
    avx_optimized::ComplexFirFilter::matchedFilter(templ, avx_optimized::TemplateScaling::UnitEnergy);

std::vector<std::complex<float>> out(maxChunkSize);
std::size_t                      samplesBefore = 0; // stream index of the current chunk's first sample

const auto findPulses = [&](std::span<const std::complex<float>> y) {
    for (std::size_t i = 0; i < y.size(); ++i) {
        const std::size_t streamIndex = samplesBefore + i;
        if (streamIndex >= matched.delay() && std::norm(y[i]) > threshold) { // (a real detector also picks local maxima)
            const std::size_t pulseStart = streamIndex - matched.delay();
            const float       phase      = std::arg(y[i]); // relative to the template's own phase
            // ... report pulseStart and phase
        }
    }
};

while (/* another chunk arrives, as std::span<const std::complex<float>> chunk */) {
    const std::span<std::complex<float>> y = std::span(out).first(chunk.size());
    matched.filter(chunk, y);
    findPulses(y);
    samplesBefore += chunk.size();
}
findPulses(matched.flush(out)); // the last delay() outputs; flush() returns exactly those
```

With `UnitEnergy` scaling, complex white noise of power σ² comes out with power σ², so the
threshold can be set relative to the noise floor (for example, `threshold = 20 * noisePower`, about
13 dB). The peak's phase is the arriving pulse's phase *relative to the template*. A template learned
from an observation carries that observation's phase.

**Interleaved float buffers** (I, Q, I, Q, ... as `float*`, common in older code):

```cpp
avx_optimized::magnitude(avx_optimized::asComplex(iqFloats), magnitudes, sampleCount); // pointer form
```

## Conventions

- **Spans or pointers.** With spans, sizes come from the spans and are checked: arrays that must
  match but don't make the call throw `std::invalid_argument` (naming the function and both sizes)
  before anything is written. With pointers, **`count` comes last** and counts elements of the
  pointed-to type: for complex functions it's the number of complex samples, and the I/Q
  conversions count complex samples too.
- **Pointer forms trust `count`.** They can't know how big your arrays are, so a wrong `count` reads
  or writes past the end. When a size comes from outside your program (a file header, a network
  packet, a message from another process), use the span forms, which check it.
- **Complex data is `std::complex<float>`**, stored as {real, imag}: the same memory as an array of
  interleaved I/Q floats. `asComplex()` and `asFloats()` convert pointers between the two, at no cost.
  The type is only used at the interface. Inside, every kernel works on plain floats.
- **Speed of your own complex loops:** with GCC and Clang, `a * b` on `std::complex<float>` follows
  the C99 Annex G infinity/NaN rules. It compiles to a call to `__mulsc3`, and the loop is not
  vectorized (true of GCC 13 at `-O3 -march=haswell`). Prefer the library's functions
  (`complexMultiply()`, `scale()`, ...), or compile your code with `-fcx-limited-range` if you don't
  need those rules.
- **No alignment is required.** For best performance use 64-byte aligned buffers anyway; this
  matters most for AVX-512. `avx_optimized::AlignedVector<T>` is a `std::vector` that provides it,
  and every function and constructor accepts it.
- **Aliasing:** in-place functions take `io`. Their second input may be the same pointer as `io`,
  but must not partially overlap it. Inputs and outputs of out-of-place functions must not overlap.
- **Errors:** pointer forms never throw. Constructors, `matchedFilter()`, `setActiveIsa()` and span
  forms that take several arrays throw `std::invalid_argument` for invalid arguments. Only filter
  construction allocates; filtering itself never does, so it is safe on real-time threads.
- **Thread safety:** all free functions can be called from any number of threads at once. A filter
  object holds stream state, so each one should be used by one thread at a time.

## Runtime dispatch

| Level | Requires | CPUs |
|---|---|---|
| `avx512` | AVX-512 F + BW + DQ + VL (x86-64-v4) | Intel Skylake-SP and later Xeons, Ice Lake, Sapphire Rapids, AMD Zen 4/5 |
| `avx2` | AVX2 + FMA (x86-64-v3) | Intel Haswell (2013) and later, AMD Zen and later |
| `scalar` | nothing | anything, including non-x86 builds |

The library checks CPU (and OS) support the first time any function is called. To see or change
the choice:

```cpp
avx_optimized::activeIsa();                                // e.g. avx_optimized::Isa::Avx2
avx_optimized::setActiveIsa(avx_optimized::Isa::Avx2);     // e.g. to avoid AVX-512 downclocking
```

Or set the starting point without code changes, which is handy for comparing results or
performance:

```sh
AVX_OPTIMIZED_ISA=avx2 ./your_program     # scalar | avx2 | avx512 (case-insensitive)
```

The variable caps the automatic choice. Asking for more than the CPU supports gives the best
supported level, and a misspelled value prints a warning on stderr. `setActiveIsa()` still
overrides it.

Only the kernel source files are compiled with AVX2/AVX-512 enabled. Everything else, including
the public headers, is built for the baseline x86-64 target. You don't need `-march=native`, and
your own compiler flags don't matter to the library.

## Building

Requirements: CMake ≥ 3.21 (3.25 for the presets), and GCC or Clang with C++20 support (tested
with GCC 13 and Clang 18). The library and projects that use it are C++20; linking to
`avx_optimized::avx_optimized` turns C++20 on for you.
[mathfun](https://github.com/wgvisots/avx_mathfun) (header-only) is used if installed; otherwise it
is downloaded automatically. GoogleTest, needed for the tests, is handled the same way.

```sh
cmake --preset release            # or: debug, clang-release, sanitize, coverage, scalar-only, benchmark
cmake --build --preset release
ctest --preset release
cmake --install build/release --prefix /where/you/want
```

`cmake --list-presets` shows all presets. Without presets, the usual
`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` works too.

| Option | Default | |
|---|---|---|
| `BUILD_SHARED_LIBS` | `OFF` | build a shared library |
| `AVX_OPTIMIZED_BUILD_TESTS` | `ON` when top-level | unit tests |
| `AVX_OPTIMIZED_BUILD_BENCHMARKS` | `OFF` | `bench/avx_optimized_benchmark` |
| `AVX_OPTIMIZED_BUILD_DOCS` | `OFF` | `avx_optimized_docs` target: HTML API reference via Doxygen, in `<build>/docs/html` |
| `AVX_OPTIMIZED_FETCH_DEPENDENCIES` | `ON` | download mathfun/GoogleTest if `find_package` can't find them |
| `AVX_OPTIMIZED_X86_KERNELS` | `ON` on x86-64 | build the AVX2/AVX-512 kernels |
| `AVX_OPTIMIZED_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` |

`include/avx_optimized/export.h` and `version.h` are generated into the build directory, so browsing
the sources you'll only find `version.h.in`.

Packages: `cmake --preset package-deb && cmake --build --preset package-deb && (cd build/package-deb && cpack)`
(likewise `package-rpm`).

## Using it from CMake

Installed:

```cmake
find_package(avx_optimized 0.2 REQUIRED)
target_link_libraries(your_target PRIVATE avx_optimized::avx_optimized)
```

Or vendored, without installing:

```cmake
include(FetchContent)
FetchContent_Declare(avx_optimized GIT_REPOSITORY https://github.com/wgvisots/avx_optimized.git GIT_TAG main)
FetchContent_MakeAvailable(avx_optimized)
target_link_libraries(your_target PRIVATE avx_optimized::avx_optimized)
```

[`examples/`](examples/) contains a complete standalone project.

## ComplexFirFilter

It computes `y[k] = Σ h[m]·x[k-m]`, taking samples before the start of the stream as zero.
- **Streaming:** splitting a stream into chunks of any size gives bit-identical output to
  processing it in one call. `filter()` produces one output per input, and `in == out` is allowed.
- **End of stream:** the last `delay()` = numTaps − 1 outputs still depend on the most recent
  inputs. `flush(out)` writes them, returns the span it wrote, and readies the filter for a new
  stream.
- **Taps:** used exactly as given, never normalized. Real taps, or complex taps whose imaginary
  parts are all zero, use a kernel with half the arithmetic.
- **When to use it:** direct convolution costs O(taps) per sample, so for long filters on large
  blocks, FFT-based overlap-save/overlap-add needs far fewer operations. Direct form still wins for
  short and moderate filters, small or latency-critical blocks, and streams where per-sample latency
  matters.

## Performance

These are `avx_optimized_benchmark` results on an Intel i7-1260P (Alder Lake P-core, AVX2; this
CPU has no AVX-512) with 4096 elements per call, in nanoseconds per element:

| Function | scalar | AVX2 | speedup |
|---|---:|---:|---:|
| `floatToInt16` | 1.09 | 0.073 | 15× |
| `addSaturated` | 0.288 | 0.020 | 14× |
| `complexMultiply` | 0.376 | 0.139 | 2.7× |
| `toPolar` | 15.3 | 0.946 | 16× |
| `powerToDecibels` (complex) | 4.96 | 0.808 | 6.1× |
| `applyLinearGainRamp` | 0.597 | 0.128 | 4.7× |
| `ComplexFirFilter`, 64 complex taps | 42.4 | 4.84 | 8.8× |
| `ComplexFirFilter`, 64 real taps | 22.9 | 2.30 | 9.9× |
| `ComplexFirFilter`, 256 complex taps | 170 | 18.8 | 9.0× |
| `ComplexFirFilter`, 256 real taps | 95.8 | 8.92 | 10.7× |

Simple streaming operations such as `add`, `scale` and `int16ToFloat` gain less (1.5–3×). They are
limited by memory bandwidth, and the compiler already auto-vectorizes the scalar versions with SSE.
To measure your own machine: `cmake --preset benchmark && cmake --build --preset benchmark && build/benchmark/bench/avx_optimized_benchmark [count]`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the source layout, how to add a function, the rules for
the instruction-set-specific files, formatting, and testing AVX-512 without AVX-512 hardware.
Porting code from the pre-0.2 API? See [docs/MIGRATION.md](docs/MIGRATION.md).

## License

Apache License 2.0; see [LICENSE](LICENSE). The optional
[mathfun](https://github.com/wgvisots/avx_mathfun) dependency is under the zlib license.
