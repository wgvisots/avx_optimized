# Changelog

## 0.2.0 (unreleased)

A rewrite in preparation for the first public release. See [docs/MIGRATION.md](docs/MIGRATION.md)
for how to port code from the old API.

### Added
- **Instruction sets and dispatch**
  - AVX-512 (F/BW/DQ/VL) implementations of every function.
  - Runtime dispatch: every function picks the best of scalar / AVX2 / AVX-512 for the running CPU.
    Includes the `cpu.h` API and the `AVX_OPTIMIZED_ISA` environment variable, which warns about
    unrecognised values.
  - Portable scalar fallback, so the library runs on any x86-64 CPU and builds on non-x86.
- **Filters**
  - `ComplexFirFilter`: a correct, streaming, allocation-free FIR filter for complex data, with any
    number of real or complex taps. Real taps use a kernel with half the arithmetic. Includes
    `delay()` and `flush()`. Replaces `Convolution`.
  - `ComplexFirFilter::matchedFilter()`, which builds a matched filter from a template, with
    explicit `TemplateScaling` (as given, or unit energy).
  - `Convolution` compatibility facade, keeping the old interleaved-float interface on top of
    `ComplexFirFilter`.
- **New functions**
  - I/Q conversions that count samples and scale in the same pass: `int16ToComplex()` and
    `complexToInt16()`. Also a `scale` parameter on `int16ToFloat()` and `floatToInt16()`.
  - Complex overloads of `add()` and `scale()`.
  - `asComplex()` / `asFloats()` for interleaved I/Q float buffers.
  - `powerToDecibels()` for power values and for complex samples.
  - `magnitudeSquared()`.
  - `std::span` forms of every function and of the filter constructors, `filter()` and `flush()`.
    They take sizes from the spans and check them.
  - `versionString()`, which reports the version of the library actually loaded.
- **Build, packaging and docs**
  - CMake package config (`find_package(avx_optimized)`), install rules, a version header, and
    symbol visibility (the shared library exports only the public API).
  - Presets, Debian/RPM packaging, and an optional Doxygen API reference.
  - `.clang-format` (a table-style layout), CONTRIBUTING.md, an example project and a benchmark
    program.
- **Tests and CI**
  - Every function is compared with a double-precision reference on every instruction set, for all
    tail lengths, with misaligned buffers and out-of-bounds write detection.
  - CI with sanitizers, emulated CPUs (Intel SDE), packaging, formatting and docs checks.
- **Security**
  - SECURITY.md, describing how to report a vulnerability privately.
  - CI actions are pinned to commit hashes and kept up to date by Dependabot, and the Intel SDE
    archive's checksum is verified on every run.
  - An unrecognised `AVX_OPTIMIZED_ISA` value is shown in the warning with non-printable characters
    replaced and long values cut short, so it can't send escape sequences to the terminal.

### Fixed
- Heap buffer overflow in `Convolution::convolve` on every call.
- Crashes (out-of-bounds access) in most functions when the input was shorter than one vector block.
- `iqToAmpAndPhase` returned wrong values for every sample.
- `applyLinearLossAdjustment` ignored `initialChange`.
- The scalar tails of int16 `add` and `floatToShort` wrapped instead of saturating, disagreeing
  with the vector path.
- `Convolution` read out of bounds for kernels that were not a multiple of 4 samples, corrupted
  the last valid output, and did not carry state between calls.
- A data race in the lazily initialized constants (`zeroVec`).
- Loading the library on a CPU without AVX2 crashed immediately, because global constants were
  initialized with AVX instructions.

### Changed
- New names and namespace (`avx_optimized`). Complex functions take `std::complex<float>*`,
  `count` is always the last parameter, and no function requires aligned memory.
- Licensed under Apache-2.0 throughout.
