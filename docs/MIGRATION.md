# Migrating from the pre-0.2 API

Version 0.2 is a rewrite of the public API.

- Everything now lives in namespace `avx_optimized` and is included with `<avx_optimized/...>`
  (or the umbrella header `<avx_optimized/avx_optimized.h>`).
- **Complex functions take `std::complex<float>*` and count complex samples.** The old functions
  took `float*` and counted *floats*, except for `complexToFloat`, which counted complex values.
- **No alignment is required anywhere**, so the separate aligned and unaligned (`UNA`) variants
  have been merged.

The CMake package is now `find_package(avx_optimized)` with the target
`avx_optimized::avx_optimized`. The old names were the project/target `avx_optimized_math`
and `-lavx_optimized_math`.

## Function mapping

Every name in the "New" column is in namespace `avx_optimized`. The table shows the pointer forms,
which are closest to the old calls. Each function also has a span form that drops `count` and
checks sizes (for example `avx_optimized::magnitude(samples, out)` with two vectors), which is
usually the easier choice for new code. `asComplex(p)` is the library's
helper that views an interleaved I/Q `float*` as `std::complex<float>*`
(`avx_optimized::asComplex`).

| Old (`namespace vectorized`) | New (`namespace avx_optimized`) | Behavior change |
|---|---|---|
| `shortToFloat(in, out, n)` | `int16ToFloat(in, out, n)`; for I/Q data, `int16ToComplex(in, samples, n / 2, scale)` | none; both can now also scale in the same pass |
| `floatToShort(in, out, n)` | `floatToInt16(in, out, n)`; for I/Q data, `complexToInt16(samples, out, n / 2, scale)` | now rounds to nearest (was: truncated toward zero), saturates consistently (the old scalar tail wrapped), and maps NaN to 0 |
| `complexToFloat(in, out, nComplex)` | `magnitude(asComplex(in), out, nComplex)` | none; new `magnitudeSquared()` skips the sqrt |
| `reverse(in, out, n)` | `reverse(in, out, n)` | none |
| `multiply(io, num, n)` | `scale(io, num, n)`; complex data: `scale(asComplex(io), num, n / 2)` | none |
| `multiplyIQ(io, i, q, nFloats)` | `scaleIQ(asComplex(io), i, q, nFloats / 2)` | count is now complex samples |
| `vectorMultiply(io, b, nFloats)` | `complexMultiply(asComplex(io), asComplex(b), nFloats / 2)` | count is now complex samples |
| `add(int16* io, second, n)`, `addUNA(...)` | `addSaturated(io, second, n)` | saturates negative overflow too (the old scalar tail wrapped) |
| `add(float* io, second, n)`, `addUNA(...)` | `add(io, second, n)` | none |
| `iqToAmpAndPhase(in, out, nFloats)` | `toPolar(asComplex(in), mag, phase, nFloats / 2)` | **was broken** (wrong output for every sample). Now writes magnitude and phase to two separate arrays; phase is `atan2(Q, I)` in [-π, π] (the old code apparently intended [0, 2π) by adding π, which actually rotated the phase). To reproduce interleaved [0, 2π) output, wrap negative phases by adding 2π and interleave yourself |
| `applyLinearLossAdjustment(data, nFloats, initial, final)` | `applyLinearGainRamp(asComplex(data), initial, final, nFloats / 2)` (note: `count` moved to the end, like every other function) | **was broken**: `initialChange` was ignored and the ramp started at 0. Now the gain goes from `startGain` on the first sample to exactly `endGain` on the last. Gains scale amplitudes: pass the square root of a power ratio |
| `zeroVec()` | removed | use `_mm256_setzero_ps()` |

## Convolution → ComplexFirFilter

The quickest port is the compatibility facade `avx_optimized::Convolution`, which keeps the old
constructor and `convolve()` signatures (interleaved floats, sizes in floats):

```cpp
// Old
#include <avx_optimized_math/Convolution.h>
Convolution conv(kernelFloats, 2 * nTaps);                 // normalized the kernel to sum to 1
conv.convolve(inFloats, outFloats, 2 * nSamples);

// New, facade: same calls
#include <avx_optimized/convolution.h>
avx_optimized::Convolution conv(kernelFloats, 2 * nTaps);  // kernel used as given
conv.convolve(inFloats, outFloats, 2 * nSamples);

// New, native API
avx_optimized::ComplexFirFilter fir(avx_optimized::asComplex(kernelFloats), nTaps);
fir.filter(avx_optimized::asComplex(inFloats), avx_optimized::asComplex(outFloats), nSamples);
```

Both have a `flush()` that writes the outputs still owed at the end of a stream (the old class never
produced them). The facade's accessor for the underlying filter is `firFilter()`.

### Matched filters

If you used `Convolution` as a matched filter by passing it the conjugated, time-reversed
template, `ComplexFirFilter::matchedFilter()` now builds those taps for you from the template
itself:

```cpp
avx_optimized::ComplexFirFilter matched =
    avx_optimized::ComplexFirFilter::matchedFilter(templ, avx_optimized::TemplateScaling::UnitEnergy);
matched.filter(rx, out, n);
// A pulse starting at input sample p peaks at output p + matched.delay(); old code found it at p.
// At the end of the stream, matched.flush(out) writes the last matched.delay() outputs.
```

The scaling argument is required, so the choice is always visible in the code. The README has a
complete chunked-detection recipe, including how to track the stream position across chunks.

`TemplateScaling::UnitEnergy` is recommended for templates learned from observations: the output
scale then doesn't depend on how strong the training observation was, and white noise keeps its
power, so detection thresholds can be set in units of the noise level. The old class divided the
template by its complex sum, which for zero-mean waveforms was close to zero, so **thresholds tuned
on old outputs need re-tuning** whichever scaling you pick.

### Differences to be aware of (facade and native API alike)

- **Taps are no longer normalized.** The old class silently divided the kernel by its complex sum,
  which also removed its phase and divided by about zero for high-pass and band-pass kernels. To
  keep the old behavior, divide the taps by their sum yourself.
- **The output is causal and streams properly.** The old class produced "valid" convolution (only
  outputs with full overlap, written at the start of `out`), left the last `nTaps` outputs garbage,
  and kept no history between calls, so each call started cold even though the header promised
  continuity. Now `out[k]` depends on the current and previous `nTaps - 1` inputs, including those
  from earlier calls. Compared with the old valid-mode output, the new output is shifted later by
  `nTaps - 1` samples (`delay()`): old `out[k]` is new `out[k + delay()]`. Code that converts
  output indices to times or positions, such as matched-filter peak picking, must subtract `delay()`.
- Any number of taps works (the old one read out of bounds unless the kernel was a multiple of 4
  samples), any length and alignment of input works, and `in == out` is allowed.
- The old class also had a heap buffer overflow on every call; that is gone.

## Allocator

| Old (`namespace aligned`) | New (`namespace avx_optimized`) |
|---|---|
| `AlignedAllocator<T, Alignment::AVX>` | `AlignedAllocator<T, 32>` (default is now 64) |
| `alignedVector<T>` | `AlignedVector<T>` (64-byte aligned) |
| `alignedFloatVector`, `alignedComplexVector` | `AlignedVector<float>`, `AlignedVector<std::complex<float>>` |
| `alignedVecVector` (`vector<__m256>`) | removed (`AlignedVector<__m256>` works, but you'll get `-Wignored-attributes`) |
| `allocateAligned<T>(align, n)` / `freeAligned(p)` | removed; use `AlignedVector`, or `std::aligned_alloc` / `free` |
| `ALIGN16_END`, `ALIGN32_END`, `ALIGN64_END` | removed; use standard `alignas(N)` |
