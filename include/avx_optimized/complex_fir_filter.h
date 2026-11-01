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
#include <span>
#include <vector>

#include "avx_optimized/aligned.h"
#include "avx_optimized/export.h"

namespace avx_optimized {

/// How ComplexFirFilter::matchedFilter() scales the template.
enum class TemplateScaling {
    AsGiven,    ///< Use the template exactly as given; the peak is (pulse amplitude) x (template energy).
    UnitEnergy, ///< Scale the template to unit energy; recommended for templates learned from observations.
};

/**
 * A streaming FIR filter (direct-form convolution) for complex data such as I/Q samples, with
 * real or complex taps.
 *
 * For input x and taps h[0..N-1], each output sample is
 *
 *     y[k] = sum_{m=0}^{N-1} h[m] * x[k - m]
 *
 * Samples before the first input are taken as zero.  The filter keeps the last N-1 input samples
 * between calls, so feeding a stream in pieces of any size gives exactly the same output as
 * feeding it all at once.  Every call produces one output sample per input sample; the outputs
 * that depend on the last N-1 inputs of a stream come out of flush().
 *
 * Real taps (the usual case for low-pass, high-pass and band-pass filters) take half the
 * arithmetic of complex taps.  Complex taps whose imaginary parts are all zero get the same
 * speedup automatically.
 *
 * The taps are used exactly as given (they are not normalized).  Any number of taps >= 1 works,
 * and filter() and flush() never allocate, so they are safe to call from real-time threads.
 *
 * Direct convolution costs O(numTaps) per sample.  It is usually the better choice for short and
 * moderate filters, small or latency-sensitive blocks, and streams where per-sample latency
 * matters.  For long filters on large blocks, overlap-save/overlap-add FFT convolution needs far
 * fewer operations; where the crossover lies depends on the machine and block size, so measure
 * (bench/ times this filter at several lengths).
 *
 * Thread safety: one filter object must not be used by several threads at once (it has stream
 * state).  Separate objects are independent.
 */
class AVX_OPTIMIZED_EXPORT ComplexFirFilter {
public:
    /**
     * Filter with complex taps.
     *
     * @param taps    filter coefficients h[0..numTaps-1], in convolution order (taps[0] multiplies
     *                the newest sample)
     * @param numTaps number of taps; must be at least 1
     * @throws std::invalid_argument if numTaps is 0 or taps is null
     */
    ComplexFirFilter(const std::complex<float>* taps, std::size_t numTaps);

    /**
     * Filter with real taps, applied to the real and imaginary parts alike.
     *
     * @param taps    filter coefficients h[0..numTaps-1], in convolution order
     * @param numTaps number of taps; must be at least 1
     * @throws std::invalid_argument if numTaps is 0 or taps is null
     */
    ComplexFirFilter(const float* taps, std::size_t numTaps);

    /// Filter with complex taps from any contiguous container (std::vector, AlignedVector, std::array, ...).
    explicit ComplexFirFilter(std::span<const std::complex<float>> taps);

    /// Filter with real taps from any contiguous container (std::vector, AlignedVector, std::array, ...).
    explicit ComplexFirFilter(std::span<const float> taps);

    /**
     * Create a matched filter (streaming cross-correlator) for the waveform `templ`.
     *
     * The taps are the time-reversed complex conjugate of the template, h[m] = conj(s[N-1-m]), so
     * each output is the correlation of the template with the newest N input samples:
     *
     *     y[k] = sum_{n=0}^{N-1} conj(s[n]) * x[k - (N-1) + n]
     *
     * If the input contains the template scaled by a complex amplitude `a` and starting at input
     * sample p, then y[p + delay()] = a * E, where E = sum |s[n]|^2 is the template energy.  Without
     * noise or other signals, that is the output with the largest magnitude.  Its phase, arg(a), is
     * the phase of the pulse *relative to the template*; a template learned from an observation
     * carries that observation's phase, so phases measured with it are relative to that observation.
     *
     * With TemplateScaling::UnitEnergy the template is first divided by sqrt(E), so the peak
     * becomes a * sqrt(E).  For a template learned from observations, that keeps the output scale
     * (and so detection thresholds) independent of the amplitude of the observation the template
     * came from, and white noise of variance sigma^2 comes out with variance sigma^2.
     *
     * In a stream processed in chunks, keep a running count of samples already filtered: a peak at
     * chunk output index i is at stream index (samplesBefore + i), and the pulse started at stream
     * index (samplesBefore + i - delay()).  Call flush() at the end of the stream so pulses in the
     * last delay() samples are not missed.
     *
     * @throws std::invalid_argument if the template is empty, or if UnitEnergy is requested for a
     *         template whose energy is zero or not finite
     */
    [[nodiscard]] static ComplexFirFilter
    matchedFilter(const std::complex<float>* templ, std::size_t length, TemplateScaling scaling);

    /// matchedFilter() for a template in any contiguous container (std::vector, AlignedVector, ...).
    [[nodiscard]] static ComplexFirFilter matchedFilter(std::span<const std::complex<float>> templ,
                                                        TemplateScaling                      scaling);

    /**
     * Filter the next `count` samples of the stream.
     *
     * `in` and `out` may be the same pointer (in-place filtering) but must not otherwise overlap.
     */
    void filter(const std::complex<float>* in, std::complex<float>* out, std::size_t count) noexcept;

    /**
     * filter() over spans: filters all of `in` into `out`.
     *
     * @throws std::invalid_argument if `out` is not the same size as `in` (nothing is filtered)
     */
    void filter(std::span<const std::complex<float>> in, std::span<std::complex<float>> out);

    /// Filter `io` in place.
    void filter(std::span<std::complex<float>> io) noexcept;

    /**
     * End the stream: write the last delay() outputs, the ones that still depend on the most recent
     * inputs, to `out` (which must have room for delay() samples).
     *
     * Equivalent to filtering delay() zeros.  Afterwards the filter is ready for a new stream, as
     * after reset().
     */
    void flush(std::complex<float>* out) noexcept;

    /**
     * flush() into a span, which must have room for at least delay() samples.  Returns the part of
     * `out` that was written (its first delay() samples), so the result can be processed directly:
     *
     *     findPulses(filter.flush(buffer));
     *
     * @throws std::invalid_argument if `out` is shorter than delay() (nothing is written)
     */
    std::span<std::complex<float>> flush(std::span<std::complex<float>> out);

    /// Forget the stream history, as if the filter had just been constructed.
    void reset() noexcept;

    /// Number of taps (at least 1).
    [[nodiscard]] std::size_t numTaps() const noexcept { return taps_.size(); }

    /**
     * numTaps() - 1: how many samples later than its first input sample a full window shows up in
     * the output.  y[k] is the first output that covers the whole window x[k - delay() .. k].  For a
     * matched filter, a pulse starting at input sample p peaks at output p + delay().  Also the
     * number of outputs flush() writes.
     */
    [[nodiscard]] std::size_t delay() const noexcept { return taps_.size() - 1; }

    /// The taps, in the order they were given to the constructor (real taps have zero imaginary parts).
    [[nodiscard]] const std::vector<std::complex<float>>& taps() const noexcept { return taps_; }

    /// True if all taps are real, in which case the faster real-tap kernel is used.
    [[nodiscard]] bool hasRealTaps() const noexcept { return realTaps_; }

private:
    void run(const std::complex<float>* in, std::complex<float>* out, std::size_t count) noexcept;

    std::vector<std::complex<float>> taps_;
    bool                             realTaps_;

    // Real and imaginary parts of the taps, in reverse order, which is the layout the kernels want.
    // reversedTapsImag_ is empty for real taps.
    AlignedVector<float> reversedTapsReal_;
    AlignedVector<float> reversedTapsImag_;

    // [numTaps-1 samples of history | up to one block of new samples].  Allocated once, in the
    // constructor.
    AlignedVector<std::complex<float>> buffer_;
};

} // namespace avx_optimized
