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

#include <cstdint>

#include "Aligned.h"

/**
 * This is a set of NON-PORTABLE X64 vectorized math/conversion functions.  Don't try to use this
 * on old X64 (pre-AVX2), ARM, Power, or other ISA hardware.
 */
namespace vectorized {
/**
 * Convert an array of std::int16_ts to an array of floats
 *
 * @param in - a pointer to an array of std::uint16_t of length 'length' to be read from
 * @param out - a pointer to an array of float of length 'length' to be written to
 * @param length - the length of in and out (number of floats)
 */
void shortToFloat(const std::int16_t* __restrict__ in, float* __restrict__ out, std::size_t length);

/**
 * Convert an array of floats to an array of std::uint16_t
 *
 * The values of 'in' should be in the valid range of a short, and are rounded/saturated otherwise.
 *
 * @param in - a pointer to an array of float of length 'length' to be read from
 * @param out - a pointer to an array of std::uint16_t of length 'length' to be written to
 * @param length - the length of in and out (number of floats)
 */
void floatToShort(const float* __restrict__ in, std::int16_t* __restrict__ out, std::size_t length);

/**
 * Rewrite the array in the opposite order
 * NOTE: Assumes aligned memory!
 * @param in - an array
 * @param out - will be filled with in, only backwards
 * @param size - the length of in and out
 */
void reverse(const float* __restrict__ in, float* __restrict__ out, std::size_t size);

/**
 * Multiply every value in 'io' by 'num', and save back into 'io'
 * NOTE: Assumes aligned memory!
 * @param io
 * @param num
 * @param size - the length of in and out (number of floats)
 */
void multiply(float* __restrict__ io, float num, std::size_t size);

/**
 * This assumes that 'io' is an array containing IQ data, arranged as I,Q,I,Q,...I,Q
 * It will multiply every I by 'ival', and every Q by 'qval', and write back to 'io'.
 * NOTE: Assumes aligned memory!
 * @param io
 * @param ival
 * @param qval
 * @param size
 */
void multiplyIQ(float* __restrict__ io, float ival, float qval, std::size_t size);

/**
 * @brief in-place multiply 'io' with 'b'.
 * Multiply every complex number in io with the corresponding complex number in b, and store the result back to io.
 *
 * NOTE: Assumes aligned memory!
 * @param io
 * @param b
 * @param size - number of numbers in io and b (number of complex numbers in io * 2)
 */
void vectorMultiply(float* __restrict__ io, const float* __restrict__ b, std::size_t size);

/**
 * Add the elements, element by element, of the two arrays ('io' and 'secon'), and write back to 'io'.
 * NOTE: Assumes aligned memory!
 * @param io
 * @param second
 * @param size
 */
void add(std::int16_t* __restrict__ io, const std::int16_t* __restrict__ second, std::size_t size); /*32 byte aligned*/

/**
 * Add the elements, element by element, of the two arrays ('io' and 'secon'), and write back to 'io'.
 * NOTE: Assumes un-aligned memory!
 * @param io
 * @param second
 * @param size
 */
void addUNA(std::int16_t* __restrict__ io, const std::int16_t* __restrict__ second, std::size_t size);

/*32 byte un-aligned*/

/**
 * Add the elements, element by element, of the two arrays ('io' and 'secon'), and write back to 'io'.
 * NOTE: Assumes aligned memory!
 * @param io
 * @param second
 * @param size
 */
void add(float* __restrict__ io, const float* __restrict__ second, std::size_t size); /*32 byte aligned*/

/**
 * Add the elements, element by element, of the two arrays ('io' and 'secon'), and write back to 'io'.
 * NOTE: Assumes un-aligned memory!
 * @param io
 * @param second
 * @param size
 */
void addUNA(float* __restrict__ io, const float* __restrict__ second, std::size_t size); /*32 byte un-aligned*/

/**
 * Take in an array of std::complex<float> (as a float*), and output an array of amplitude/phase.
 *
 * @param in I,Q,I,Q...
 * @param out Amp,Phase,Amp,Phase...
 * @param size the number of floats in 'in' (2X number of samples)
 */
void iqToAmpAndPhase(const float* __restrict__ in, float* __restrict__ out, std::size_t size);

/**
 * Simple helper to give you a constant __m256 (array of 8 floats) that are all set to 0.0F.
 * @return
 */
const __m256& zeroVec();

/**
 * The cannonical use case for this function is applying free space path loss estimates.  The 'data' is your IQ data,
 * 'initialChange' is how much loss is estimated at the start of the data, and 'finalChange' is the loss estimate at
 * the end of the data.
 *
 * implemented as:
 * data[n] = data[n] * (((finalChange-initialChange)/(size/2)) * n)
 *
 * @param data std::complex<float>, flattened to an array of floats [I,Q,I,Q,...,I,Q]
 * @param size number of numbers (2X number of samples)
 * @param initialChange loss at the beginning of the data (expressed as a ratio)
 * @param finalChange loss at the end of the data (expressed as a ratio)
 */
void applyLinearLossAdjustment(float* __restrict__ data, std::size_t size, const float& initialChange, const float& finalChange);
}
