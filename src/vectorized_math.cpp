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

#include <cassert>
#include <mutex>

#include <immintrin.h>

#include "mathfun/avx_mathfun.h"

#include "Aligned.h"
#include "vectorized_math.h"

namespace{
    constexpr std::uint16_t uFFFF{0xFFFF};
    constexpr std::int16_t  iFFFF{static_cast<std::int16_t>(uFFFF)};
    const __m256i magic1      ALIGN32_END = _mm256_setr_epi8 (2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 22, 23, 20, 21, 26, 27, 24, 25, 30, 31, 28, 29);
    const __m256i magic2      ALIGN32_END = _mm256_setr_epi8 (0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 16, 17, 24, 25, 18, 19, 26, 27, 20, 21, 28, 29, 22, 23, 30, 31);
    const __m256i magic3      ALIGN32_END = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i magic4      ALIGN32_END = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    const __m256i negateAdd   ALIGN32_END = _mm256_setr_epi16(0x0, 0x0001, 0x0, 0x0001, 0x0, 0x0001, 0x0, 0x0001, 0x0, 0x0001, 0x0, 0x0001, 0x0, 0x0001, 0x0, 0x0001);
    const __m256i negateMask  ALIGN32_END = _mm256_setr_epi16(0x0, iFFFF, 0x0, iFFFF, 0x0, iFFFF, 0x0, iFFFF, 0x0, iFFFF, 0x0, iFFFF, 0x0, iFFFF, 0x0, iFFFF);
    const __m256  negateFloat ALIGN32_END = _mm256_setr_ps   (1, -1, 1, -1, 1, -1, 1, -1);

    inline std::size_t doVectorizedMultiply(float* __restrict__ io, const __m256& mul , const std::size_t size){
        assert(size >= 64);
        //    __m256 mul ALIGN32_END;
        __m256 f0  ALIGN32_END;
        __m256 f1  ALIGN32_END;
        __m256 f2  ALIGN32_END;
        __m256 f3  ALIGN32_END;
        __m256 f4  ALIGN32_END;
        __m256 f5  ALIGN32_END;
        __m256 f6  ALIGN32_END;
        __m256 f7  ALIGN32_END;

        const std::size_t stop = size - 64;
        std::size_t i = 0;
        for (; i < stop; i += 64)
        {
            f0 = _mm256_load_ps(io + i + 0);
            f1 = _mm256_load_ps(io + i + 8);
            f2 = _mm256_load_ps(io + i + 16);
            f3 = _mm256_load_ps(io + i + 24);
            f4 = _mm256_load_ps(io + i + 32);
            f5 = _mm256_load_ps(io + i + 40);
            f6 = _mm256_load_ps(io + i + 48);
            f7 = _mm256_load_ps(io + i + 56);

            f0 = _mm256_mul_ps(f0, mul);
            f1 = _mm256_mul_ps(f1, mul);
            f2 = _mm256_mul_ps(f2, mul);
            f3 = _mm256_mul_ps(f3, mul);
            f4 = _mm256_mul_ps(f4, mul);
            f5 = _mm256_mul_ps(f5, mul);
            f6 = _mm256_mul_ps(f6, mul);
            f7 = _mm256_mul_ps(f7, mul);

            _mm256_store_ps(io + i + 0, f0);
            _mm256_store_ps(io + i + 8, f1);
            _mm256_store_ps(io + i + 16, f2);
            _mm256_store_ps(io + i + 24, f3);
            _mm256_store_ps(io + i + 32, f4);
            _mm256_store_ps(io + i + 40, f5);
            _mm256_store_ps(io + i + 48, f6);
            _mm256_store_ps(io + i + 56, f7);
        }
        return i;
    }

}
void vectorized::shortToFloat(const std::int16_t * __restrict__ in, float * __restrict__ out, const std::size_t length){
    std::size_t i{0};
    if(length > 64){
        const std::size_t stop{length - 64};
        for(i = 0; i < stop; i += 64)
        {
            //done like this so the compiler will unroll the inner loop for us
            for(int j = 0; j < 64; j += 16)
            {
                const __m256i in0 = _mm256_load_si256((const __m256i *)(in + i + j));
                const __m128i in1 = _mm256_extracti128_si256(in0, 0);
                const __m128i in2 = _mm256_extracti128_si256(in0, 1);
                const __m256i tmp1 = _mm256_cvtepi16_epi32(in1);
                const __m256i tmp2 = _mm256_cvtepi16_epi32(in2);
                const __m256  out1 = _mm256_cvtepi32_ps(tmp1);
                const __m256  out2 = _mm256_cvtepi32_ps(tmp2);
                _mm256_store_ps(out + i + j + 0, out1);
                _mm256_store_ps(out + i + j + 8, out2);
            }
        }
        _mm256_zeroupper();
    }
    for(; i < length; i++)
    {
        out[i] = (float)in[i];
    }
}

void vectorized::floatToShort(const float * __restrict__ in, std::int16_t * __restrict__ out, const std::size_t length){
    std::size_t i{0};
    if(length > 16){
        __m256  in0  ALIGN32_END;
        __m256  in1  ALIGN32_END;
        __m256i out0 ALIGN32_END;
        __m256i out1 ALIGN32_END;

        const std::size_t stop{length - 16};

        for(i = 0; i < stop; i += 16)
        {
            in0 = _mm256_load_ps(in + i + 0);              //read float 0-7
            in1 = _mm256_load_ps(in + i + 8);              //read float 8-15
            out0 = _mm256_cvttps_epi32(in0);               //convert to int
            out1 = _mm256_cvttps_epi32(in1);               //convert to int

            out0 = _mm256_packs_epi32(out0, out1);         //pack integers as saturated shorts (packed in this order: 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15)
            out0 = _mm256_permute4x64_epi64(out0, 0XD8);   //shuffle to 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

            _mm256_store_si256((__m256i *)(out + i), out0); //store
        }
    }
    for(; i < length; i++)
    {
        out[i] = (short)in[i];
    }

}

void vectorized::complexToFloat(const float* __restrict__ in, float* __restrict__ out, std::size_t length) {
    std::size_t i;
    const std::size_t stop{(length*2) - 32};

    //NOTE: I'm counting over the floats of 'in', not complexes.
    for(i = 0; i < stop; i += 32) {
        __m256 v0 = _mm256_load_ps(in + i +  0);
        __m256 v1 = _mm256_load_ps(in + i +  8);
        __m256 v2 = _mm256_load_ps(in + i + 16);
        __m256 v3 = _mm256_load_ps(in + i + 24);

        v0 = _mm256_mul_ps(v0, v0);
        v1 = _mm256_mul_ps(v1, v1);
        v2 = _mm256_mul_ps(v2, v2);
        v3 = _mm256_mul_ps(v3, v3);

        v0 = _mm256_hadd_ps(v0,v1);
        v2 = _mm256_hadd_ps(v2,v3);

        v0 = _mm256_permutevar8x32_ps(v0, magic4);
        v2 = _mm256_permutevar8x32_ps(v2, magic4);

        _mm256_store_ps(out + (i>>1) + 0, v0);
        _mm256_store_ps(out + (i>>1) + 8, v2);
    }
    //not I'm back to counting over complexes.  Please forgive my sins.
    for (i/=2; i < length; i++) {
        const float& ii = in[i*2];
        const float& iq = in[i*2 + 1];
        out[i] = ii*ii + iq*iq;
    }

    //At this point, I have the square of the magnitude.  I'm going to do the sqrt in a different loop, to try
    //getting more streaming benefit (12 CPI latency, 6 CPI throughput).
     const std::size_t stop2{length - 31};
     for (i = 0; i < stop2; i += 32) {
         __m256 v0 = _mm256_load_ps(out + i +  0);
         __m256 v1 = _mm256_load_ps(out + i +  8);
         __m256 v2 = _mm256_load_ps(out + i + 16);
         __m256 v3 = _mm256_load_ps(out + i + 24);

         v0 = _mm256_sqrt_ps(v0);
         v1 = _mm256_sqrt_ps(v1);
         v2 = _mm256_sqrt_ps(v2);
         v3 = _mm256_sqrt_ps(v3);

         _mm256_store_ps(out + i +  0, v0);
         _mm256_store_ps(out + i +  8, v1);
         _mm256_store_ps(out + i + 16, v2);
         _mm256_store_ps(out + i + 24, v3);
     }
     for (; i < length; i++) {
         out[i] = std::sqrt(out[i]);
     }
}

void vectorized::reverse(const float * __restrict__ in, float * __restrict__ out,  std::size_t length) {
    std::size_t i;
    const std::size_t s2 = length - 32;
    for(i = 0; i < s2; i += 32)
    {
        __m256 v0 = _mm256_load_ps(in + i +   0);
        __m256 v1 = _mm256_load_ps(in + i +   8);
        __m256 v2 = _mm256_load_ps(in + i +  16);
        __m256 v3 = _mm256_load_ps(in + i +  24);

        v0 = _mm256_permute2f128_ps(v0, v0, 1);
        v1 = _mm256_permute2f128_ps(v1, v1, 1);
        v2 = _mm256_permute2f128_ps(v2, v2, 1);
        v3 = _mm256_permute2f128_ps(v3, v3, 1);
        v0 = _mm256_permute_ps(v0, 0x1B);
        v1 = _mm256_permute_ps(v1, 0x1B);
        v2 = _mm256_permute_ps(v2, 0x1B);
        v3 = _mm256_permute_ps(v3, 0x1B);

        _mm256_storeu_ps((out + length) - (i +  8), v0);
        _mm256_storeu_ps((out + length) - (i + 16), v1);
        _mm256_storeu_ps((out + length) - (i + 24), v2);
        _mm256_storeu_ps((out + length) - (i + 32), v3);
    }
    _mm256_zeroupper();
    for(; i < length; i++)
    {
        out[(length - i) - 1] = in[i];
    }
}

void vectorized::multiply(  float* __restrict__ io, const float num,  const std::size_t size){
    __m256 mul ALIGN32_END;

    mul = _mm256_broadcast_ss(&num);
    std::size_t i = doVectorizedMultiply(io, mul, size);
    _mm256_zeroupper();

    for (; i < size; i++){
        io[i] = io[i] * num;
    }
}

void vectorized::multiplyIQ(float* __restrict__ io,  float ival,  float qval,  std::size_t size){
    float mulArray[8] ALIGN32_END;
    mulArray[0] = ival;
    mulArray[1] = qval;
    mulArray[2] = ival;
    mulArray[3] = qval;
    mulArray[4] = ival;
    mulArray[5] = qval;
    mulArray[6] = ival;
    mulArray[7] = qval;

    std::size_t i = doVectorizedMultiply(io, *((__m256*)(&mulArray[0])), size);
    _mm256_zeroupper();
    for(; i < size; i+= 2){
        io[i]   *= ival;
        io[i+1] *= qval;
    }

}

void vectorized::add(std::int16_t* __restrict__ io, const std::int16_t* __restrict__ second,  std::size_t size){
    std::size_t i = 0;
    const std::size_t stop = size - 32;
    for (; i <= stop; i += 32)
    {
        __m256i ini0 = _mm256_load_si256((__m256i *)(io + i + 0));
        __m256i ini1 = _mm256_load_si256((__m256i *)(io + i + 16));
        __m256i ins0 = _mm256_load_si256((const __m256i *)(second + i + 0));
        __m256i ins1 = _mm256_load_si256((const __m256i *)(second + i + 16));

        ini0 = _mm256_adds_epi16(ini0, ins0);
        ini1 = _mm256_adds_epi16(ini1, ins1);

        _mm256_store_si256((__m256i *)(io + i + 0), ini0);
        _mm256_store_si256((__m256i *)(io + i + 16), ini1);
    }
    _mm256_zeroupper();
    for (; i < size; i++)
    {
        io[i] = (int16_t)std::min(io[i] + second[i], (int)std::numeric_limits<int16_t>::max());
    }

}

void vectorized::addUNA(std::int16_t* __restrict__ io, const std::int16_t* __restrict__ second,  std::size_t size){
    std::size_t i = 0;
    const std::size_t stop = size - 32;
    for (; i <= stop; i += 32)
    {
        __m256i ini0 = _mm256_loadu_si256((__m256i *)(io + i + 0));
        __m256i ini1 = _mm256_loadu_si256((__m256i *)(io + i + 16));
        __m256i ins0 = _mm256_loadu_si256((const __m256i *)(second + i + 0));
        __m256i ins1 = _mm256_loadu_si256((const __m256i *)(second + i + 16));

        ini0 = _mm256_adds_epi16(ini0, ins0);
        ini1 = _mm256_adds_epi16(ini1, ins1);

        _mm256_storeu_si256((__m256i *)(io + i + 0), ini0);
        _mm256_storeu_si256((__m256i *)(io + i + 16), ini1);
    }
    _mm256_zeroupper();
    for (; i < size; i++)
    {
        io[i] = (int16_t)std::min(io[i] + second[i], (int)std::numeric_limits<int16_t>::max());
    }
}

void vectorized::add(float* __restrict__ io, const float* __restrict__ second,  std::size_t size){
    std::size_t i = 0;
    const std::size_t stop = size - 32;
    for(; i <= stop; i += 32)
    {
        __m256 ini0 = _mm256_load_ps(io     + i +  0);
        __m256 ini1 = _mm256_load_ps(io     + i +  8);
        __m256 ini2 = _mm256_load_ps(io     + i + 16);
        __m256 ini3 = _mm256_load_ps(io     + i + 24);
        __m256 ins0 = _mm256_load_ps(second + i +  0);
        __m256 ins1 = _mm256_load_ps(second + i +  8);
        __m256 ins2 = _mm256_load_ps(second + i + 16);
        __m256 ins3 = _mm256_load_ps(second + i + 24);

        ini0 = _mm256_add_ps(ini0, ins0);
        ini1 = _mm256_add_ps(ini1, ins1);
        ini2 = _mm256_add_ps(ini2, ins2);
        ini3 = _mm256_add_ps(ini3, ins3);

        _mm256_store_ps(io + i +  0, ini0);
        _mm256_store_ps(io + i +  8, ini1);
        _mm256_store_ps(io + i + 16, ini2);
        _mm256_store_ps(io + i + 24, ini3);
    }
    _mm256_zeroupper();
    for(; i < size; i++)
    {
        io[i] = io[i] + second[i];
    }
}

void vectorized::addUNA(float* __restrict__ io, const float* __restrict__ second, std::size_t size){
    std::size_t i = 0;
    const std::size_t stop = size - 32;
    for(; i <= stop; i += 32)
    {
        __m256 ini0 = _mm256_loadu_ps(io     + i +  0);
        __m256 ini1 = _mm256_loadu_ps(io     + i +  8);
        __m256 ini2 = _mm256_loadu_ps(io     + i + 16);
        __m256 ini3 = _mm256_loadu_ps(io     + i + 24);
        __m256 ins0 = _mm256_loadu_ps(second + i +  0);
        __m256 ins1 = _mm256_loadu_ps(second + i +  8);
        __m256 ins2 = _mm256_loadu_ps(second + i + 16);
        __m256 ins3 = _mm256_loadu_ps(second + i + 24);

        ini0 = _mm256_add_ps(ini0, ins0);
        ini1 = _mm256_add_ps(ini1, ins1);
        ini2 = _mm256_add_ps(ini2, ins2);
        ini3 = _mm256_add_ps(ini3, ins3);

        _mm256_storeu_ps(io + i +  0, ini0);
        _mm256_storeu_ps(io + i +  8, ini1);
        _mm256_storeu_ps(io + i + 16, ini2);
        _mm256_storeu_ps(io + i + 24, ini3);
    }
    _mm256_zeroupper();
    for(; i < size; i++)
    {
        io[i] = io[i] + second[i];
    }
}

void vectorized::iqToAmpAndPhase(const float * __restrict__ in, float * __restrict__ out, std::size_t size) {
    static constexpr float piFloat{static_cast<float>(M_PI)};
    __m256 val0 ALIGN32_END;
    __m256 val1 ALIGN32_END;
    __m256 tmp0 ALIGN32_END;
    __m256 tmp1 ALIGN32_END;
    __m256 tmp2 ALIGN32_END;
    __m256 tmp3 ALIGN32_END {};

    __m256 tp ALIGN32_END;
    tp = _mm256_broadcast_ss(&piFloat);

    const std::size_t stop = size - 16;
    std::size_t i = 0;
    for(; i < stop; i+=16){
        val0 = _mm256_load_ps(in + i);
        val1 = _mm256_load_ps(in + i + 8);

        tmp0 = _mm256_mul_ps(val0, val0);
        tmp1 = _mm256_mul_ps(val1, val1);

        tmp2 = _mm256_hadd_ps(tmp0, tmp1);
        //TODO -> find a way to make rsqrt work!!!
        tmp2 = _mm256_sqrt_ps(tmp2);          // 0,1,4,5,2,3,6,7

        //TODO -> this needs tested!
        // avx_mathfun::atan2(val0, val1, tmp3); // 0,1,4,5,2,3,6,7
        tmp3 = avx_mathfun::atan2(tmp0, tmp1);

        // convert from -pi,+pi to 0,2pi
        tmp3 = _mm256_add_ps(tmp3, tp);

        tmp0 = _mm256_unpackhi_ps(tmp2, tmp3);
        tmp1 = _mm256_unpacklo_ps(tmp2, tmp3);

        _mm256_store_ps(out + i + 0, tmp1);
        _mm256_store_ps(out + i + 8, tmp0);
    }
    _mm256_zeroupper();
    for(; i < size; i+=2){
        out[i+0] = std::hypot(in[i], in[i+1]);
        out[i+1] = std::atan2(in[i], in[i+1]) + piFloat;
    }
}

const __m256& vectorized::zeroVec() {
    static __m256 ret ALIGN32_END;
    static bool set{false};
    if (!set) {
        std::mutex lock;
        std::lock_guard<std::mutex> lk(lock);
        if (!set) {
            constexpr float zero{0.0F};
            set = true;
            ret = _mm256_broadcast_ss(&zero);
        }
    }
    return ret;
}

void vectorized::vectorMultiply(float* __restrict__ io, const float* __restrict__ b, const std::size_t size)
{
    // The below code performs the complex multiplication "(x + yi)(u + vi) = (xu - yv)+(xv + yu)i" on
    // eight pairs of complex numbers per loop iteration.
    // It does this in two sets of four (the variables that end in 0 are one set, 1 is the other set).
    //
    // Why eight?  Because that is the most I can do while both: using <= 8 variables, maintaining some sort of readability

    __m256 inb0  ALIGN32_END;
    __m256 inc0  ALIGN32_END;
    __m256 inb1  ALIGN32_END;
    __m256 inc1  ALIGN32_END;
    __m256 real0 ALIGN32_END;
    __m256 imag0 ALIGN32_END;
    __m256 real1 ALIGN32_END;
    __m256 imag1 ALIGN32_END;

    const std::size_t stop = size - 16;
    std::size_t i = 0;
    for (; i <= stop; i += 16)
    {
        inb0 = _mm256_load_ps(io + i + 0);           //x y x y  (8X32)
        inb1 = _mm256_load_ps(io + i + 8);           //x y x y  (8X32)
        inc0 = _mm256_load_ps(b + i + 0);            //u v u v  (8X32)
        inc1 = _mm256_load_ps(b + i + 8);            //u v u v  (8X32)

        real0 = _mm256_mul_ps(inc0, negateFloat);    // u -v u -v (8X32)
        real1 = _mm256_mul_ps(inc1, negateFloat);    // u -v u -v (8X32)

        imag0 = _mm256_permute_ps(inc0, 0xB1);       //v u v u (8X32)
        imag1 = _mm256_permute_ps(inc1, 0xB1);       //v u v u (8X32)

        real0 = _mm256_mul_ps(real0, inb0);          //xu -yv xu -yv (8X32)
        real1 = _mm256_mul_ps(real1, inb1);          //xu -yv xu -yv (8X32)

        imag0 = _mm256_mul_ps(imag0, inb0);          //xv yu xv yu (8X32)
        imag1 = _mm256_mul_ps(imag1, inb1);          //xv yu xv yu (8X32)

        real0 = _mm256_hadd_ps(real0, imag0);        //xu-yv xu-yv xv+yu xv+yu (8X32)
        real1 = _mm256_hadd_ps(real1, imag1);        //xu-yv xu-yv xv+yu xv+yu (8X32)

        imag0 = _mm256_permute_ps(real0, 0xD8);      //xu-yv xv+yu xu-yv xv+yu (8X32)
        imag1 = _mm256_permute_ps(real1, 0xD8);      //xu-yv xv+yu xu-yv xv+yu (8X32)

        _mm256_store_ps(io + i + 0, imag0);
        _mm256_store_ps(io + i + 8, imag1);
    }
    _mm256_zeroupper();
    for (; i < size; i += 2)
    {
        std::complex<float> *d = reinterpret_cast<std::complex<float>*>(&(io[i]));
        const std::complex<float> *c = reinterpret_cast<const std::complex<float>*>(&(b[i]));
        *d = *d * *c;
    }
}

void vectorized::applyLinearLossAdjustment(float* __restrict__ data, std::size_t size, const float& initialChange, const float& finalChange) {
    static constexpr float fourS{4};
    static const __m256 increment ALIGN32_END = _mm256_setr_ps(0,0,1,1,2,2,3,3);
    static const __m256 four      ALIGN32_END = _mm256_broadcast_ss(&fourS);

    const float changePerStep = (finalChange - initialChange) / static_cast<float>(size / 2);
    const __m256 step = _mm256_broadcast_ss(&changePerStep);


    const std::size_t stop = size - 15;
    std::size_t i = 0;
    if (size > 15) {
        for(; i < stop; i+=16) {
            __m256 a = _mm256_load_ps(data + i);    //first 8 numbers (four samples)
            __m256 b = _mm256_load_ps(data + i + 8);//second 8 numbers (four samples)

            const float iFa = static_cast<float>(i/8);
            const float iFb = static_cast<float>(iFa+1.0F);

            const __m256 thisStepa =  _mm256_fmadd_ps(four, _mm256_broadcast_ss(&iFa), increment);
            const __m256 thisStepb =  _mm256_fmadd_ps(four, _mm256_broadcast_ss(&iFb), increment);

            const __m256 thisStepChangea = _mm256_mul_ps(thisStepa, step);
            const __m256 thisStepChangeb = _mm256_mul_ps(thisStepb, step);

            a = _mm256_mul_ps(a, thisStepChangea);
            b = _mm256_mul_ps(b, thisStepChangeb);

            _mm256_store_ps(data + i + 0, a);
            _mm256_store_ps(data + i + 8, b);
        }
    }
    for (; i < size; i += 2) {
        const float change = changePerStep * (float)(i/2);
        data[i] *= change;
        data[i + 1] *= change;
    }
}