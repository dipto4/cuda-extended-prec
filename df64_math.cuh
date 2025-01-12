#pragma once
#include "df64.cuh"


__host__ __device__ df64 sqrt(df64 a) {
    float xn = rsqrtf(a.val.x);
    float yn = a.val.x * xn;
    df64 ynsqr = yn;
    ynsqr = yn * yn;
    float diff = (a + -ynsqr).val.x;
    float2 prod = _twoProd(xn, diff);
    prod.x = prod.x / 2;
    prod.y = prod.y / 2;
    df64 sum = df64(yn) + df64(prod);
    return sum;
}

__host__ __device__ inline df64 log(const df64& a) {

}
 
__host__ __device__ inline df64 sin(const df64& a) {

}

__host__ __device__ inline df64 cos(const df64& a) {

}

__host__ __device__ inline df64 exp(const df64& a) {

}





