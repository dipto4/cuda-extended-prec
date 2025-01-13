#pragma once
#include "df64.cuh"

#define ONE 1.0
#define ZERO 0.0


__host__ __device__ df64 sqrt(const df64 &a) {
    float xn = rsqrtf(a.val.x);
    float yn = a.val.x * xn;
    df64 ynsqr = df64(df64::_twoProdComp(yn , yn));

    df64 diff_ = -ynsqr + a;
    float diff = diff_.val.x;
    df64 prod = df64(df64::_twoProdComp(xn / 2.0, diff));
    df64 sum = df64(yn) + prod;
    return sum;
}



__host__ __device__  df64 exp(const df64& a) {
    const float thresh = 1.0e-20 * expf(a.val.x);
    df64 t, p , f, s, x;
    float m;

    s = df64(1.0) + a;
    p = a * a;
    m = 2.0f;
    f = df64(2.0f);
    t = p / df64(2.0f);

    while(abs(t.val.x) > thresh) {
        s = s + t;
        p = p * a;
        m += 1.0f;
        f = f * df64(m);
        t = p / f;
    }

    return s + t;
}


__host__ __device__ df64 log(const df64& a) {
    df64 xi;

    if(!(a == df64(1.0f))) {
        if(a.val.x <= 0.0) {
            xi  = make_float2(NAN, NAN);
        } else {
            xi.val.x = logf(a.val.x);
            xi  = (xi + (exp(-xi) * a)) + df64(-1.0);  
        }
    }
    return xi;
}
 
__host__ __device__ df64 sin(const df64& a) {
    const float thresh = 1.0e-20 * abs(a.val.x);
    df64 t, p , f , s, x;
    float m;

    df64 sin_a;

    if(a.val.x == 0.0f) {
        sin_a =  df64(make_float2(ZERO,ZERO));
    } else {
        x = - (a * a);
        s = a;
        p = a;
        m = ONE;
        f = df64(make_float2(ONE,ZERO));

        while(true) {
            p = p * x;
            m += 2.0f;
            f = f * df64(m*(m-1));
            t = p / f;
            s = s + t;
            if(abs(t.val.x) < thresh) {
                break;
            }
        }
        sin_a = s;
    }

    return sin_a;
}

__host__ __device__ df64 cos(const df64& a) {
    const float thresh = 1.0e-20 * abs(a.val.x);
    df64 t, p , f , s, x;
    float m;

    df64 cos_a;

    if(a.val.x == 0.0f) {
        cos_a =  df64(make_float2(ONE,ZERO));
    } else {
        x = - (a * a);
        s = a;
        p = a;
        m = ONE;
        f = df64(make_float2(ONE,ZERO));

        while(true) {
            p = p * x;
            m += 2.0f;
            f = f * df64(m*(m-1));
            t = p / f;
            s = s + t;
            if(abs(t.val.x) < thresh) {
                break;
            }
        }
        cos_a = sqrt(df64(ONE) + - (s * s));
    }

    return cos_a;

}



