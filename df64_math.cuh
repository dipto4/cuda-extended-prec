#pragma once
#include "df64.cuh"

#define ONE 1.0
#define ZERO 0.0


__host__ __device__ df64 sqrt(const df64 &a) {
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



__host__ __device__ inline df64 exp(const df64& a) {
    const float thresh = 1.0e-20 * exp(a->get_x());
    df64 t, p , f, s, x;
    float m;

    s = a + df64(1.0f);
    p = a * a;
    m = 2.0f;
    f = df64(2.0f);
    t = p / df64(2.0f);

    while(abs(t->get_x()) > thresh) {
        s = s + t;
        p = p * a;
        m += 1.0f;
        f = f * df64(m);
        t = p / f;
    }

    return s + t;
}


__host__ __device__ inline df64 log(const df64& a) {
    df64 xi;

    if(!(a == 1.0f)) {
        if(a.get_x() <= 0.0) {
            xi  = make_float2(NAN, NAN);
        } else {
            xi.x = logf(a.get_x());
            xi  = (xi + (exp(-xi) * a)) + df64(-1.0);  
        }
    }
    return xi;
}
 
__host__ __device__ inline df64 sin(const df64& a) {
    const float thresh = 1.0e-20 * abs(a->get(x));
    df64 t, p , f , s, x;
    float m;

    df64 sin_a;

    if(a->get_x() == 0.0f) {
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
            f = f * ds64(m*(m-1));
            t = p / f;
            s = s + t;
            if(abs(t->get_x()) < thresh) {
                break;
            }
        }
        sin_a = s;
    }

    return sin_a;
}

__host__ __device__ inline df64 cos(const df64& a) {
    const float thresh = 1.0e-20 * abs(a->get(x));
    df64 t, p , f , s, x;
    float m;

    df64 cos_a;

    if(a->get_x() == 0.0f) {
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
            f = f * ds64(m*(m-1));
            t = p / f;
            s = s + t;
            if(abs(t->get_x()) < thresh) {
                break;
            }
        }
        cos_a = sqrt(df64(ONE) + - (s * s));
    }

    return cos_a;

}



