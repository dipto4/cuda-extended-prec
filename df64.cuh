#pragma once

#include<cuda_runtime.h>
#include<cstdio>


constexpr double SPLITTER = (1 << 29) + 1;
constexpr float SPLITTERMULT = (1 << 12) + 1;


class df64 {
    float2 val;

    __host__ __device__ float2 _quickTwoSum(float a, float b) {
        float s = a + b;
        float e = b - (s - a);
        return make_float2(s, e);
    }

    __host__ __device__ float2 _twoSum(float a, float b) {
        float s = a + b;
        float v = s - a;
        float e = (a - (s - v)) + (b-v);
        return make_float2(s, e);
    }
    

    __host__ __device__ float2 _split(double a) {
        double t = a * SPLITTER;
        double t_hi = t - (t - a);
        double t_lo = a - t_hi;

        return make_float2(t_hi, t_lo);
    }

    __host__ __device__ float4 _twoSumComp(float2 a , float2 b) {
        float2 s = make_float2(a.x + b.x, a.y + b.y);
        float2 v = make_float2(s.x - a.x, s.y-a.y);
        float2 e = make_float2((a.x - (s.x-v.x)) + (b.x-v.x), (a.y - (s.y-v.y)) + (b.y-v.y));
        return make_float4(s.x, e.x, s.y, e.y);
    }

    __host__ __device__ float2 _splitMult(float a) {
        float t = a * SPLITTERMULT;
        float a_hi = t - (t - a);
        float a_lo = a - a_hi;
        return make_float2(a_hi, a_lo);
    }

    __host__ __device__ float4 _splitMultComp(float2 a) {
        float2 t = make_float2(a.x * SPLITTERMULT, a.y * SPLITTERMULT);

        float2 a_hi = make_float2(t.x - (t.x - a.x), t.y - (t.y - a.y));
        float2 a_lo = make_float2(a.x - a_hi.x, a.y - a_hi.y);
        return make_float4(a_hi.x, a_lo.x, a_hi.y, a_lo.y);
    }

    __host__ __device__ float2 _twoProd(float a, float b) {
        float p = a * b;
        float2 aS = _splitMult(a);
        float2 bS = _splitMult(b);
        float err = ((aS.x * bS.x -p) + aS.x * bS.y + aS.y * bS.x) + aS.y * bS.y;
        return make_float2(p , err);
    }
    
    __host__ __device__ float2 _twoProdComp(float a, float b) {
        float p = a * b;
        float4 abS = _splitMultComp(make_float2(a,b));
        //abS.x = aS.x , abS.y = aS.y , abS.z = bS.x, abS.w = bS.y
        float err = ((abS.x * abS.z -p) + abS.x * abS.w + abS.y * abS.z) + abS.y * abS.w;
        return make_float2(p,err);
    }


    public:
    // constructors and assignment operators 
    __host__ __device__ df64() : val(make_float2(0.0,0.0)) {}
    
    __host__ __device__ df64(const float a) : val(make_float2(a, 0.0)) {}
    
    __host__ __device__ df64(const double a) : val(_split(a)) {}
    
    __host__ __device__ df64(const float2 a) : val(a) {}

    __host__ __device__ df64(const df64& a) : val(a.val) {}

    __host__ __device__ df64& operator=(const float& a) {
        this->val = make_float2(a,0.0);
        
        return *this;
    }

    __host__ __device__ df64& operator=(const double& a) {
        this->val = _split(a);
        
        return *this;
    }
    
    __host__ __device__ df64& operator=(const df64& a) {
        if(this != &a) {
            this->val = a.val;
        }
        return *this;
    }
    
    __host__ __device__ df64 operator-() const {
        return df64(make_float2(-val.x,-val.y));
    }


    __host__ __device__ float to_float() {
        return static_cast<float>(val.x) + static_cast<float>(val.y);
    }

    __host__ __device__ double to_double() {
        return static_cast<double>(val.x) + static_cast<double>(val.y);
    }

    // mathematical operators begin here

    __host__ __device__ df64 operator+(df64 const& a) {
        // NOTE: the following version works but doesn't use vectorization
        // that is why this is not being used. Set the flags to use this
#ifdef NO_USE_VECTORSUM
        float2 s, t;
        s = _twoSum(this->val.x, a.val.x);
        t = _twoSum(this->val.y, a.val.y);
        s.y += t.x;
        s = _quickTwoSum(s.x, s.y);
        s.y += t.y;
        s = _quickTwoSum(s.x,s.y);
        val = s;
        return *this;
#else
        float4 st;
        float2 xy;
        st = _twoSumComp(this->val, a.val);
        st.y += st.z;
        xy = _quickTwoSum(st.x,st.y);
        st.y += st.w;
        xy = _quickTwoSum(st.x,st.y);
        val = xy;
        return *this;
#endif

    }

    __host__ __device__ df64 operator*(df64 const& a) {
        float2 p;
#ifdef NO_USE_VECTORMULT
        p = _twoProd(this->val.x, a.val.x);
#else
        p = _twoProdComp(this->val.x,a.val.x);
#endif
        p.y += this->val.x * a.val.y;
        p.y += this->val.y * a.val.x;
        p = _quickTwoSum(p.x, p.y);
        this->val = p;
        return *this;
    }


    // note this represents division BY a
    __host__ __device__ df64 operator/(df64 const& a) {
        // use the Karp method for division
        
        float xn = 1.0f/a.val.x;
        float yn = this->val.x * xn;
        df64 prod0 = df64(yn);
        prod0 =  prod0 * a;
        float diff = (*this +  -prod0).val.x;
        float2 prod = _twoProd(xn, diff);
        df64 sum = df64(prod) + df64(yn);
        this->val = sum.val;
        return *this;


    }
    
    // some comparsion operators

    __host__ __device__ bool operator==(df64 const& a) {
        return (this->val.x == a.x) && (this->val.y == a.y);
    }
    
    __host__ __device__ bool operator!=(df64 const& a) {
        return (this->val.x != a.x) || (this->val.y != a.y);
    }

    __host__ __device__ bool operator<=(df64 const& a) {
        return (this->val.x < a.x || (this->val.x == a.x && this->val.y < a.y));
    }

    // some relevant mathematical tools
};
