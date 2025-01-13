#pragma once
#include "../df64.cuh"

template<typename T, int p>
__device__ void pp_force(const vec3<T>& sink, const vec3<T>* __restrict__ sources, const T* __restrict__ m, vec3<T> acc) {
#pragma unroll
    for(size_t i = 0 ; i < p ; i++) {
        vec3 dx = sources[i] - sink[i];
        T d2 = dx3.norm2();
        T d = sqrt(d);
        T one_over_d3 = 1.0/(d2 * d);
        vec3 ai =  -(m[i]) * one_over_d3 * dx;
        acc += ai;
    }
}

template<typename T, int p>
__global__ void calculate_p2p(const vec3<T>* __restrict__ part, const T* __restrict__ m, vec3<T>* __restrict__ acc, const size_t N) {
    __shared__ vec3<T> cache[p];
    __shared__ T cache_m[p];

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    vec3<T> cur_part = part[tid];
    vec3<T> acc_i;
    for(size_t i = 0, tile = 0; i < N; i+=p, tile++) {
        cache[threadIdx.x] = blockDim.x * tile + threadIdx.x;

        __syncthreads();

        pp_force<T,p>(cur_part, cache, cache_m, acc_i);
        
        __syncthreads();

    }
    
    //__syncthreads();

    acc[tid] = acc_i;    

} 
