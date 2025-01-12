# Mixed precision math library for CUDA
A lightweight library to emulate extended precision on retail NVIDIA GPUs for higher performance than native doubles.

Retail GPUs usually lack many fp64 cores leading to a loss of performance. A simple way to fix that issue is to emulate the performance of doubles by using two floats. This library allows the user to do just that. 

This library can be extended to do mixed precision math using a combination of any formats leading to potential higher accuracy and performance for non-scientific applications such as machine learning. 

The difference in performance between native double precision and the simulated double precision has been computed for the N-body problem and presented here.


