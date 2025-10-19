// hello012.cu

#include <stdio.h>  // kernel function do not support <iostream> of cpp

int main() {
    hello0();  // host print
    // kernel function print, with device
    hello1<<<1, 1>>>();  // 1, 1: grid size and block size
    cudaDeviceSynchronize();  // wait for kernel function to finish
    hello2<<<1, 1>>>();  // 1, 1: grid size and block size
    cudaDeviceSynchronize();  // wait for kernel function to finish
    return 0;
}

void hello0() {
    printf("Hello, world!\n");
}

__global__ void hello1() {
    printf("Hello, world!\n");
}

void __global__ hello2() {
    printf("Hello, world!\n");
}