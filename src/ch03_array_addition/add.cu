#include <stdio.h>
#include <math.h>
#include <stdlib.h>

const double epsilon = 1.0e-15;
const double a = 1.05;
const double b = 2.65;
const double c = 5.34;

// GPU 核函数：并行数组相加
__global__ void add_kernel(const double *x, const double *y, double *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

// 检查结果（在 CPU 上）
void check(const double *z, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(z[i] - c) > epsilon) {
            printf("Error: z[%d] = %f\n", i, z[i]);
            exit(1);
        }
    }
}

int main(void) {
    const int N = 1000;
    printf("Add two arrays of %d elements using CUDA\n", N);

    size_t size = N * sizeof(double);

    // 在主机上分配数组
    double *h_x = (double *)malloc(size);
    double *h_y = (double *)malloc(size);
    double *h_z = (double *)malloc(size);

    // 初始化数组
    for (int i = 0; i < N; i++) {
        h_x[i] = a;
        h_y[i] = b;
    }

    // 在设备上分配数组
    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_z, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // 核函数执行配置
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 在 GPU 上执行核函数
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, N);
    cudaDeviceSynchronize();  // 等待 GPU 执行完成

    // 将结果从设备复制回主机
    cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

    // 检查结果
    check(h_z, N);

    printf("Check passed!\n");

    // 释放内存
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}
