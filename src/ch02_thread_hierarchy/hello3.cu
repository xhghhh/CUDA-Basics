// hello3.cu
// multi-thread in blocks

int main() {
    hello_with_gpu<<<1, 2>>>();  // 1, 2: grid size and block size
    cudaDeviceSynchronize();  // wait for kernel function to finish
    hello_with_gpu<<<2, 3>>>();  // 2, 3: grid size and block size, 2 times 3 equals 6 threads in total, so there are 6 "Hello, world!\n" printed
    cudaDeviceSynchronize();  // wait for kernel function to finish
    return 0;
}


__global__ void hello_with_gpu() {
    printf("------------------\n");
    printf("Hello, world!\n");
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    printf("block_id: %d, thread_id: %d\n", block_id, thread_id);
    printf("------------------\n");
}
