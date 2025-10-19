1. 主机（Host）：CPU。负责程序控制、数据管理和核函数调用。

2. 设备（Device）：GPU。负责执行核函数进行大规模并行计算。

3. 核函数（Kernel Function）：在 GPU 上执行的函数，需要使用 `__global__` 关键字进行声明。

4. 调用核函数时，需要指定 **网格大小（grid size）** 和 **块大小（block size）**。格式为：

   ```cpp
   kernel<<<grid_size, block_size>>>();
   ```

   示例：

   ```cpp
   hello<<<1, 1>>>();
   ```

   表示一个网格中有一个块，该块有一个线程。

   * 一个 GPU 有很多计算核心，可以支持大量线程执行并行任务，因此需要指定网格和块的大小来控制线程总数。

5. `cudaDeviceSynchronize()`：等待设备上所有线程执行完毕。作用：确保主机上的代码在继续执行之前，核函数已经完成执行。

6. CUDA 头文件：

   * `cuda_runtime.h`：提供 CUDA 运行时 API。
   * `cuda.h`：包含 `stdlib.h` 等标准库头文件，nvcc 编译 *.cu 文件时会自动包含。

7. nvcc：NVIDIA CUDA Compiler，用于编译 CUDA 程序。

8. 查看 nvcc 版本：

   ```bash
   nvcc -V
   ```

9. 编译 CUDA 文件生成可执行程序：

   ```bash
   nvcc -o hello3 hello.cu
   ```

10. 使用 CMake 编译 CUDA 程序：

    * 在 `CMakeLists.txt` 中包含：

      ```cmake
      find_package(CUDA REQUIRED)
      ```
    * 好处：管理大型项目、自动处理依赖、跨平台编译。

11. nvcc 编译流程：

    ```
    CUDA 代码 --> PTX（Parallel Thread Execution）--> cubin（二进制代码）
    ```

12. nvcc Just-In-Time（JIT）编译：

    * 在运行时，将 CUDA 代码先编译为 PTX 代码（中间表示），再生成 cubin 二进制代码。

13. 指定 GPU 架构与计算能力：

    ```bash
    -gencode arch=compute_XX,code=sm_XX
    ```

    示例：

    ```bash
    -gencode arch=compute_75,code=sm_75
    ```

    * 表示编译目标为 Turing 架构（计算能力 7.5）的 GPU。