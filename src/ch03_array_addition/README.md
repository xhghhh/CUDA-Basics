1. **一个典型的 CUDA 程序基本框架**

```cpp
// 头文件包含
// 常量定义，宏定义
// C++ 自定义函数和 CUDA 核函数声明

int main(void){
    // 分配 host & device 内存
    // 初始化 host 内存
    // 将 host 内存拷贝到 device 内存
    // 调用 CUDA 核函数
    // 将 device 内存拷贝到 host 内存
    // 释放 host & device 内存
}

// C++ 自定义函数和 CUDA 核函数实现
```

---

2. **GPU 加速原理**

* 在 GPU 的不同计算单元上同时执行多个线程，实现并行计算。
* 这是 GPU 加速矩阵运算等的基本原理。

---

3. **分配内存**

* 在 C++ 中用 `malloc` 分配 host 内存。
* CUDA 用 `cudaMalloc` 分配 device 内存：

```cpp
cudaError_t cudaMalloc(void **devPtr, size_t size);
```

* `devPtr`：指向 device 内存指针的地址（双重指针）

* `size`：要分配的内存大小（字节）

* 返回值为 `cudaError_t`，表示分配是否成功

* **双重指针原因**：`cudaMalloc` 需要修改指针本身的值（分配后的地址），所以传入指针的地址：

```cpp
cudaMalloc(&devPtr, size);
```

---

4. **释放内存**

```cpp
cudaError_t cudaFree(void *devPtr);
```

* `devPtr`：要释放的 device 内存指针
* 返回值为 `cudaError_t`

---

5. **Host 与 Device 数据传递**

```cpp
cudaError_t cudaMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind);
```

* `dst`：目标内存地址
* `src`：源内存地址
* `size`：拷贝字节数
* `kind`：拷贝方向，如：

  * `cudaMemcpyHostToDevice`：主机 → 设备
  * `cudaMemcpyDeviceToHost`：设备 → 主机
  * `cudaMemcpyDeviceToDevice`：设备 → 设备
  * `cudaMemcpyHostToHost`：主机 → 主机
  * `cudaMemcpyDefault`：根据地址自动选择方向

**示例**：

```cpp
cudaMemcpy(dev_x, x, M, cudaMemcpyHostToDevice);
cudaMemcpy(dev_y, y, M, cudaMemcpyHostToDevice);

add<<<grid, block>>>(dev_x, dev_y, dev_z, N);

cudaMemcpy(z, dev_z, M, cudaMemcpyDeviceToHost);
```

---

6. **核函数（Kernel Function）**

* 在 GPU 上并行执行的函数，每个线程都有唯一的 ID。
* 线程 ID 获取：

  * 线程块 ID：`blockIdx.x`、`blockIdx.y`、`blockIdx.z`
  * 线程 ID：`threadIdx.x`、`threadIdx.y`、`threadIdx.z`
* 核函数声明必须在全局作用域，用法：

```cpp
kernel<<<grid, block>>>(args);
```

* 核函数 **无返回值**，通过参数传递结果
* 调用后最好用 `cudaDeviceSynchronize()` 等待所有线程完成

---

7. **线程和线程块的索引计算**

* **线程块内线程索引**：

```cpp
threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y
```

* **全局线程索引**：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

---

8. **Grid 和 Block 的关系**

* **Grid（网格）**：线程块的集合
* **Block（线程块）**：线程的集合，每个线程块包含固定数量的线程

**线程数量计算示例**：

```cpp
const int N = 1000;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
```

* `threadsPerBlock`：线程块大小
* `blocksPerGrid`：向上取整，保证覆盖所有元素
* 多余线程通过 `if(idx < N)` 防止越界

---

9. **自定义函数修饰符**
   | 修饰符 | 说明 |
   |---------|------|
   | `__device__` | 在 GPU 上执行的函数 |
   | `__global__` | 核函数，在 GPU 上执行，由 CPU 调用 |
   | `__host__` | 在 CPU 上执行的函数 |
   | `__host__ __device__` | 在 CPU 或 GPU 上都可以执行的函数 |

* **区别**：

  * `__global__`：必须用 `<<<>>>` 调用，只能在 GPU 上执行
  * `__host__ __device__`：可在 CPU 或 GPU 上调用，不可用 `<<<>>>`

---

10. **内联函数（Inline Function）**

* 在编译时展开函数体，避免函数调用开销
* 适合简单的计算或转换函数
