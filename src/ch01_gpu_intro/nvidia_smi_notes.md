# `nvidia-smi` 使用笔记

`nvidia-smi`（NVIDIA System Management Interface）是 NVIDIA 提供的命令行工具，用于查询 GPU 信息、监控状态和管理 GPU 资源。

---

## 1. 基本用法

```bash
nvidia-smi
```

显示当前 GPU 状态，包括：

* GPU 型号与序列号
* 驱动版本
* 显存使用情况（Total / Used / Free）
* GPU 利用率（%）
* 当前运行的进程信息

---

## 2. 实时刷新监控

```bash
nvidia-smi -l 5
```

* `-l <seconds>`：每隔指定秒数刷新一次
* 常用于实时观察 GPU 利用率和显存占用

---

## 3. 查看驱动和 GPU 信息

```bash
nvidia-smi -q
```

* `-q`：显示详细信息（包括温度、电源、PCIe、计算能力等）

```bash
nvidia-smi -q -d MEMORY,UTILIZATION,TEMPERATURE
```

* `-d`：指定显示特定类别的信息，例如 MEMORY（显存）、UTILIZATION（使用率）、TEMPERATURE（温度）

---

## 4. 显示 GPU 设备列表

```bash
nvidia-smi -L
```

* 列出所有可用 GPU 及其索引，例如：

```
GPU 0: A100-SXM4-40GB (UUID: GPU-xxxxxxx)
GPU 1: A100-SXM4-40GB (UUID: GPU-yyyyyyy)
```

---

## 5. 显示指定 GPU 信息

```bash
nvidia-smi -i 0
```

* `-i <GPU_ID>`：指定 GPU 查看状态
* 可结合 `-q` 使用，例如：

```bash
nvidia-smi -i 0 -q
```

---

## 6. 查询 GPU 使用历史（利用 NVIDIA Persistence Mode）

```bash
nvidia-smi --query-gpu=name,index,driver_version,memory.total,memory.used,utilization.gpu --format=csv
```

* `--query-gpu=<fields>`：自定义查询字段
* `--format=csv|table|json`：输出格式
* 常用字段：

  * `name`：GPU 型号
  * `index`：GPU 序号
  * `driver_version`：驱动版本
  * `memory.total` / `memory.used` / `memory.free`
  * `utilization.gpu` / `utilization.memory`

---

## 7. 导出信息到文件

```bash
nvidia-smi -q -x > gpu_info.xml
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv > gpu_info.csv
```

* `-x`：XML 格式导出
* `--format=csv`：CSV 格式导出，方便做数据统计或脚本处理

---

## 8. 控制 GPU 功能（可选）

```bash
nvidia-smi -pm 1
```

* `-pm 1/0`：开启或关闭 Persistence Mode（保持 GPU 驱动常驻，减少初始化延迟）

```bash
nvidia-smi -pl 250
```

* `-pl <watts>`：设置 GPU 功率上限

---

## 9. 查看运行的进程

```bash
nvidia-smi pmon -i 0
```

* `pmon`：实时显示 GPU 上运行的进程及显存占用
* `-c <cycle>`：刷新周期（秒）

---

## 10. 常用组合命令示例

* 实时每 2 秒刷新 GPU 利用率：

```bash
watch -n 2 nvidia-smi
```

* 查询所有 GPU 的显存使用情况：

```bash
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=table
```

* 导出为 CSV 并供脚本解析：

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > gpu_log.csv
```

---

| 分类         | 参数                         | 功能描述                              |
| ---------- | -------------------------- | --------------------------------- |
| **信息查询**   | `nvidia-smi -q`            | 显示所有 GPU 的详细信息（查询模式）              |
|            | `nvidia-smi -L`            | 以简洁格式列出所有 GPU（列表模式）               |
|            | `nvidia-smi -i <gpu_id>`   | 仅针对指定 ID 的 GPU 进行操作               |
| **监控与格式**  | `nvidia-smi -l <秒数>`       | 循环查询模式，每隔指定秒数更新一次                 |
|            | `nvidia-smi -lms <毫秒数>`    | 以毫秒为单位进行循环查询（高频率）                 |
|            | `nvidia-smi -f <文件名>`      | 将输出重定向到文件                         |
|            | `--format=csv`             | 以 CSV 格式输出，便于脚本处理                 |
|            | `-x` 或 `--xml-format`      | 以 XML 格式输出，便于程序解析                 |
| **功耗与时钟**  | `--persistence-mode=<0/1>` | 禁用/启用持久化模式（影响功耗和响应）               |
|            | `-pl <功耗值>`                | 设置 GPU 的功耗限制（单位：瓦）                |
|            | `-rgc`                     | 重置 GPU 的时钟设置（恢复默认）                |
| **性能与计算**  | `-ac`                      | 设置应用时钟（需要权限）                      |
|            | `-dmp`                     | 转储性能数据（用于调试）                      |
|            | `--compute-mode=<MODE>`    | 设置计算模式（如默认、独占进程等）                 |
| **进程与显存**  | `-p` 或 `--display=POLICY`  | 控制显存/进程信息的显示（如 MEMORY, PROCESSES） |
|            | `--gpu-reset`              | 重置 GPU（在驱动无响应时使用）                 |
| **自动化与日志** | `--help-query-gpu`         | 列出可与 `--query-gpu` 一起使用的所有可查询字段   |
|            | `--query-gpu=...`          | 自定义查询特定指标（用于自动化脚本）                |


