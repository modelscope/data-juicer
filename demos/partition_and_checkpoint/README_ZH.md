# DataJuicer 容错处理与检查点和事件日志记录

本目录包含具有全面检查点、分区和事件日志记录功能的容错、可恢复 DataJuicer 处理的实现。

## 🚀 已实现功能

### ✅ 核心功能
- **作业特定目录隔离**: 每个作业都有自己专用的目录结构
- **灵活存储架构**: 事件日志（快速存储）和检查点（大容量存储）的独立存储路径
- **可配置检查点策略**: 多种检查点频率和策略
- **Spark 风格事件日志记录**: 用于可恢复性的 JSONL 格式全面事件跟踪
- **作业恢复功能**: 从最后一个检查点恢复失败或中断的作业
- **全面作业管理**: 作业摘要、元数据跟踪和恢复命令

### ✅ 检查点策略
- `EVERY_OP`: 每个操作后检查点（最容错，较慢）
- `EVERY_PARTITION`: 仅在分区完成时检查点（平衡）
- `EVERY_N_OPS`: 每 N 个操作后检查点（可配置）
- `MANUAL`: 仅在指定操作后检查点
- `DISABLED`: 完全禁用检查点

### ✅ 事件日志记录
- **人类可读日志**: 基于 Loguru 的日志记录，用于调试和监控
- **机器可读日志**: JSONL 格式，用于程序化分析和恢复
- **全面事件类型**: 作业开始/完成/失败、分区事件、操作事件、检查点事件
- **实时监控**: 实时事件流和状态报告

### ✅ 作业管理
- **有意义的作业 ID**: 格式：`{YYYYMMDD}_{HHMMSS}_{config_name}_{unique_suffix}`
- **作业摘要文件**: 每个作业运行的全面元数据
- **恢复命令**: 自动生成恢复作业的确切命令
- **作业验证**: 验证作业恢复参数和现有状态

## 📁 目录结构

```
{work_dir}/
├── {job_id}/                    # 作业特定目录
│   ├── job_summary.json         # 作业元数据和恢复信息
│   ├── metadata/                # 作业元数据文件
│   │   ├── dataset_mapping.json
│   │   └── final_mapping_report.json
│   ├── partitions/              # 输入数据分区
│   ├── intermediate/            # 中间处理结果
│   └── results/                 # 最终处理结果
├── {event_log_dir}/{job_id}/    # 灵活事件日志存储
│   └── event_logs/
│       ├── events.jsonl         # 机器可读事件
│       └── events.log           # 人类可读日志
└── {checkpoint_dir}/{job_id}/   # 灵活检查点存储
    ├── checkpoint_*.json        # 检查点元数据
    └── partition_*_*.parquet    # 分区检查点
```

## 🛠️ 配置

### 配置结构

配置使用**逻辑嵌套结构**，按关注点分组相关设置：

#### 新的逻辑结构（推荐）
```yaml
# 分区配置
partition:
  size: 1000  # 每个分区的样本数
  max_size_mb: 64  # 分区最大大小（MB）

# 容错配置
fault_tolerance:
  enabled: true
  max_retries: 3
  retry_backoff: "exponential"  # exponential, linear, fixed

# 中间存储配置（格式、压缩和生命周期管理）
intermediate_storage:
  # 文件格式和压缩
  format: "parquet"  # parquet, arrow, jsonl
  compression: "snappy"  # snappy, gzip, none
  use_arrow_batches: true
  arrow_batch_size: 500
  arrow_memory_mapping: false
  
  # 文件生命周期管理
  preserve_intermediate_data: true  # 保留临时文件用于调试/恢复
  cleanup_temp_files: true
  cleanup_on_success: false
  retention_policy: "keep_all"  # keep_all, keep_failed_only, cleanup_all
  max_retention_days: 7
```

#### 传统扁平结构（仍支持）
```yaml
# 传统扁平配置（仍有效）
partition_size: 1000
max_partition_size_mb: 64
enable_fault_tolerance: true
max_retries: 3
preserve_intermediate_data: true
storage_format: "parquet"
use_arrow_batches: true
arrow_batch_size: 500
arrow_memory_mapping: false
```

**注意**: 系统首先从新的嵌套部分读取，如果未找到则回退到传统扁平配置。

### 配置部分说明

#### `partition` - 分区和容错
控制数据集如何分割以及如何处理故障：
- **自动配置**（推荐）：
  - `auto_configure`: 根据数据模态启用自动分区大小优化
- **手动分区**（当 `auto_configure: false` 时）：
  - `size`: 每个分区的样本数
    - **50-100**: 调试、快速迭代、小数据集
    - **100-300**: 生产、容错和效率的良好平衡 ⭐
    - **300-500**: 具有稳定处理的大数据集
    - **500+**: 仅适用于故障风险最小的大数据集
  - `max_size_mb`: 分区最大大小（MB）
- **容错**：
  - `enable_fault_tolerance`: 启用/禁用重试逻辑
  - `max_retries`: 每个分区的最大重试次数
  - `retry_backoff`: 重试策略（`exponential`、`linear`、`fixed`）

#### `intermediate_storage` - 中间数据管理
控制中间数据的文件格式、压缩和生命周期管理：
- **文件格式和压缩**：
  - `format`: 存储格式（`parquet`、`arrow`、`jsonl`）
  - `compression`: 压缩算法（`snappy`、`gzip`、`none`）
  - `use_arrow_batches`: 使用 Arrow 批处理
  - `arrow_batch_size`: Arrow 批大小
  - `arrow_memory_mapping`: 启用内存映射
- **文件生命周期管理**：
  - `preserve_intermediate_data`: 保留临时文件用于调试
  - `cleanup_temp_files`: 启用自动清理
  - `cleanup_on_success`: 即使成功完成也清理
  - `retention_policy`: 文件保留策略（`keep_all`、`keep_failed_only`、`cleanup_all`）
  - `max_retention_days`: X 天后自动清理

### 基本配置
```yaml
# 启用容错处理
executor_type: ray_partitioned

# 作业管理
job_id: my_experiment_001  # 可选：如果未提供则自动生成

# 检查点配置
checkpoint:
  enabled: true
  strategy: every_op  # every_op, every_partition, every_n_ops, manual, disabled
  n_ops: 2            # 用于 every_n_ops 策略
  op_names:           # 用于 manual 策略
    - clean_links_mapper
    - whitespace_normalization_mapper

# 事件日志记录配置
event_logging:
  enabled: true
  max_log_size_mb: 100
  backup_count: 5

# 灵活存储路径
event_log_dir: /tmp/fast_event_logs      # 事件日志的快速存储
checkpoint_dir: /tmp/large_checkpoints   # 检查点的大容量存储

# 分区配置
partition:
  # 基本分区设置
  # 推荐分区大小：
  # - 50-100: 用于调试、快速迭代、小数据集
  # - 100-300: 用于生产、容错和效率的良好平衡
  # - 300-500: 用于具有稳定处理的大数据集
  # - 500+: 仅适用于故障风险最小的大数据集
  size: 200  # 每个分区的样本数（较小以获得更好的容错性）
  max_size_mb: 32  # 分区最大大小（MB）（减少以加快处理速度）
  
  # 容错设置
  enable_fault_tolerance: true
  max_retries: 3
  retry_backoff: "exponential"  # exponential, linear, fixed

# 中间存储配置（格式、压缩和生命周期管理）
intermediate_storage:
  # 文件格式和压缩
  format: "parquet"  # parquet, arrow, jsonl
  compression: "snappy"  # snappy, gzip, none
  use_arrow_batches: true
  arrow_batch_size: 500
  arrow_memory_mapping: false
  
  # 文件生命周期管理
  preserve_intermediate_data: true  # 保留临时文件用于调试/恢复
  cleanup_temp_files: true
  cleanup_on_success: false
  retention_policy: "keep_all"  # keep_all, keep_failed_only, cleanup_all
  max_retention_days: 7
```

## 🚀 快速开始

### 1. 基本用法
```bash
# 使用自动生成的作业 ID 运行
dj-process --config configs/demo/checkpoint_config_example.yaml

# 使用自定义作业 ID 运行
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id my_experiment_001
```

### 2. 恢复作业
```bash
# 使用作业 ID 恢复
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id my_experiment_001
```

### 3. 不同的检查点策略
```bash
# 每个分区检查点
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id partition_test --checkpoint.strategy every_partition

# 每 3 个操作检查点
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id n_ops_test --checkpoint.strategy every_n_ops --checkpoint.n_ops 3

# 手动检查点
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id manual_test --checkpoint.strategy manual --checkpoint.op_names clean_links_mapper,whitespace_normalization_mapper
```

### 4. 运行综合演示
```bash
# 运行展示所有功能的完整演示
python demos/partition_and_checkpoint/run_comprehensive_demo.py
```

## 📊 监控和调试

### 查看作业信息
```bash
# 检查作业摘要
cat ./outputs/demo-checkpoint-strategies/{job_id}/job_summary.json

# 查看事件日志
cat /tmp/fast_event_logs/{job_id}/event_logs/events.jsonl

# 查看人类可读日志
cat /tmp/fast_event_logs/{job_id}/event_logs/events.log
```

### 列出可用作业
```bash
# 列出所有作业目录
ls -la ./outputs/demo-checkpoint-strategies/
```

### 检查灵活存储
```bash
# 检查快速存储中的事件日志
ls -la /tmp/fast_event_logs/

# 检查大容量存储中的检查点
ls -la /tmp/large_checkpoints/
```

## 📈 作业管理工具

DataJuicer 提供全面的作业管理工具，用于监控进度和停止正在运行的作业。这些工具位于 `data_juicer/utils/job/` 中，提供命令行和程序化接口。

### 📊 作业进度监控器

一个全面的工具，用于监控和显示 DataJuicer 作业的进度信息。显示分区状态、操作进度、检查点和整体作业指标。

#### 功能特性

- **实时进度跟踪**: 监控具有分区级详细信息的作业进度
- **操作性能**: 查看详细的操作指标，包括吞吐量和数据减少
- **检查点监控**: 跟踪检查点保存和恢复点
- **监视模式**: 连续监控作业，自动更新
- **程序化访问**: 作为 Python 函数使用，集成到其他工具中

#### 命令行用法

##### 基本用法
```bash
# 显示作业的基本进度
python -m data_juicer.utils.job.monitor 20250728_233517_510abf

# 显示详细进度和操作指标
python -m data_juicer.utils.job.monitor 20250728_233517_510abf --detailed

# 监视模式 - 每 10 秒连续更新进度
python -m data_juicer.utils.job.monitor 20250728_233517_510abf --watch

# 监视模式，自定义更新间隔（30 秒）
python -m data_juicer.utils.job.monitor 20250728_233517_510abf --watch --interval 30

# 使用自定义基础目录
python -m data_juicer.utils.job.monitor 20250728_233517_510abf --base-dir /custom/path
```

##### 命令行选项
- `job_id`: 要监控的作业 ID（必需）
- `--base-dir`: 包含作业输出的基础目录（默认：`outputs/partition-checkpoint-eventlog`）
- `--detailed`: 显示详细的操作信息
- `--watch`: 监视模式 - 连续更新进度
- `--interval`: 监视模式的更新间隔（秒）（默认：10）

#### Python API

##### 基本函数用法
```python
from data_juicer.utils.job.monitor import show_job_progress

# 显示进度并获取数据
data = show_job_progress("20250728_233517_510abf")

# 显示详细进度
data = show_job_progress("20250728_233517_510abf", detailed=True)

# 使用自定义基础目录
data = show_job_progress("20250728_233517_510abf", base_dir="/custom/path")
```

##### 基于类的用法
```python
from data_juicer.utils.job.monitor import JobProgressMonitor

# 创建监控器实例
monitor = JobProgressMonitor("20250728_233517_510abf")

# 显示进度
monitor.display_progress(detailed=True)

# 获取进度数据作为字典
data = monitor.get_progress_data()

# 访问特定信息
job_status = data['overall_progress']['job_status']
progress_percentage = data['overall_progress']['progress_percentage']
partition_status = data['partition_status']
```

### 🛑 作业停止器

一个工具，通过读取事件日志来查找进程和线程 ID，然后终止这些特定的进程和线程来停止正在运行的 DataJuicer 作业。

#### 功能特性

- **精确进程终止**: 使用事件日志识别要终止的确切进程和线程
- **优雅关闭**: 首先发送 SIGTERM 进行优雅关闭，然后在需要时发送 SIGKILL
- **安全检查**: 在停止前验证作业存在性和运行状态
- **全面日志记录**: 终止过程的详细日志记录
- **程序化访问**: 可以作为 Python 函数或命令行工具使用

#### 命令行用法

##### 基本用法
```bash
# 优雅地停止作业（SIGTERM）
python -m data_juicer.utils.job.stopper 20250728_233517_510abf

# 强制停止作业（SIGKILL）
python -m data_juicer.utils.job.stopper 20250728_233517_510abf --force

# 使用自定义超时停止（60 秒）
python -m data_juicer.utils.job.stopper 20250728_233517_510abf --timeout 60

# 使用自定义基础目录
python -m data_juicer.utils.job.stopper 20250728_233517_510abf --base-dir /custom/path

# 列出所有正在运行的作业
python -m data_juicer.utils.job.stopper --list
```

##### 命令行选项
- `job_id`: 要停止的作业 ID（必需，除非使用 --list）
- `--base-dir`: 包含作业输出的基础目录（默认：`outputs/partition-checkpoint-eventlog`）
- `--force`: 使用 SIGKILL 强制杀死而不是优雅的 SIGTERM
- `--timeout`: 优雅关闭的超时时间（秒）（默认：30）
- `--list`: 列出所有正在运行的作业而不是停止一个

#### Python API

##### 基本函数用法
```python
from data_juicer.utils.job.stopper import stop_job

# 优雅地停止作业
result = stop_job("20250728_233517_510abf")

# 强制停止作业
result = stop_job("20250728_233517_510abf", force=True)

# 使用自定义超时停止
result = stop_job("20250728_233517_510abf", timeout=60)

# 使用自定义基础目录
result = stop_job("20250728_233517_510abf", base_dir="/custom/path")
```

##### 基于类的用法
```python
from data_juicer.utils.job.stopper import JobStopper

# 创建停止器实例
stopper = JobStopper("20250728_233517_510abf")

# 停止作业
result = stopper.stop_job(force=False, timeout=30)

# 检查作业是否正在运行
is_running = stopper.is_job_running()

# 获取作业摘要
summary = stopper.get_job_summary()
```

### 🔧 通用工具

监控器和停止器工具都通过 `data_juicer.utils.job.common` 共享通用功能：

```python
from data_juicer.utils.job.common import JobUtils, list_running_jobs

# 列出所有正在运行的作业
running_jobs = list_running_jobs()

# 创建作业工具实例
job_utils = JobUtils("20250728_233517_510abf")

# 加载作业摘要
summary = job_utils.load_job_summary()

# 加载事件日志
events = job_utils.load_event_logs()

# 获取分区状态
partition_status = job_utils.get_partition_status()
```

### 输出信息

#### 作业概览
- 作业状态（已完成、处理中、失败等）
- 数据集路径和大小
- 分区配置
- 开始时间和持续时间

#### 整体进度
- 进度百分比
- 分区完成状态
- 样本处理计数
- 估计剩余时间（对于运行中的作业）

#### 分区状态
- 带有视觉指示器的单个分区状态
- 每个分区的样本计数
- 当前操作（如果正在处理）
- 已完成操作的数量
- 已保存检查点的数量

#### 操作详情（使用 --detailed 标志）
- 每个分区的操作性能
- 持续时间、吞吐量和数据减少指标
- 操作完成顺序

#### 检查点摘要
- 已保存检查点的总数
- 按分区和操作的检查点详情
- 时间戳信息

### 示例输出

```
================================================================================
DataJuicer 作业进度监控器
作业 ID: 20250728_233517_510abf
================================================================================

📊 作业概览
   状态: 已完成
   数据集: /Users/yilei.z/Downloads/c4-train.00000-of-01024.jsonl
   总样本数: 356,317
   分区大小: 50,000 样本
   开始时间: 2025-07-28 16:35:18
   持续时间: 441.1 秒

🎯 整体进度
   进度: 100.0% (8/8 分区)
   状态: 8 已完成, 0 处理中, 0 失败
   样本: 356,317/356,317

📦 分区状态
   分区  0: ✅ 已完成
              样本: 44,539
              已完成: 8 个操作
              检查点: 2 个已保存
   分区  1: ✅ 已完成
              样本: 44,540
              已完成: 8 个操作
              检查点: 2 个已保存
   ...

💾 检查点摘要
   总检查点: 16
```

### 集成示例

#### 监控多个作业
```python
from data_juicer.utils.job.monitor import show_job_progress

job_ids = ["job1", "job2", "job3"]
for job_id in job_ids:
    try:
        data = show_job_progress(job_id)
        print(f"作业 {job_id}: {data['overall_progress']['progress_percentage']:.1f}%")
    except FileNotFoundError:
        print(f"作业 {job_id}: 未找到")
```

#### 自定义监控脚本
```python
from data_juicer.utils.job.monitor import JobProgressMonitor
import time

def monitor_job_until_completion(job_id, check_interval=30):
    monitor = JobProgressMonitor(job_id)
    
    while True:
        data = monitor.get_progress_data()
        status = data['overall_progress']['job_status']
        
        if status == 'completed':
            print(f"作业 {job_id} 已完成！")
            break
        elif status == 'failed':
            print(f"作业 {job_id} 失败！")
            break
        
        print(f"作业 {job_id} 仍在运行... {data['overall_progress']['progress_percentage']:.1f}%")
        time.sleep(check_interval)
```

#### 作业管理工作流
```python
from data_juicer.utils.job.monitor import show_job_progress
from data_juicer.utils.job.stopper import stop_job
from data_juicer.utils.job.common import list_running_jobs

# 列出所有正在运行的作业
running_jobs = list_running_jobs()
print(f"发现 {len(running_jobs)} 个正在运行的作业")

# 监控并可能停止作业
for job_info in running_jobs:
    job_id = job_info['job_id']
    
    # 检查进度
    try:
        data = show_job_progress(job_id)
        progress = data['overall_progress']['progress_percentage']
        
        # 停止卡住的作业（1小时后进度仍少于10%）
        if progress < 10 and data['overall_progress']['elapsed_time_seconds'] > 3600:
            print(f"停止卡住的作业 {job_id}（进度: {progress:.1f}%）")
            stop_job(job_id, force=True)
        else:
            print(f"作业 {job_id}: {progress:.1f}% 完成")
            
    except Exception as e:
        print(f"监控作业 {job_id} 时出错: {e}")
```

## 🤖 自动配置系统

### **按模态智能分区大小调整**

DataJuicer 现在包含一个智能自动配置系统，可以根据您的数据特征自动确定最佳分区大小：

#### **工作原理**

1. **模态检测**: 分析您的数据集以检测主要模态（文本、图像、音频、视频、多模态）
2. **数据集分析**: 检查样本特征（文本长度、媒体数量、文件大小）
3. **管道复杂性**: 考虑处理操作的复杂性
4. **资源优化**: 调整分区大小以获得最佳内存使用和容错性

#### **模态特定优化**

| 模态 | 默认大小 | 最大大小 | 内存倍数 | 使用场景 |
|------|----------|----------|----------|----------|
| **文本** | 200 样本 | 1000 | 1.0x | 高效处理，低内存 |
| **图像** | 50 样本 | 200 | 5.0x | 中等内存，图像处理 |
| **音频** | 30 样本 | 100 | 8.0x | 高内存，音频处理 |
| **视频** | 10 样本 | 50 | 20.0x | 极高内存，复杂处理 |
| **多模态** | 20 样本 | 100 | 10.0x | 多种模态，中等复杂性 |

#### **启用自动配置**

```yaml
partition:
  auto_configure: true  # 启用自动优化
  # 当 auto_configure 为 true 时忽略手动设置
  size: 200
  max_size_mb: 32
```

#### **手动覆盖**

```yaml
partition:
  auto_configure: false  # 禁用自动配置
  size: 100  # 使用您自己的分区大小
  max_size_mb: 64
```

## 📊 分区大小指南

### **为什么较小的分区更好**

**容错性**: 较小的分区意味着较小的故障单元。如果分区失败，您损失的工作更少。

**恢复速度**: 失败的分区可以更快地重试，减少总体作业时间。

**进度可见性**: 更细粒度的进度跟踪和更快的反馈。

**内存效率**: 每个分区更低的内存使用，更适合资源受限的环境。

**调试**: 更容易隔离和调试较小块中的问题。

### **分区大小建议**

| 使用场景 | 分区大小 | 何时使用 |
|----------|----------|----------|
| **调试** | 50-100 样本 | 快速迭代、测试、小数据集 |
| **生产** ⭐ | 100-300 样本 | 大多数用例，良好平衡 |
| **大数据集** | 300-500 样本 | 稳定处理，大数据集 |
| **超大** | 500+ 样本 | 仅在故障风险最小时 |

### **需要考虑的因素**

- **数据集大小**: 较大的数据集可以使用较大的分区
- **处理复杂性**: 复杂操作受益于较小的分区
- **故障率**: 较高的故障率需要较小的分区
- **内存约束**: 有限的内存需要较小的分区
- **时间敏感性**: 更快的反馈需要较小的分区

## 🔧 实现细节

### 核心组件

1. **`EventLoggingMixin`** (`data_juicer/core/executor/event_logging_mixin.py`)
   - 为执行器提供事件日志记录功能
   - 管理作业特定目录和灵活存储
   - 处理作业摘要创建和验证
   - 实现 Spark 风格事件日志记录模式

2. **`PartitionedRayExecutor`** (`data_juicer/core/executor/ray_executor_partitioned.py`)
   - 使用分区和容错扩展 Ray 执行器
   - 实现可配置检查点策略
   - 与 EventLoggingMixin 集成以进行全面日志记录
   - 处理从检查点恢复作业

3. **配置集成** (`data_juicer/config/config.py`)
   - 添加了作业管理的命令行参数
   - 添加了检查点配置选项
   - 添加了灵活存储路径配置

### 事件类型
- `JOB_START`, `JOB_COMPLETE`, `JOB_FAILED`
- `PARTITION_START`, `PARTITION_COMPLETE`, `PARTITION_FAILED`
- `OP_START`, `OP_COMPLETE`, `OP_FAILED`
- `CHECKPOINT_SAVE`, `CHECKPOINT_LOAD`
- `PROCESSING_START`, `PROCESSING_COMPLETE`, `PROCESSING_ERROR`
- `RESOURCE_USAGE`, `PERFORMANCE_METRIC`
- `WARNING`, `INFO`, `DEBUG`

## 🎯 使用场景

### 1. 大数据集处理
- 处理对于内存来说太大的数据集
- 具有容错的自动分区
- 故障后恢复处理

### 2. 实验工作流
- 使用有意义的作业 ID 跟踪不同实验
- 比较不同配置的结果
- 维护实验历史和可重现性

### 3. 生产管道
- 强大的错误处理和恢复
- 全面监控和日志记录
- 不同性能要求的灵活存储

### 4. 研究和开发
- 具有检查点恢复的迭代开发
- 用于分析的详细事件日志记录
- 不同场景的可配置检查点

## 🔍 故障排除

### 常见问题

1. **作业恢复失败**
   - 检查作业摘要是否存在：`ls -la ./outputs/{work_dir}/{job_id}/job_summary.json`
   - 验证检查点文件是否存在：`ls -la /tmp/large_checkpoints/{job_id}/`

2. **找不到事件日志**
   - 检查灵活存储路径：`ls -la /tmp/fast_event_logs/{job_id}/`
   - 验证配置中是否启用了事件日志记录

3. **检查点不工作**
   - 验证配置中的检查点策略
   - 检查检查点目录是否可写
   - 确保 checkpoint.enabled 为 true

4. **性能问题**
   - 根据可用内存调整分区大小
   - 考虑不同的检查点策略
   - 使用适当的存储格式（大数据集使用 parquet）

### 调试命令
```bash
# 检查 Ray 集群状态
ray status

# 查看 Ray 仪表板
open http://localhost:8265

# 检查 DataJuicer 日志
tail -f /tmp/fast_event_logs/{job_id}/event_logs/events.log
```

## 📊 理解中间数据

### 什么是中间数据？

中间数据是指在处理管道期间生成的临时结果，存在于操作之间和最终输出之前。在 DataJuicer 的分区处理中，这包括：

1. **分区级中间数据**: 分区内每个操作后的结果
2. **操作级中间数据**: 操作之间存在的数据（例如，在 `clean_links_mapper` 之后但在 `whitespace_normalization_mapper` 之前）
3. **检查点中间数据**: 检查点期间创建的临时文件

### 何时保留中间数据

**当您需要以下功能时启用 `preserve_intermediate_data: true`：**
- **调试**: 检查每个操作后数据的样貌
- **恢复**: 如果作业失败，查看确切失败位置和数据样貌
- **分析**: 了解每个操作如何转换数据
- **开发**: 通过详细检查迭代处理管道

**当您想要以下功能时禁用 `preserve_intermediate_data: false`：**
- **性能**: 更快的处理，更少的磁盘 I/O
- **存储效率**: 减少磁盘空间使用
- **生产**: 无临时文件累积的清洁处理

### 带有中间数据的目录结构示例

```
{job_dir}/intermediate/
├── partition_000000/
│   ├── op_000_clean_links_mapper.parquet      # clean_links_mapper 之后
│   ├── op_001_clean_email_mapper.parquet      # clean_email_mapper 之后
│   ├── op_002_whitespace_normalization_mapper.parquet
│   └── op_003_fix_unicode_mapper.parquet      # fix_unicode_mapper 之后
└── partition_000001/
    ├── op_000_clean_links_mapper.parquet
    └── ...
```

## 📈 性能考虑

### 检查点开销
- `EVERY_OP`: 最高开销，最大容错性
- `EVERY_PARTITION`: 平衡的开销和容错性
- `EVERY_N_OPS`: 可配置开销
- `MANUAL`: 最小开销，需要仔细规划

### 存储建议
- **事件日志**: 使用快速存储（SSD）进行实时监控
- **检查点**: 使用大容量存储（HDD/网络存储）以提高成本效率
- **分区**: 使用本地存储以提高处理速度

### 内存管理
- 根据可用内存调整 `partition_size`
- 使用 `max_partition_size_mb` 限制分区大小
- 考虑 `preserve_intermediate_data` 用于调试与性能

## 🎉 成功指标

实现成功展示了：
- ✅ **容错性**: 作业可以在故障后恢复
- ✅ **可扩展性**: 通过分区处理大数据集
- ✅ **可观察性**: 全面日志记录和监控
- ✅ **灵活性**: 可配置检查点和存储
- ✅ **可用性**: 具有有意义的作业 ID 的简单命令行界面
- ✅ **性能**: 从检查点快速恢复
- ✅ **可靠性**: 强大的错误处理和验证

## 🔮 未来增强

未来开发的潜在领域：
- **分布式检查点**: 多节点检查点协调
- **增量检查点**: 仅保存更改的数据
- **检查点压缩**: 减少存储要求
- **高级监控**: 用于作业监控的基于 Web 的仪表板
- **检查点版本控制**: 支持多个检查点版本
- **与外部系统集成**: 云存储、监控系统 