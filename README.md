# DBS Dither

**Direct Binary Search (DBS) Dithering Tool** based on Rust.

这是一个高性能的直接二值搜索 (DBS) 抖动算法实现。DBS 是一种高质量的半色调（Halftoning）算法，它通过迭代优化来最小化半色调图像与原始图像之间基于人类视觉系统 (HVS) 模型的感知误差。

## ✨ 特性

- **高质量抖动**：使用 DBS 算法生成视觉上极佳的二值图像。
- **色彩支持**：支持单色（黑白）和全彩（RGB）抖动处理。
- **高性能**：
  - 使用 `rayon` 进行多线程并行处理。
  - 核心计算采用定点数 (Fixed-point arithmetic) 优化。
- **交互式 CLI**：简单易用的命令行界面，支持图片缩放。
- **可配置 HVS**：通过配置文件调整视觉模型的参数 (Sigma, Kernel Size)。

## 🚀 快速开始

### 环境要求

- [Rust Toolchain](https://www.rust-lang.org/tools/install) (推荐最新稳定版)

### 安装与构建

1. 克隆仓库：
   ```bash
   git clone https://github.com/YourUsername/dbs_dither.git
   cd dbs_dither
   ```

2. 编译（推荐使用 Release 模式以获得最佳性能）：
   ```bash
   cargo build --release
   ```

### 使用方法

直接运行程序：

```bash
cargo run --release
```

程序会交互式地引导你完成以下步骤：

1. **输入图片路径**：输入你想要处理的图片文件的路径（支持拖拽文件到终端）。
2. **输入输出高度**（可选）：输入想要调整的高度（保持长宽比）。直接回车则保持原始尺寸。
3. **选择模式**：输入 `y` 选择单色处理，输入 `n` 选择彩色处理。

处理完成后，结果将保存为 `[原文件名]_dbs.png`。

### 示例

![Example Output](IMG_3293_dbs.png)
*(注意：请确保项目根目录下有示例图片，或者替换此占位符)*

## ⚙️ 配置

项目根目录下的 `config.yaml` 文件用于配置 HVS 模型参数：

```yaml
hvs_sigma: 1.0        # 高斯核的标准差，控制模糊程度/视觉敏感度
hvs_kernel_size: 27   # HVS 核的大小（建议为奇数），越大计算越慢但可能效果更好
```

## 📝 算法简介

DBS 算法通过评估每一个像素的改变（翻转像素值或与邻域交换）对整体感知误差的影响。它是一个迭代过程：
1. **初始化**：生成初始半色调图像。
2. **迭代**：遍历所有像素，尝试"翻转"（Toggle）或"交换"（Swap）。如果操作能降低感知误差，则应用该操作。
3. **收敛**：当一次完整的迭代中没有任何像素发生改变时，算法停止。

本项目使用了分块并行策略来加速这一过程。

## 许可证

[MIT License](LICENSE)
