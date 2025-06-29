# micrograd-rs
micrograd-rs 是一个用 Rust 语言实现的简易自动求导和神经网络库，灵感来源于 [karpathy/micrograd](https://github.com/karpathy/micrograd)（感谢大师的无私奉献）。
本项目旨在使用rust语言帮助大家（自己）了解自动求导机制和简单神经网络的实现原理。
## 特性
- 自动求导: 实现了基本的自动求导功能，支持加法、减法、乘法、除法、幂运算和 ReLU 激活函数等操作。
- 神经网络组件: 包含神经元、层和多层感知机（MLP）等基本神经网络组件。
- 数据集生成: 提供了 MakeMoonDataset 数据集生成器，用于生成月牙形状的分类数据集。
- 可视化: 支持使用 plotters 库对数据集和模型预测结果进行可视化。
## 快速开始
### 环境准备
>确保你已经安装了 Rust 开发环境。如果还没有安装，可以从 Rust 官方网站 下载并安装。

克隆仓库
```bash
git clone git@github.com:578223592/micrograd-rs.git
cd micrograd-rs
```
运行示例
项目中提供了一个使用多层感知机（MLP）对月牙数据集进行分类的示例。运行以下命令来执行示例：

```bash
cargo run
```
运行成功后，会在项目根目录下生成一个名为 moon_dataset_pred.png 的图像文件，展示了模型的预测结果。
代码结构
- src/main.rs: 包含示例代码，演示了如何使用 MLP 对月牙数据集进行训练和预测，并可视化结果。
- src/lib.rs: 定义了核心的数据结构 Value，用于表示计算图中的节点，并实现了自动求导的核心逻辑。
- src/data.rs: 包含数据集生成器 MakeMoonDataset，用于生成月牙形状的分类数据集。
- src/nn.rs: 定义了神经网络的基本组件，如神经元、层和多层感知机（MLP）。
- src/math_cal.rs: 实现了 Value 类型的基本数学运算和激活函数，如加法、减法、乘法、除法、幂运算和 ReLU 激活函数。




## 贡献
如果你发现了 bug 或者有新的功能建议，欢迎提交 issue 或者 pull request。在提交 pull request 之前，请确保你的代码通过了测试，并且遵循了项目的代码风格。
## 许可证
本项目采用 MIT 许可证，具体内容请参考 LICENSE 文件。

## 注意📢
本项目的运行结果的最高精度只有88%，不知道为何达不到100%，有清楚原因的朋友欢迎指点