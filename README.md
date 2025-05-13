# 巴黎地铁线路识别系统

这个项目实现了一个用于识别巴黎地铁线路标志的计算机视觉系统。系统能够从图像中检测并识别不同的地铁线路标志，为视障人士提供导航辅助。

## 项目结构

```
├── config/                 # 配置文件目录
│   ├── classification/     # 分类器配置
│   ├── dataset/            # 数据集配置
│   ├── evaluation/         # 评估配置
│   ├── mode/               # 运行模式配置
│   ├── preprocessing/      # 预处理配置
│   ├── roi_detection/      # ROI检测配置
│   └── config.yaml         # 主配置文件
├── progsPython/            # 原始Python脚本
├── src/                    # 源代码目录
│   ├── classification/     # 分类器模块
│   ├── data/               # 数据处理模块
│   ├── evaluation/         # 评估模块
│   ├── pipeline/           # 处理流水线
│   ├── preprocessing/      # 预处理模块
│   ├── roi_detection/      # ROI检测模块
│   ├── training/           # 训练模块
│   └── evaluation.py       # 评估脚本
├── utils/                  # 工具函数
├── run.py                  # 主入口脚本
└── requirements.txt        # 依赖项列表
```

## 功能特点

- **模块化设计**：系统采用高度模块化的设计，各组件可以独立开发和测试
- **多种ROI检测方法**：支持基于颜色、形状和混合方法的ROI检测
- **灵活的分类器**：实现了模板匹配和CNN两种分类方法，以及它们的混合策略
- **基于Hydra的配置系统**：使用Hydra实现灵活的配置管理
- **完整的评估系统**：提供全面的评估指标和可视化工具

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/paris-metro-recognition.git
cd paris-metro-recognition
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据集：
   - 将图像数据放入`BD_METRO`目录
   - 确保`Apprentissage.mat`和`Test.mat`文件位于正确位置

## 使用方法

### 训练模式

```bash
python run.py mode=train
```

参数说明：
- `create_templates=true/false`：是否创建模板
- `train_cnn=true/false`：是否训练CNN模型

### 测试模式

```bash
python run.py mode=test
```

参数说明：
- `type=Test/Learn`：测试的数据集类型
- `view_images=true/false`：是否显示图像结果

## 自定义配置

可以通过修改`config`目录下的YAML文件或通过命令行参数来自定义配置：

```bash
python run.py mode=train dataset.val_split=0.3 training.epochs=100
```

## 评估结果

系统使用混淆矩阵、准确率、误检率等指标对识别结果进行评估。评估结果保存在`results/evaluation`目录中。 