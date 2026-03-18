# LeNet-5 手写数字识别

基于 PyTorch 实现的 LeNet-5 卷积神经网络，用于手写数字识别。

## 项目结构

```
le-net5/
├── lenet5.py          # 模型定义（BackBone + Head 模块）
├── train.py           # 训练代码
├── evaluate.py        # 评估和预测代码
├── requirements.txt   # 依赖包
├── training_img/      # 训练数据
├── checkpoints/       # 保存的模型权重（训练后生成）
└── results/          # 训练曲线图（训练后生成）
```

## 环境要求

- Python 3.6+
- PyTorch 1.8.0+
- torchvision 0.9.0+
- matplotlib
- Pillow
- tqdm

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练完成后会生成：
- `checkpoints/lenet5_best.pth` - 最佳验证集上的模型权重
- `checkpoints/lenet5_last.pth` - 最后一个 epoch 的模型权重
- `results/training_curves.png` - loss 和 accuracy 随 epoch 的变化曲线

### 2. 评估模型

```bash
python evaluate.py
```

会加载最佳模型权重并进行评估。

## 网络架构

### BackBone（卷积部分）
- **C1**: Conv2d(1, 6, 5x5) + ReLU → 输出 28x28x6
- **S2**: AvgPool2d(2x2) → 输出 14x14x6
- **C3**: Conv2d(6, 16, 5x5) + ReLU → 输出 10x10x16
- **S4**: AvgPool2d(2x2) → 输出 5x5x16

### Head（全连接部分）
- **C5**: Linear(16*5*5, 120) + ReLU
- **F6**: Linear(120, 84) + ReLU
- **Output**: Linear(84, 10)

## 特性

✅ 模块化设计：BackBone 和 Head 分离  
✅ 训练过程可视化：loss 和 accuracy 曲线  
✅ 进度条显示：使用 tqdm 显示训练进度  
✅ 模型保存：自动保存最佳模型和最新模型  
✅ 学习率调度：StepLR 自动调整学习率  

## 作者

AI 助手生成
