# LeNet-5 环境配置指南

## 方法一：使用 Conda 创建环境（推荐）

### 1. 创建 conda 环境
```bash
conda create -n lenet5 python=3.8 -y
```

### 2. 激活环境
```bash
conda activate lenet5
```

### 3. 安装 PyTorch（根据系统选择）

#### Windows CPU 版本
```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

#### Windows NVIDIA GPU 版本（推荐，如果有显卡）
```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y
```

### 4. 安装其他依赖
```bash
pip install -r requirements.txt
```

### 5. 验证环境
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 方法二：完整的 conda 环境配置

### 1. 创建并配置环境（一步到位）
```bash
# 创建环境
conda create -n lenet5 python=3.8 -y

# 激活环境
conda activate lenet5

# 使用 conda 安装主要依赖
conda install pytorch torchvision matplotlib pillow tqdm -c pytorch -y
```

### 2. 验证安装
运行测试脚本：
```bash
python lenet5.py
```

## 环境使用说明

### 激活环境
每次使用前需要激活环境：
```bash
conda activate lenet5
```

### 退出环境
```bash
conda deactivate
```

### 查看已安装的环境
```bash
conda env list
```

### 导出环境配置（可选）
```bash
conda env export > environment.yml
```

### 从 environment.yml 创建环境（可选）
```bash
conda env create -f environment.yml
```

## 开始训练

环境配置完成后：

```bash
# 1. 激活环境
conda activate lenet5

# 2. 开始训练
python train.py
```

## 常见问题

### 1. conda 命令不可用
确保已正确安装 Anaconda 或 Miniconda，并添加到系统 PATH。

### 2. PyTorch 安装失败
访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适合您系统的安装命令。

### 3. CUDA 版本不匹配
如果使用 GPU 版本，确保 NVIDIA 驱动版本与 CUDA 版本匹配。
