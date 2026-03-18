import torch
import torch.nn as nn
import torch.nn.functional as F


class BackBone(nn.Module):
    """LeNet-5 的卷积和池化部分"""
    
    def __init__(self):
        super(BackBone, self).__init__()
        
        # C1: 卷积层 1 - 输入 32x32, 输出 28x28
        self.conv1 = nn.Conv2d(
            in_channels=1,      # 单通道灰度图
            out_channels=6,     # 6 个特征图
            kernel_size=5,      # 5x5 卷积核
            stride=1,           # 步幅为 1
            padding=0           # 无填充
        )
        
        # S2: 池化层 1 - 输入 28x28, 输出 14x14
        self.pool1 = nn.AvgPool2d(
            kernel_size=2,      # 2x2 池化
            stride=2            # 步幅为 2
        )
        
        # C3: 卷积层 2 - 输入 14x14, 输出 10x10
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )
        
        # S4: 池化层 2 - 输入 10x10, 输出 5x5
        self.pool2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )
    
    def forward(self, x):
        # C1: 卷积 + 激活
        x = self.conv1(x)
        x = F.relu(x)
        
        # S2: 池化
        x = self.pool1(x)
        
        # C3: 卷积 + 激活
        x = self.conv2(x)
        x = F.relu(x)
        
        # S4: 池化
        x = self.pool2(x)
        
        return x


class Head(nn.Module):
    """LeNet-5 的全连接部分"""
    
    def __init__(self):
        super(Head, self).__init__()
        
        # C5: 全连接层 1 - 输入 16*5*5, 输出 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        # F6: 全连接层 2 - 输入 120, 输出 84
        self.fc2 = nn.Linear(120, 84)
        
        # OUTPUT: 全连接层 3 - 输入 84, 输出 10
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 展平
        x = x.view(x.size(0), -1)
        
        # C5: 全连接 + 激活
        x = self.fc1(x)
        x = F.relu(x)
        
        # F6: 全连接 + 激活
        x = self.fc2(x)
        x = F.relu(x)
        
        # OUTPUT: 全连接层
        x = self.fc3(x)
        
        return x


class LeNet5(nn.Module):
    """完整的 LeNet-5 网络"""
    
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # 卷积部分
        self.backbone = BackBone()
        
        # 全连接部分
        self.head = Head()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    # 测试网络结构
    model = LeNet5()
    print("模型结构:")
    print(model)
    
    # 测试前向传播
    test_input = torch.randn(1, 1, 32, 32)
    output = model(test_input)
    print(f"\n输入形状：{test_input.shape}")
    print(f"输出形状：{output.shape}")
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量：{total_params:,}")
