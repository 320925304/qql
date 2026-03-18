import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from lenet5 import LeNet5
from tqdm import tqdm
import os


def get_transform():
    """获取数据预处理"""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def load_data(data_dir, batch_size=64):
    """加载数据集"""
    transform = get_transform()
    
    # 从文件夹加载数据
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    
    # 划分训练集和验证集 (80% 训练，20% 验证)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    

    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"训练集大小：{len(train_dataset)}")
    print(f"验证集大小：{len(val_dataset)}")
    print(f"类别数：{len(dataset.classes)}")
    print(f"类别名称：{dataset.classes}")
    
    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train(model, train_loader, val_loader, epochs=20, lr=0.001, save_dir='checkpoints'):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 创建保存目录
    Path(save_dir).mkdir(exist_ok=True)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_acc = 0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        print('-' * 50)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # 学习率调整
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'lenet5_best.pth'))
            print(f'保存最佳模型 (验证准确率：{val_acc:.2f}%)')
        
        # 保存最后一个 epoch 的模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, os.path.join(save_dir, 'lenet5_last.pth'))
    
    print(f'\n训练完成！最佳验证准确率：{best_acc:.2f}%')
    
    return history


def plot_history(history, save_dir='results'):
    """绘制训练曲线"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制 Loss 曲线
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制 Accuracy 曲线
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    print(f'训练曲线已保存到：{os.path.join(save_dir, "training_curves.png")}')
    
    # 显示图表
    plt.show()


def main():
    """主函数"""
    # 配置参数
    data_dir = 'training_img'  # 数据目录
    batch_size = 64
    epochs = 10
    lr = 0.001
    save_dir = 'checkpoints'
    results_dir = 'results'
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader = load_data(data_dir, batch_size)
    
    # 创建模型
    print("\n创建模型...")
    model = LeNet5()
    print(model)
    
    # 训练
    print("\n开始训练...")
    history = train(
        model, 
        train_loader, 
        val_loader, 
        epochs=epochs, 
        lr=lr,
        save_dir=save_dir
    )
    
    # 绘制曲线
    print("\n绘制训练曲线...")
    plot_history(history, save_dir=results_dir)
    
    print("\n所有任务完成！")
    print(f"模型权重已保存到：{save_dir}/")
    print(f"训练曲线已保存到：{results_dir}/")


if __name__ == '__main__':
    main()
