import torch
from lenet5 import LeNet5
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


def load_model(model_path):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = LeNet5()
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"成功加载模型：{model_path}")
    print(f"验证准确率：{checkpoint['val_acc']:.2f}%")
    
    return model, device


def predict_image(model, image_path, device):
    """预测单张图片"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 读取图片
    image = Image.open(image_path).convert('L')  # 转为灰度图
    
    # 预处理
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = output.max(1)
        confidence = torch.softmax(output, dim=1)[0, predicted].item()
    
    return predicted.item(), confidence


def evaluate_model(model, val_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'验证集准确率：{accuracy:.2f}%')
    
    return accuracy


def main():
    """主函数"""
    # 模型路径
    model_path = 'checkpoints/lenet5_best.pth'
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行 train.py 训练模型")
        return
    
    # 加载模型
    model, device = load_model(model_path)
    
    # 如果有测试图片，进行预测
    test_dir = 'test_images'
    if os.path.exists(test_dir):
        print(f"\n在 {test_dir} 目录中查找测试图片...")
        test_images = [f for f in os.listdir(test_dir) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if test_images:
            print(f"找到 {len(test_images)} 张测试图片\n")
            
            for img_name in test_images:
                img_path = os.path.join(test_dir, img_name)
                predicted, confidence = predict_image(model, img_path, device)
                print(f"图片：{img_name:30} 预测：{predicted}, 置信度：{confidence:.4f}")
        else:
            print(f"在 {test_dir} 目录中没有找到测试图片")
    else:
        print(f"\n未找到测试目录 {test_dir}")
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()
