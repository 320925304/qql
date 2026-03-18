import torch
from lenet5 import LeNet5
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse


def load_model(model_path):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LeNet5()
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"成功加载模型：{model_path}")
    print(f"训练时验证准确率：{checkpoint['val_acc']:.2f}%")
    
    return model, device


def get_transform():
    """获取数据预处理"""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def predict_image(model, image_path, device, class_names=None):
    """预测单张图片"""
    transform = get_transform()
    
    image = Image.open(image_path).convert('L')
    original_image = image.copy()
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    predicted_class = predicted.item()
    confidence_value = confidence.item()
    
    if class_names:
        predicted_label = class_names[predicted_class]
    else:
        predicted_label = str(predicted_class)
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'confidence': confidence_value,
        'original_image': original_image,
        'probabilities': probabilities[0].cpu().numpy()
    }


def predict_batch(model, image_dir, device, class_names=None):
    """批量预测目录中的图片"""
    results = []
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"在 {image_dir} 中没有找到图片文件")
        return results
    
    print(f"找到 {len(image_files)} 张图片，开始预测...\n")
    
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        result = predict_image(model, img_path, device, class_names)
        result['image_name'] = img_name
        result['image_path'] = img_path
        results.append(result)
    
    return results


def visualize_prediction(result, class_names=None, save_path=None):
    """可视化预测结果"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(result['original_image'], cmap='gray')
    axes[0].set_title(f"预测: {result['predicted_label']}\n置信度: {result['confidence']:.2%}")
    axes[0].axis('off')
    
    num_classes = len(result['probabilities'])
    if class_names:
        labels = class_names
    else:
        labels = [str(i) for i in range(num_classes)]
    
    axes[1].bar(range(num_classes), result['probabilities'])
    axes[1].set_xticks(range(num_classes))
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel('类别')
    axes[1].set_ylabel('概率')
    axes[1].set_title('各类别概率分布')
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
    
    plt.show()


def print_results(results, class_names=None):
    """打印预测结果"""
    print("=" * 60)
    print(f"{'图片名称':<30} {'预测类别':<10} {'置信度':<10}")
    print("=" * 60)
    
    for result in results:
        print(f"{result['image_name']:<30} {result['predicted_label']:<10} {result['confidence']:.2%}")
    
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='手写数字识别预测脚本')
    parser.add_argument('--model', type=str, default='checkpoints/lenet5_best.pth',
                        help='模型文件路径')
    parser.add_argument('--image', type=str, default=None,
                        help='单张图片路径')
    parser.add_argument('--dir', type=str, default=None,
                        help='图片目录路径（批量预测）')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化预测结果')
    parser.add_argument('--save', type=str, default=None,
                        help='保存可视化结果的路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"错误：找不到模型文件 {args.model}")
        print("请先运行 train.py 训练模型")
        return
    
    model, device = load_model(args.model)
    
    class_names = ['0', '1']
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"错误：找不到图片文件 {args.image}")
            return
        
        print(f"\n预测图片: {args.image}")
        result = predict_image(model, args.image, device, class_names)
        
        print(f"\n预测结果: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.2%}")
        
        if args.visualize:
            visualize_prediction(result, class_names, args.save)
    
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"错误：找不到目录 {args.dir}")
            return
        
        results = predict_batch(model, args.dir, device, class_names)
        
        if results:
            print_results(results, class_names)
            
            if args.visualize:
                for result in results:
                    visualize_prediction(result, class_names)
    
    else:
        test_dir = 'test_images'
        if os.path.exists(test_dir):
            results = predict_batch(model, test_dir, device, class_names)
            if results:
                print_results(results, class_names)
        else:
            print("\n使用方法:")
            print("  预测单张图片: python predict.py --image 图片路径 --visualize")
            print("  批量预测:     python predict.py --dir 图片目录 --visualize")
            print("\n示例:")
            print("  python predict.py --image test.png --visualize")
            print("  python predict.py --dir test_images")
    
    print("\n预测完成！")


if __name__ == '__main__':
    main()
