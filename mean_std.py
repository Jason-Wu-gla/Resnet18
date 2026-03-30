import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def calculate_mean_std(data_loader):
    """
    计算数据加载器中所有图像的通道均值和标准差。
    方法：对每个通道在所有像素级别求和，再除以总像素数，得到全局均值；
    再用全局方差公式：E[x^2] - E[x]^2，得到标准差。这样得到的结果才是对整数据集的真实统计量。
    """
    # 使用 CPU 的张量，确保最终输出为 numpy 数组
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    total_count = 0  # 总像素数（N * C * H * W 在遍历结束时应一致）

    for images, _ in data_loader:
        # images: [batch_size, 3, height, width]
        B, C, H, W = images.shape
        images = images.view(B, C, -1)  # [B, 3, H*W]
        # 对每个通道的所有像素求和
        sum_ += images.sum(dim=(0, 2))          # shape [3]
        sum_sq += (images ** 2).sum(dim=(0, 2))  # shape [3]
        total_count += B * images.size(2)

    mean = sum_ / total_count
    var = (sum_sq / total_count) - mean * mean
    std = torch.sqrt(var)

    return mean.numpy(), std.numpy()

def main():
    # 定义基础 transform，仅转换为 tensor，不做标准化
    transform = transforms.ToTensor()

    # 加载 CIFAR-10 训练集和测试集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    print("正在计算训练集均值和标准差...")
    train_mean, train_std = calculate_mean_std(train_loader)

    print("正在计算测试集均值和标准差...")
    test_mean, test_std = calculate_mean_std(test_loader)

    # 输出结果
    print("\n=== CIFAR-10 数据集统计信息 ===")
    print(f"训练集均值: {train_mean}")
    print(f"训练集标准差: {train_std}")
    print(f"测试集均值: {test_mean}")
    print(f"测试集标准差: {test_std}")

    # 通常使用训练集的统计值进行标准化
    print("\n=== 推荐标准化参数（基于训练集）===")
    print(f"mean = {train_mean}")
    print(f"std = {train_std}")

    # 验证与文献值是否接近
    reference_mean = [0.4914, 0.4822, 0.4465]
    reference_std = [0.2470, 0.2435, 0.2616]
    print("\n=== 与文献参考值对比 ===")
    print(f"参考均值: {reference_mean}")
    print(f"参考标准差: {reference_std}")
    print(f"均值差异: {[abs(a - b) for a, b in zip(train_mean, reference_mean)]}")
    print(f"标准差差异: {[abs(a - b) for a, b in zip(train_std, reference_std)]}")

if __name__ == "__main__":
    main()
