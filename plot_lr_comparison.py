import os
from plot import plot_learning_rate_comparison

# 定义存储 step-loss 数据的目录
data_dir = "results/data/loss"

# 获取所有以 train_step_losses_lr_ 开头的 .npy 文件
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("train_step_losses_lr_") and f.endswith(".npy")]

# 调用绘图函数
plot_learning_rate_comparison(file_paths, save_path="results/plot/lr_comparison.png", show=True)