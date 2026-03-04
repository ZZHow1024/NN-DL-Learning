"""
案例：演示自动微分的真实应用场景。

结论：
    1. 先前向传播（正向传播）计算出预测值(z)
    2. 基于损失函数，结合预测值(z)和真实值(y)，来计算梯度。
    3. 结合权重更新公式 W新 = W旧 - 学习率 * 梯度，来更新权重。
"""
import torch

# 定义 x，表示：特征（输入数据）。2 行 5 列，全 1 矩阵
x = torch.ones(2, 5)
print(f'x = {x}')

# 定义 y，表示：标签（真实值）。2 行 3 列，全 0 矩阵
y = torch.zeros(2, 3)
print(f'y = {y}')

# 初始化可自动微分的权重和偏置
w = torch.randn(5, 3, requires_grad=True)  # y = x @ w + b
print(f'w = {w}')

b = torch.randn(3, requires_grad=True)
print(f'b = {b}')

# 前向传播（正向传播）计算出预测值(z)
z = torch.matmul(x, w) + b
# z = x @ w + b
print(f'z = {z}')

# 定义损失函数
criterion = torch.nn.MSELoss()
loss = criterion(z, y)
print(f'loss = {loss}')

# 进行自动微分，求导，结合反向传播，更新权重
loss.sum().backward()

# 打印 w, b 用来更新的梯度
print(f'w = {w.grad}')
print(f'b = {b.grad}')

# 后续根据“W新 = W旧 - 学习率 * 梯度”来更新权重
