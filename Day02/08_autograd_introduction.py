"""
案例：演示自动微分模块，具体如何求导。

权重更新公式：W新 = W旧 - 学习率 * 梯度
            梯度 = 损失函数的导数
"""
import torch

# 定义变量，记录初始权重 W旧
# 参 1：初始值，参 2：是杏自动微分（求导），参 3：数据类型
w = torch.tensor(10, requires_grad=True, dtype=torch.float)

# 定义 loss 变量，表示损失函数
loss = 2 * w ** 2  # loss = 2w² -> 求导 4w

# 打印梯度函数类型
print(f'梯度函数类型 = {type(loss.grad_fn)}')
print(f'loss.sum() = {loss.sum()}')

# 计算梯度
loss.sum().backward()  # 保证 loss 是一个标量

# 带入权重更新公式：W新 = W旧 - 学习率 * 梯度
w.data = w.item() - 0.01 * w.grad

# 打印最终结果
print(f'更新后的权重 = {w}')
