"""
案例：演示自动微分模块，实现循环计算梯度，更新参数。

需求：求 y = x**2 + 20 的极小值点并打印 y 是最小值时 w 的值（梯度）

解题步骤：
    1 定义点 x=10 requires_grad=True dtype=torch.float32
    2 定义两数 y = x**2 + 20
    3 利用梯度下降法，循环送代 100 求最优解
    3-1 正向计算（前向传递）
    3-2 梯度清零 x.grad.zero_()
    3-3 反向传播
    3-4 梯度更新 x.data = x.data - 0.01 * x.grad
"""
import torch

# 1 定义点 w=10 requires_grad=True dtype=torch.float32
# 参 1：初始值，参 2：是杏自动微分（求导），参 3：数据类型
w = torch.tensor(10, requires_grad=True, dtype=torch.float)

# 2 定义两数 loss = w**2 + 20
loss = w ** 2 + 20  # 求导：loss' = 2w

# 3 利用梯度下降法，循环送代 100 求最优解
print(f'开始，权重初始值 = {w}，(0.01 * w.grad)：无，loss = {loss}')

# 迭代 100 次，求最优解
for i in range(1, 101):
    # 3-1 正向计算（前向传递）
    loss = w ** 2 + 20
    # 3-2 梯度清零 w.grad.zero_()
    # 第一次时还没计算梯度，为 None，需要做非空判断
    if w.grad is not None:
        w.grad.zero_()
    # 3-3 反向传播
    loss.sum().backward()
    # 3-4 梯度更新 w.data = w.data - 0.01 * w.grad
    print(f'梯度值为：{w.grad}')
    w.data = w.item() - 0.01 * w.grad
    # 打印本次梯度更新后的权重参数结果
    print(f'第 {i} 次，权重初始值 = {w}，(0.01 * w.grad) = {0.01 * w.grad:.5f}，loss = {loss:.5f}')

# 打印最终结果
print(f'最终结果，权重 = {w}，梯度 = {w.grad:.5f}，loss = {loss:.5f}')
