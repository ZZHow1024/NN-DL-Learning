"""
案例：演示回归任务的损失函数

回归任务常用损失函数：
    MAE：Mean Absolute Error，平均绝矿误差。
    MSE：Mean Squared Error，均方误差。
    Smooth L1：平滑 L1。
"""
import torch
import torch.nn as nn


# 1. 演示：MAE 损失函数
def demo01():
    # 1. 定义变量，记录真实值
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    # 2. 定义变量，记录预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True, dtype=torch.float)

    # 3. 创建 MAE 损失函数对象
    criterion = nn.L1Loss()

    # 4. 计算损失
    loss = criterion(y_pred, y_true)

    # 5. 输出损失值
    print(f'MAE损失值 = {loss}')


# 2. 演示：MSE 损失函数
def demo02():
    # 1. 定义变量，记录真实值
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    # 2. 定义变量，记录预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True, dtype=torch.float)

    # 3. 创建 MAE 损失函数对象
    criterion = nn.MSELoss()

    # 4. 计算损失
    loss = criterion(y_pred, y_true)

    # 5. 输出损失值
    print(f'MSE 损失值 = {loss}')


# 3. 演示：MSE 损失函数
def demo03():
    # 1. 定义变量，记录真实值
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)

    # 2. 定义变量，记录预测值
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True, dtype=torch.float)

    # 3. 创建 Smooth L1 损失函数对象
    criterion = nn.SmoothL1Loss()

    # 4. 计算损失
    loss = criterion(y_pred, y_true)

    # 5. 输出损失值
    print(f'Smooth L1 损失值 = {loss}')


if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
