"""
案例：演示多分类任务的交叉熵损失函数。

损失函数介绍：
    概述：损失函数也叫成本函数、目标函数、代价函数、误差函数，就是用来衡量模型好坏（模型拟合情況）的。
    分类：
        分类问题：
            多分类交叉熵损失：CrossEntropyLoss
            二分类交叉熵损失：BCELoss
        回归问题：
            MAE：Mean Absolute Error，平均绝矿误差。
            MSE：Mean Squared Error，均方误差。
            Smooth L1：结合上述两个的特点做的升级优化。

多分类交叉熵损失：CrossEntropyLoss
    设计思路：Loss = -Σlog(S(f(x)))
    简单记亿：
        x：样本。
        f(x)：加权求和。
        S(f(x))：处理后的概率。
        y：样本 x 属于某一个类别的真实概率。
"""
import torch
import torch.nn as nn


# 演示：多分类交叉熵损失
def demo01():
    # 1. 手动创建样本的真实值 -> y
    # y_true = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.float)
    y_true = torch.tensor([1, 2])

    # 2. 手动创建样本的预测值 -> f(x)
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], requires_grad=True, dtype=torch.float)

    # 3. 创建多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 4. 计算损失值
    loss = criterion(y_pred, y_true)
    print(f'损失值 = {loss}')


if __name__ == '__main__':
    demo01()
