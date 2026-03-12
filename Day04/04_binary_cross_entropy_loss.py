"""
案例：演示二分类任务的损失函数。

二分类任务的损失函数(BCELoss)：
    公式：Loss = -ylog(预测值) - (1-y)log(1 - 预测值)
    细节：因为公式中没有包含 Sigmoid 激活函数，所以使用 BCELoss 的时候，还需要手动指定 Sigmoid。
"""
import torch
import torch.nn as nn


# 1. 演示：而分类任务的损失函数
def demo01():
    # 1. 设置真实值
    y_true = torch.tensor([0, 1, 0], dtype=torch.float)

    # 2. 设置预测值（概率）
    y_pred = torch.tensor([0.6801, 0.5423, 0.2639])

    # 3. 创建二分类交叉熵损失函数
    criterion = nn.BCELoss()  # reduction：str = "mean" -> 均值

    # 4. 计算损失值
    loss = criterion(y_pred, y_true)
    print(f'loss = {loss}')


if __name__ == '__main__':
    demo01()
