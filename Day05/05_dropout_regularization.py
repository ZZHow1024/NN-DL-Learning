"""
案例：演示随机失活

正则化的作用：缓解模型的过拟合情況。

正则化的方式：
    L1正则化：权重可以变为口，相当于：降维。
    L2正则化：权重可以无限接近 0。
    Dropout：随机失活，每批次样本训练时，随机让一部分神经元死亡，防止一些特征对结果的影购较大（防止过拟合）。
    BN（批量归一化）
"""
import torch
import torch.nn as nn


# 演示：随机失活 (Dropout)
def demo01():
    # 1. 创建隐藏层
    t1 = torch.randint(0, 10, size=(1, 4)).float()
    print(f't1 = {t1}')

    # 2. 进行下一层加权求和和激活函数计算
    # 2.1 创建全连接层
    linear1 = nn.Linear(4, 5)

    # 2.2 加权求和
    l1 = linear1(t1)
    print(f'l1 = {l1}')

    # 2.3 激活函数
    output = torch.relu(l1)
    print(f'output = {output}')

    # 3. 对激活值进行随机失活 Dropout 处理 -> 只有训练阶段有，测试阶段没有。
    dropout = nn.Dropout(p=0.4)
    d1 = dropout(output)
    print(f'd1（随机失活后的数据） = {d1}')


if __name__ == '__main__':
    demo01()
