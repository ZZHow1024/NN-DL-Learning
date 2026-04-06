"""
案例：演示批量归一化，它也属于正则化的一种，用于缓解模型的过拟合情况。

批量归一化：
    思路：先对数据做标准化（会丢失一些信息），然后再对数据做缩放（λ, 理解为：W 权重）和平移 (β, 理解为：b 偏置），再找补回一些信息。
    应用场景：批量归一化在计算机视觉领域使用较多。
    BatchNorm1d：主要应用于全连接层或处理一维数据的网络，例如文本处理。它接收形状为 (N, num_features) 的张量作为输入。
    BatchNorm2d：主要应用于卷积神经网络，处理二维图像数据或特征图。它接收形状为 (N, C, H, W) 的张量作为输入。
    BatchNorm3d：主要用于三维卷积神经网络 (3D CNN)，处理三维数据，例如视频或医学图像。它接收形状为 (N, C, D, H, W) 的张量作为输入。
"""
import torch
import torch.nn as nn


# 演示：处理二维数据
def demo01():
    print('演示：处理二维数据')
    # 1. 创建图像样本数据
    # 1 张图片，2 个通道，3 行 4 列
    input_2d = torch.randn(size=(1, 2, 3, 4))
    print(f'input_2d = {input_2d}')

    # 2. 创建批量归一化层（BN 层）
    # 参数 1：输入特征数
    # 参数 2：噪声值（小常数），默认为 1e-5
    # 参数 3：动量值，用于计算移动平均统计量的动量值
    # 参数 4：表示使用可学习的变换参数 (λ, β) 对归一化（标准化）后的数据进行缩放和平移
    bn2d = nn.BatchNorm2d(num_features=2, eps=1e-5, momentum=0.1, affine=True)

    # 3. 对数据进行批量归一化处理
    output_2d = bn2d(input_2d)
    print(f'output_2d = {output_2d}')
    print('-' * 10)


# 演示：处理一维数据
def demo02():
    print('\n演示：处理一维数据')
    # 1. 创建样本数据
    # 2 行 2 列，2 条样本，每个样本有 2 个特征
    input_1d = torch.randn(size=(2, 2))
    print(f'input_1d = {input_1d}')

    # 2. 创建线性层
    linear1 = nn.Linear(2, 4)

    # 3. 对数据进行线性变换
    l1 = linear1(input_1d)
    print(f'l1 = {l1}')

    # 4. 创建批量归一化层
    bn1d = nn.BatchNorm1d(num_features=4)

    # 5. 对线性处理结果 l1 进行批量归一化处理
    output_1d = bn1d(l1)
    print(f'output_1d = {output_1d}')


if __name__ == '__main__':
    demo01()
    demo02()
