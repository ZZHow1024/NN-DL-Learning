"""
案例：演示池化层相关操作。

池化层 (Pooling)：
    目的：降维
    方法：
        最大池化
        平均池化
    特点：池化不会改变数据的通道数。
"""
import torch
import torch.nn as nn


# 演示：单通道池化
def demo01():
    print('演示：单通道池化')
    # 1. 创建 1 个 1 通道 3 * 3 的二维矩阵
    inputs = torch.tensor([[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]], dtype=torch.float)
    print(f'inputs: {inputs}, shape: {inputs.shape}')

    # 2. 创建最大池化层
    print('演示最大池化')
    # 参数 1：池化核大小；参数 2：步长；参数 3：填充
    pool1 = nn.MaxPool2d(2, 1, 0)
    outputs = pool1(inputs)
    print(f'outputs: {outputs}, shape: {outputs.shape}')

    # 3. 创建平均池化
    print('演示平均池化')
    # 参数 1：池化核大小；参数 2：步长；参数 3：填充
    pool2 = nn.AvgPool2d(2, 1, 0)
    outputs = pool2(inputs)
    print(f'outputs: {outputs}, shape: {outputs.shape}')
    print('-' * 10)


# 演示：多通道池化
def demo02():
    print('演示：多通道池化')
    # 1. 创建 1 个 3 通道 3 * 3 的二维矩阵
    inputs = torch.tensor([[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]],
                           [[10, 20, 30],
                            [40, 50, 60],
                            [70, 80, 90]],
                           [[11, 22, 33],
                            [44, 55, 66],
                            [77, 88, 99]]
                           ], dtype=torch.float)
    print(f'inputs: {inputs}, shape: {inputs.shape}')

    # 2. 创建最大池化层
    print('演示最大池化')
    # 参数 1：池化核大小；参数 2：步长；参数 3：填充
    pool1 = nn.MaxPool2d(2, 1, 0)
    outputs = pool1(inputs)
    print(f'outputs: {outputs}, shape: {outputs.shape}')

    # 3. 创建平均池化
    print('演示平均池化')
    # 参数 1：池化核大小；参数 2：步长；参数 3：填充
    pool2 = nn.AvgPool2d(2, 1, 0)
    outputs = pool2(inputs)
    print(f'outputs: {outputs}, shape: {outputs.shape}')


if __name__ == '__main__':
    demo01()
    demo02()
