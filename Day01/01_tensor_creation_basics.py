"""
案例：演示张量的基本创建方式

张量基本创建方式：
    torch.tensor 根据指定数据创建张量。
    torch.Tensor 根据形状创建张量, 其也可用来创建指定数据的张量。
    torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量。
"""
import torch
import numpy as np


# 1. 演示：torch.tensor 根据指定数据创建张量
def demo01():
    print('演示：torch.tensor 根据指定数据创建张量')
    # 场景 1：标量 张量
    print('场景 1：标量 张量')
    t1 = torch.tensor(10)
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 10)

    # 场景 2：二维列表 -> 张量
    print('场景 2：二维列表 -> 张量')
    data = [[1, 2, 3],
            [4, 5, 6]]
    t2 = torch.tensor(data)
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 10)

    # 场景 3：numpy nd 数组 -> 张量
    print('场景 3：numpy nd 数组 -> 张量')
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.tensor(data, dtype=torch.float)
    print(f't3 = {t3}, type = {type(t3)}')
    print('-' * 10)

    # 场景 4：尝试直接创建指定维度的张量（通过 torch.tensor() 的方式不行）
    # t4 = torch.tensor(2, 3)
    # print(f't4 = {t4}, type = {type(t4)}')


# 2. 演示：torch.Tensor 根据形状创建张量，其也可用来创建指定数据的张量
def demo02():
    print('\n演示：torch.Tensor 根据形状创建张量，其也可用来创建指定数据的张量')
    # 场景 1：标量 张量
    print('场景 1：标量 张量')
    t1 = torch.Tensor(1)
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 10)

    # 场景 2：二维列表 -> 张量
    print('场景 2：二维列表 -> 张量')
    data = [[1, 2, 3],
            [4, 5, 6]]
    t2 = torch.Tensor(data)
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 10)

    # 场景 3：numpy nd 数组 -> 张量
    print('场景 3：numpy nd 数组 -> 张量')
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.Tensor(data)
    print(f't3 = {t3}, type = {type(t3)}')
    print('-' * 10)

    # 场景 4：直接创建指定维度的张量
    print('场景 4：直接创建指定维度的张量')
    t4 = torch.Tensor(2, 3)
    print(f't4 = {t4}, type = {type(t4)}')


# 3. 演示：torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量
def demo03():
    print('\n演示：torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量')
    # 场景 1：标量 张量
    print('场景 1：标量 张量')
    t1 = torch.IntTensor(1)
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 10)

    # 场景 2：二维列表 -> 张量
    print('场景 2：二维列表 -> 张量')
    data = [[1, 2, 3],
            [4, 5, 6]]
    t2 = torch.IntTensor(data)
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 10)

    # 场景 3：numpy nd 数组 -> 张量
    print('场景 3：numpy nd 数组 -> 张量')
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f't3 = {t3}, type = {type(t3)}')
    print('-' * 10)

    # 场景 4：如果类型不匹配，会尝试自动转换类型
    print('场景 4：如果类型不匹配，会尝试自动转换类型')
    data = np.random.randint(0, 10, size=(2, 3))
    t4 = torch.FloatTensor(data)
    print(f't4 = {t4}, type = {type(t4)}')
    print('-' * 10)


if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
