"""
案例：演示如何创建全 0、全 1、指定值的张量

涉及到的两数如下：
    torch.ones 和 torch.ones_like 创建全 1 张量
    torch.zeros 和 torch.zeros_like 创建全 0 张量
    torch.full 和 torch.full_like 创建全为指定值张量

需要掌握的函数：
    zeros(), full()
"""
import torch


# 演示：torch.ones 和 torch.ones_like 创建全 1 张量
def demo01():
    print('演示：torch.ones 和 torch.ones_like 创建全 1 张量')
    # 场景 1：创建 2 行 3 列全 1 张量
    print('场景 1：创建 2 行 3 列全 1 张量')
    t1 = torch.ones(2, 3)
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 10)

    # 场景 2：创建 3 行 2 列张量
    print('场景 2：创建 3 行 2 列张量')
    t2 = torch.tensor([[1, 2],
                       [2, 3],
                       [4, 5]])
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 10)

    # 场景 3：基于 t2 的形状，创建全 1 张量
    print('场景 3：基于 t2 的形状，创建全 1 张量')
    t3 = torch.ones_like(t2)
    print(f't3 = {t3}, type = {type(t3)}')


# 演示：torch.zeros 和 torch.zeros_like 创建全 0 张量
def demo02():
    print('\n演示：torch.zeros 和 torch.zeros_like 创建全 0 张量')
    # 场景 1：创建 2 行 3 列全 1 张量
    print('场景 1：创建 2 行 3 列全 0 张量')
    t1 = torch.zeros(2, 3)
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 10)

    # 场景 2：创建 3 行 2 列张量
    print('场景 2：创建 3 行 2 列张量')
    t2 = torch.tensor([[1, 2],
                       [2, 3],
                       [4, 5]])
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 10)

    # 场景 3：基于 t2 的形状，创建全 0 张量
    print('场景 3：基于 t2 的形状，创建全 0 张量')
    t3 = torch.zeros_like(t2)
    print(f't3 = {t3}, type = {type(t3)}')


# 演示：torch.full 和 torch.full_like 创建全为指定值张量
def demo03():
    print('\n演示：torch.full 和 torch.full_like 创建全为指定值张量')
    # 场景 1：创建 2 行 3 列全 255 张量
    print('场景 1：创建 2 行 3 列全 255 张量')
    t1 = torch.full(size=(2, 3), fill_value=255)
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 10)

    # 场景 2：创建 3 行 2 列张量
    print('场景 2：创建 3 行 2 列张量')
    t2 = torch.tensor([[1, 2],
                       [2, 3],
                       [4, 5]])
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 10)

    # 场景 3：基于 t2 的形状，创建全 255 张量
    print('场景 3：基于 t2 的形状，创建全 255 张量')
    t3 = torch.full_like(t2, 255)
    print(f't3 = {t3}, type = {type(t3)}')


if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
