"""
案例：演示张量和 numpy 之间如何相互转换，以及如何从标量张量中提取其内容。

涉及到的 API：
    场景 1：张量 -> numpy nd 数组对象
        张量对象.numpy()        共享对象
        张量对象.numpy().copy() 不共享内存，链式编程写法

    场景 2：numpy nd 数组 -> 张量
        from_numpy()            共享内存
        torch.tensor(nd 数组)    不共享内存

    场景 3：从标量张量中提取其内容
        标量张量.item()

    需要掌握的内容：
        张量 -> numpy：张量对象.numpy()
        numpy -> 张量：torch.tensor(nd 数组)
        从标量张量中提取其内容：标量张量.item()
"""
import torch
import numpy as np


# 场景 1. 演示：张量 -> numpy nd 数组对象
def demo01():
    print('演示：张量 -> numpy nd 数组对象')
    # 1. 创建张量
    t1 = torch.tensor([1, 2, 3, 4, 5])
    print(f't1 = {t1}, type = {type(t1)}')
    # 2. 张量 -> numpy nd 数组对象
    n1 = t1.numpy()
    n2 = t1.numpy().copy()
    print(f'n1 = {n1}, type = {type(n1)}')
    # 3. 演示共享内存
    n1[0] = 100
    print(f'n1 = {n1}\nn2 = {n2}\nt1 = {t1}')
    print('-' * 10)


# 场景 2. 演示：numpy nd 数组 -> 张量
def demo02():
    print('\n演示：numpy nd 数组 -> 张量')
    # 1. 创建 numpy nd 数组
    n1 = np.array([1, 2, 3])
    print(f'n1 = {n1}, type = {type(n1)}')
    # 2. numpy nd 数组 -> 张量
    t1 = torch.from_numpy(n1)
    t2 = torch.tensor(n1)
    print(f't1 = {t1}, type = {type(t1)}')
    # 3. 演示共享内存
    n1[0] = 100
    print(f'n1 = {n1}')
    print(f't1 = {t1}')
    print(f't2 = {t2}')
    print('-' * 10)


# 场景 3. 演示：从标量张量中提取其内容
def demo03():
    print('\n演示：从标量张量中提取其内容')
    # 1. 创建张量
    t1 = torch.tensor(100)
    print(f't1 = {t1}, type = {type(t1)}')
    # 2. 从张量中提取内容
    i1 = t1.item()
    print(f'i1 = {i1}, type = {type(i1)}')


if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
