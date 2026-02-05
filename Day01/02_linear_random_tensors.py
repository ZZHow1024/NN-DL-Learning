"""
案例：演示 PyTorch 中如何创建线性和随机张量

涉及到的函数：
    torch.arange() 和 torch.linspace() 创建线性张量
    torch.random.initial_seed () 和 torch.random.manual_seed () 随机种子没置
    torch.rand / randn() 创建随机浮点类型张量
    torch.randint(low, high, size=()) 创建随机整数类型张量

需要掌握的函数：arange(), linspace(), manual_seed(), randint()
"""
import torch


# 1. 演示：创建线性张量
def demo01():
    print('演示：创建线性张量')
    # 场景 1：创建指定范围的线性张量
    print('场景 1：创建指定范围的线性张量')
    # 参数 1：起始值，参数 2：结束值，参数3：步长
    t1 = torch.arange(0, 10, 2)
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 10)

    # 场景 2：创建指定范围的线性张量 -> 等差数列
    print('场景 2：创建指定范围的线性张量 -> 等差数列')
    # 参数 1：起始值，参数 2：结束值，参数3：元素的个数
    t2 = torch.linspace(1, 10, 4)
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 10)


# 2. 演示：创建随机张量
def demo02():
    print('\n演示：创建随机张量')
    # 场景 1：均匀分布的 (0, 1) 随机张量
    print('场景 1：均匀分布的 (0, 1) 随机张量')
    # step1. 设置随机种子
    # torch.initial_seed()  # 默认采用当前系统的时间戳作为随机种子
    torch.manual_seed(1)  # 设置随机种子

    # step2. 创建随机张量
    t1 = torch.randn(size=(2, 3))
    print(f't1 = {t1}, type = {type(t1)}')
    print('-' * 10)

    # 场景 2：符合正态分布的随机张量
    print('场景 2：符合正态分布的随机张量')
    t2 = torch.randn(size=(2, 3), requires_grad=True)
    print(f't2 = {t2}, type = {type(t2)}')
    print('-' * 10)

    # 场景 3：创建随机整数张量
    print('场景 3：创建随机整数张量')
    t3 = torch.randint(0, 10, size=(2, 3))
    print(f't3 = {t3}, type = {type(t3)}')
    print('-' * 10)


if __name__ == '__main__':
    demo01()
    demo02()
