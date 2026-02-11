"""
案例：演示张量的形状操作。

涉及到的函数：
    reshape()       在不改变张量内容的前提下，对其形状做改变
    unsqueeze()     在指定的轴上增加一个(1)维度，等价于：升维
    squeeze()       删除所有为 1 的维度，等价于：降维
    transpose()     一次只能交换 2 个维度
    permute()       一次可以同时交换多个维度
    view()          只能修改连续的张量的形状，连续张量=内存中存储顺序和在张量中显示的顺序相同
    contiguous()    把不连续的张量 -> 连续的张量，即：基于张量中显示的顺序，修改内存中的存储顺序
    is_contiguous() 判断张量是否是连续的

需要掌握的函数：reshape(), unsqueeze(), permute(), view()
"""
import torch
from numpy.testing.print_coercion_tables import print_new_cast_table

# 指定随机种子
torch.manual_seed(1)


# 演示：reshape() 函数
def demo01():
    print('演示：reshape() 函数')
    # 定义 2 行 3 列的张量
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1 = {t1}, shape = {t1.shape}, row = {t1.shape[0]}, columns={t1.shape[1]}, {t1.shape[-1]}')

    # 通过 reshape() 函数，把 t1 转换成 3 行 2 列， 1 行 6 列，6 行 1 列
    t2 = t1.reshape(3, 2)
    print(f't2 = {t2}, shape = {t2.shape}, row = {t2.shape[0]}, columns={t2.shape[1]}, {t2.shape[-1]}')
    t2 = t1.reshape(1, 6)
    print(f't2 = {t2}, shape = {t2.shape}, row = {t2.shape[0]}, columns={t2.shape[1]}, {t2.shape[-1]}')
    t2 = t2.reshape(6, 1)
    print(f't2 = {t2}, shape = {t2.shape}, row = {t2.shape[0]}, columns={t2.shape[1]}, {t2.shape[-1]}')
    print('-' * 10)


# 演示：unsqueeze(), squeeze() 函数
def demo02():
    print('\n演示：unsqueeze(), squeeze() 函数')
    # 定义 2 行 3 列的张量
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1 = {t1}, shape = {t1.shape}')
    # 在 0 维上，添加一个维度
    t2 = t1.unsqueeze(0)
    print(f't2 = {t2}, shape = {t2.shape}')
    # 在 1 维上，添加一个维度
    t2 = t1.unsqueeze(1)
    print(f't2 = {t2}, shape = {t2.shape}')
    # 在 2 维上，添加一个维度
    t2 = t1.unsqueeze(2)
    print(f't2 = {t2}, shape = {t2.shape}')

    # 删除所有为 1 的维度
    t3 = torch.randint(1, 10, size=(2, 1, 3, 1, 1))
    print(f't3 = {t3}, shape = {t3.shape}')
    t4 = t3.squeeze()
    print(f't4 = {t4}, shape = {t4.shape}')
    print('-' * 10)


# 演示：transpose(), permute() 函数
def demo03():
    print('\n演示：transpose(), permute() 函数')
    # 定义张量
    t1 = torch.randint(1, 10, size=(2, 3, 4))

    # 改变维度 (2, 3, 4) -> (3, 2, 4)
    t2 = t1.transpose(0, -1)
    print(f't2 = {t2}, shape = {t2.shape}')

    # 改变维度 (2, 3, 4) -> (4, 2, 3)
    t2 = t1.permute(2, 0, 1)
    print(f't2 = {t2}, shape = {t2.shape}')
    print('-' * 10)


# 演示：view(), contiguous(), is_contiguous() 函数
def demo04():
    print('\n演示：view(), contiguous(), is_contiguous() 函数')
    # 定义张量
    t1 = torch.randint(1, 10, size=(2, 3))
    print(f't1 = {t1}, shape = {t1.shape}')

    # 判断张量是否连续
    print(t1.is_contiguous())  # True

    # 通过 view() 函数修改上述张量形状
    t2 = t1.view(3, 2)
    print(f't2 = {t2}, shape = {t2.shape}')
    print(t2.is_contiguous())  # True

    # 通过 transpose() 交换维度，交换之后不连续了
    t3 = t1.transpose(0, 1)
    print(f't3 = {t3}, shape = {t3.shape}')
    print(t3.is_contiguous())  # False

    # 通过 contiguous() 函数，把 t3 张量转为连续张量
    t4 = t3.contiguous().view(2, 3)
    print(f't4 = {t4}, shape = {t4.shape}')
    print(t4.is_contiguous())  # True


if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
    demo04()
