"""
案例：演示张量的点乘和矩阵乘法操作。

点乘：
    要求：两个张量的维度保持一致，对应元素直接做相应的操作。
    API：
        t1 * t2
        t1.mul(t2)

矩降乘法：
    要求：两个张量，第一个张量的列数，等于第二个张量的行数（A 列 = B 行）
    结果：A 行 B 列
    API：
        t1 @ t2
        t1.matmul(t2)
        t1.dot (t2)     # 扩展：只针对于一维张量有效。
"""
import torch


# 1. 演示：张量点乘
def demo01():
    print('演示：张量点乘')
    # 1. 定义张量，2 行 3 列
    t1 = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
    print(f't1 = {t1}')
    # 2. 定义张量，2 行 3 列
    t2 = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
    print(f't2 = {t2}')
    # 3. 演示张量点乘操作
    t3 = t1 * t2
    t3 = t1.mul(t2)  # 效果同上
    # 4. 打印结果
    print(f't3 = {t3}')
    print('-' * 10)


# 2. 演示：矩阵乘法
def demo02():
    print('\n演示：矩阵乘法')
    # 条件：A 列 = B 行，结果：A 行 B 列
    # 1. 定义张量，2 行 3 列
    t1 = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
    print(f't1 = {t1}')
    # 2. 定义张量，2 行 3 列
    t2 = torch.tensor([[1, 2],
                       [3, 4],
                       [5, 6]])
    print(f't2 = {t2}')
    # 3. 演示张量矩阵乘法操作
    t3 = t1 @ t2
    t3 = t1.matmul(t2)  # 效果同上
    # 4. 打印结果
    print(f't3 = {t3}')
    print('-' * 10)


# 3. 演示：一维张量的点积
def demo03():
    print('\n演示：一维张量的点积')
    # 条件：一维张量
    # 1. 定义一维张量
    t1 = torch.tensor([1, 2, 3])
    print(f't1 = {t1}')
    # 2. 定义一维张量
    t2 = torch.tensor([4, 5, 6])
    print(f't2 = {t2}')
    # 3. 演示张量矩阵乘法操作
    t3 = t1.dot(t2)
    # 4. 打印结果
    print(f't3 = {t3}')


if __name__ == '__main__':
    demo01()
    demo02()
    demo03()
