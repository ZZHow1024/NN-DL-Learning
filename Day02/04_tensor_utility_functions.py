"""
案例：

涉及到的函数：
    sum(), max(), min(), mean()  # 都有 dim 参数，0 表示列，1 表示 行
    pow(), sqrt(), exp(), log(), log2(), log10()  # 没有 dim 参数

需要掌握的函数：
    sum(), max(), min(), mean()
    a ** 5
"""
import torch

# 1, 定义张量，记录初值
t1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]], dtype=torch.float)
print(f't1 = {t1}')

# 2. 演示有 dim 参数的函数
print('\n演示有 dim 参数的函数')
# sum() 求和
print('sum() 求和')
print(t1.sum(dim=0))  # 按列求和
print(t1.sum(dim=1))  # 按行求和
print(t1.sum())  # 整体求和
print('-' * 10)

# max() 求最大值，min() 同理
print('max() 求最大值，min() 同理')
print(t1.max(dim=0))  # 按列求最大值
print(t1.max(dim=1))  # 按行求最大值
print(t1.max())  # 整体求最大值
print('-' * 10)

# mean() 计算平均值
print('mean() 计算平均值')
print(t1.mean(dim=0))  # 按列求平均值
print(t1.mean(dim=1))  # 按行求平均值
print(t1.mean())  # 整体求平均值
print('-' * 10)

# 3. 演示没有 dim 参数的函数
print('\n演示没有 dim 参数的函数')
# pow() n 次幂
print(f't1.pow(2) = {t1.pow(2)}')
print(f't1 ** 2 = {t1 ** 2}')  # 效果同上

# sqrt() 平方根
print(f't1.sqrt() = {t1.sqrt()}')

# exp() e 的 n 次幂
print(f't1.exp() = {t1.exp()}')

# log() log2(), log10() 对数
print(f't1.log() = {t1.log()}')
print(f't1.log2() = {t1.log2()}')
print(f't1.log10() = {t1.log10()}')
