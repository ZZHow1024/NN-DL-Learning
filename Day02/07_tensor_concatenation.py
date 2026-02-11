"""
案例：演示张量的拼接操作

涉及到的函数：
    cat()   不改变维度数，拼接张量。除了拼接的那个维度外，其它维度数必须保持一致
    stack() 会改变维度数，拼接张量。所有的维度都必须保持一致
"""
import torch

# 设置随机数种子
torch.manual_seed(1)

# 创建两个张量
t1 = torch.randint(1, 10, (2, 3))
print(f't1 = {t1}')

t2 = torch.randint(1, 10, (2, 3))
print(f't2 = {t2}')

# 演示：张量的拼接
print('演示：张量的拼接')
# cat() 拼接张量
print('cat() 拼接张量')
t3 = torch.cat([t1, t2], dim=0)  # (2, 3) + (2, 3) = (4, 3)
print(f't3 = {t3}, shape = {t3.shape}')
print('-' * 10)

# stack() 拼接张量，可以是新维度，但是无论新旧维度，所有维度都必须保持一致。
print('stack() 拼接张量')
t4 = torch.stack([t1, t2], dim=0)  # (2, 3) + (2, 3) = (2, 2, 3)
print(f't4 = {t4}, shape = {t4.shape}')

t5 = torch.stack([t1, t2], dim=1)  # (2, 3) + (2, 3) = (2, 2, 3)
print(f't5 = {t5}, shape = {t5.shape}')

t6 = torch.stack([t1, t2], dim=2)  # (2, 3) + (2, 3) = (2, 3, 2)
print(f't6 = {t6}, shape = {t6.shape}')
