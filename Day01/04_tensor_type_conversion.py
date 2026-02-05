"""
案例：创建指定类型的张量

涉及到的函数：
    data.type(torch.DoubleTensor)
    data.half() / double() / float() / short() / int() / long()

需要掌握的函数：type()
"""
import torch

# 演示：直接创建指定类型的张量
t1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)  # 默认是 float32
print(f't1 = {t1}，元素类型 = {t1.dtype}，张量类型 = {type(t1)}')

# 演示：创建好张量后，做类型转换
# 方式 1. type() 函数
t2 = t1.type(torch.int16)
print(f't2 = {t2}，元素类型 = {t2.dtype}，张量类型 = {type(t2)}')
print('-' * 10)

# 方式 2. half() / double() / float() / short() / int() / long()
print(t2.half())
print(t2.float())
print(t2.double())
print(t2.short())
print(t2.int())
print(t2.long())
