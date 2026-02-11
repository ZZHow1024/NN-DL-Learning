"""
案例：演示张量的基木运算。

涉及到的API：
    add(), sub(), mul(), div(), neg() -> 加减乘除取反
    add_(), sub_(), mul(), div_(), neg_() -> 功能同上，但可以修政源数据

需要掌握的：+, -, *, /
"""
import torch

# 1. 创建张量
t1 = torch.tensor([1, 2, 3])

# 2. 演示加法
t2 = t1.add(10)  # 不会修改原数据
t2 = t1 + 10  # 效果同上

t1.add_(10)  # 会修改原数据
t1 += 10  # 效果同上

# 演示其他函数
t2 = t1.sub(1)  # 减法
t2 = t1.mul(2)  # 乘法
t2 = t1.div(2)  # 除法
t2 = t1.neg()  # 取反

# 打印结果
print(f't1 = {t1}, type = {type(t1)}')
print(f't2 = {t2}, type = {type(t2)}')
