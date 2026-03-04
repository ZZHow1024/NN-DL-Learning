"""
案例：演示 detach() 函数的功能，解决自动微分的弊端。

问题：一个张量一旦设置了自动微分，这个张量就不能直接转成 numpy 的 ndarray 对象了，需要通过 detach() 函数解决。
"""
import torch

# 定义张量
t1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float)
print(f't1 = {t1}, type(t1) = {type(t1)}')

# 通过 detach() 函数拷贝一份张量，然后转换
t2 = t1.detach()
print(f't2 = {t2}, type(t2) = {type(t2)}')

# 测试 t1 和 t2 是否共享同一空间 -> 共享
t1.data[0] = 100
print(f't1 = {t1}, type(t1) = {type(t1)}')
print(f't2 = {t2}, type(t2) = {type(t2)}')

# 查看 t1 和 t2 谁可以自动微分
print(f't1.requires_grad = {t1.requires_grad}')
print(f't2.requires_grad = {t2.requires_grad}')

# 将 t2 转 numpy 的 ndarray 对象
n1 = t2.numpy()
print(f'n1 = {n1}, type(n1) = {type(n1)}')

# 最终版
n2 = t1.detach().numpy()
print(f'n2 = {n2}, type(n2) = {type(n2)}')
