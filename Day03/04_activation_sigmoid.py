"""
案例：绘制激活函数 Sigmoid 的 函数图像 和 导数图像。

Sigmoid激活两数：
    主要应用于二分类的输出层，且适用于浅层神经网络（不超过5层），
    数据在 [-6, 6] 之间有效果，在 [-3, 3] 之间效果明显，会将数据映射到 [0, 1]，
    求导后范围在 [0, 0.25]。
"""
import torch
import matplotlib.pyplot as plt
import zhplot

zhplot.matplotlib_chineseize()

# 1. 创建画布和坐标轴，1 行 2 列
fig, axes = plt.subplots(1, 2)

# 2. 生成 -20 ~ 20 之间的 1000 个数据点
x = torch.linspace(-20, 20, 1000)

# 3. 计算上述 1000 个点, Sigmoid 激活函数处理后的值
y = torch.sigmoid(x)

# 4. 在第 1 个子图中绘制 Sigmoid 激活函数的图像
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('Sigmoid 函数图像')

# 5. 在第 2 个图上，绘制 Sigmoid 激活函数的导数图像
x = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.sigmoid(x).sum().backward()
# x.detach()：输入值x的数值
# x.grad：计算梯度，求导
axes[1].plot(x.detach(), x.grad)
axes[1].grid()
axes[1].set_title('Sigmoid 导数图像')
plt.show()
