"""
案例：绘制激活函数 ReLU 的 函数图像 和 导数图像。

ReLU 激活两数：
    计算公式为 max(0, x)，计算量相对较小，训练成本低，多应用于隐藏层，且适合深层神经网络，
    求导后值要么是 0 要么是 1，相较于 Tanh 收敛速度更快，
    默认情况下 ReLU 只考虑正样本，可以使用 Leaky ReLU、PReLU 来考虑正负样本。
"""
import torch
import matplotlib.pyplot as plt
import zhplot

zhplot.matplotlib_chineseize()

# 1. 创建画布和坐标轴，1 行 2 列
fig, axes = plt.subplots(1, 2)

# 2. 生成 -20 ~ 20 之间的 1000 个数据点
x = torch.linspace(-20, 20, 1000)

# 3. 计算上述 1000 个点，ReLU 激活函数处理后的值
y = torch.relu(x)

# 4. 在第 1 个子图中绘制 ReLU 激活函数的图像
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('ReLU 函数图像')

# 5. 在第 2 个图上，绘制 ReLU 激活函数的导数图像
x = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.relu(x).sum().backward()
# x.detach()：输入值 x 的数值
# x.grad：计算梯度，求导
axes[1].plot(x.detach(), x.grad)
axes[1].grid()
axes[1].set_title('ReLU 导数图像')
plt.show()
