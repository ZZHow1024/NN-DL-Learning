# 神经网络与深度学习笔记_Day3

@ZZHow(ZZHow1024)

参考课程：

【**黑马程序员AI大模型《神经网络与深度学习》全套视频课程，涵盖Pytorch深度学习框架、BP神经网络、CNN图像分类算法及RNN文本生成算法**】

[[**https://www.bilibili.com/video/BV1c5yrBcEEX**](https://www.bilibili.com/video/BV1c5yrBcEEX)]

# 01_自动微分小问题_detach函数

- 梯度计算注意点
    - 不能将自动微分的张量转换成 numpy 数组，会发生报错，可以通过 detach() 方法实现。
- 案例演示：[**01_detach_function.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day03/01_detach_function.py)（自动微分_detach () 函数介绍）

# 02_自动微分真实应用场景

- 本小节主要讲解了 PyTorch 中非常重要的自动微分模块的使用和理解。
- 我们对需要计算梯度的张量需要设置 `requires_grad=True` 属性。
- 案例演示：[**02_autograd_real_world_applications.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day03/02_autograd_real_world_applications.py)（自动微分真实应用场景）

# 03-05_PyTorch模拟线性回归

- 我们使用 PyTorch 的各个组件来构建线性回归的实现。在 PyTorch 中进行模型构建的整个流程一般分为四个步骤：
    - 准备训练集数据
        - numpy 对象 → 张量 Tensor → 数据集对象 TensorDataset → 数据加载器 DataLoader
    - 构建要使用的模型
    - 设置损失函数和优化器
    - 模型训练
- 要使用的 API
    - 使用 PyTorch 的 `nn.MSELoss()` 代替平方损失函数
    - 使用 PyTorch 的 `data.DataLoader` 代替数据加截器
    - 使用 PyTorch 的 `optim.SGD` 代替优化器
    - 使用 PyTorch 的 `nn.Linear` 代替假设函数
- 案例演示：[**03_linear_regression_simulation.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day03/03_linear_regression_simulation.py)（PyTorch框架_模拟线性回归）

# 06_如何构建神经网络(neural network)

- 什么是神经网络
    - 人工神经网络 (Artificial Neural Network，简写为 **ANN**) 也简称神经网络(NN)，是一种模仿生物神经网络结构和功能的**计算模型**。人脑可以看做是一个生物神经网络，由众多的**神经元**连接而成。各个神经元传递复杂的电信号，树突接收到**输入信号**，然后对信号进行处理，通过轴突**输出信号**。
- 如何构建神经网络
    - 神经网络是由多个神经元组成，构建神经网络就是在构建神经元。
    
    ![构建神经元](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day3/%E6%9E%84%E5%BB%BA%E7%A5%9E%E7%BB%8F%E5%85%83.png)
    
    构建神经元
    
    - 这个过程就像来源不同树突（树突都会有不同的权重）的信息，进行的加权计算，输入到细胞中做加和，再通过激活函数输出细胞值。
    - 接下来，我们使用多个神经元来构建神经网络，相邻层之间的神经元相互连接，并给每一个连接分配一个强度。
    
    ![构建神经网络](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day3/%E6%9E%84%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.png)
    
    构建神经网络
    

# 07_神经网络_文字介绍

- 如何构建神经网络
    - 神经网络中信息只向一个方向移动, 即从输入节点向前移动, 通过隐藏节点, 再向输出节点移动。其中的基本部分是：
        1. **输入层(Input Layer)**：即输入 x 的那一层（如图像、文本、声音等）。每个输入特征对应一个神经元。输入层将数据传递给下一层的神经元。
        2. 输出层(Output Layer)：即输出 y 的那一层。输出层的神经元根据网络的任务（回归、分类等）生成最终的预测结果。
        3. 隐藏层(Hidden Layers)：输入层和输出层之间都是隐藏层，神经网络的“深度”通常由隐藏层的数量决定。隐藏层的神经元通过加权和激活函数处理输入，并将结果传递到下一层。
    - 特点是：
        - 同一层的神经元之间没有连接
        - 第 N 层的每个神经元和第 N-1 层的所有神经元相连（这就是 full connected 的含义)，这就是全连接神经网络
        - 全连接神经网络接收的样本数据是二维的，数据在每一层之间需要以二维的形式传递
        - 第 N-1 层神经元的输出就是第N层神经元的输入
        - 每个连接都有一个权重值（w 系数和 b 系数）
- 神经网络内部状态值和激活值
    
    ![神经网络内部状态值和激活值](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day3/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%86%85%E9%83%A8%E7%8A%B6%E6%80%81%E5%80%BC%E5%92%8C%E6%BF%80%E6%B4%BB%E5%80%BC.png)
    
    神经网络内部状态值和激活值
    
- 每一个神经元工作时，**前向传播**会产生两个值，内部状态值（加权求和值）和激活值；**反向传播**时会产生激活值梯度和内部状态值梯度。
- 内部状态值
    - 神经元或隐藏单元的内部存储值，它反映了当前神经元接收到的输入、历史信息以及网络内部的权重计算结果。
    - $z = W⋅x + b$
        - $W$：权重矩阵
        - $x$：输入值
        - $b$：偏置
- 激活值
    - 通过激活函数（如 ReLU、Sigmoid、Tanh）对内部状态值进行非线性变换后得到的结果。激活值决定了当前神经元的输出。
    - $a = f(z)$
        - $f$：激活函数
        - $z$：内部状态值

# 08_激活函数介绍

- **激活函数**用于对每层的输出数据进行变换，进而为整个网络注入了非线性因素。此时，神经网络就可以拟合各种曲线。
    1. 没有引入非线性因素的网络等价于使用一个线性模型来拟合。
    2. 通过给网络输出增加激活函数，实现引入非线性因素，使得网络模型可以逼近任意函数，提升网络对复杂问题的拟合能力。
- 如果**不使用激活函数**，整个网络虽然看起来复杂，其本质还相当于一种**线性模型**。

# 09_Sigmoid激活函数介绍

- Sigmoid 激活函数公式：$f(x) = \frac{1}{1 + e^{-x}}$
- Sigmoid 激活函数求导公式：$f'(x) = (\frac{1}{1+e^{-x}})' = \frac{1}{1 + e^{-x}}(1 - \frac{1}{1 + e^{-x}}) = f(x)(1 - f(x))$
- Sigmoid 激活函数的函数图像
    
    ![Sigmoid激活函数的函数图像](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day3/Sigmoid%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E7%9A%84%E5%87%BD%E6%95%B0%E5%9B%BE%E5%83%8F.png)
    
    Sigmoid激活函数的函数图像
    
- Sigmoid 函数可以将**任意的输入**映射到 **(0, 1)** 之间，当输入的值大致在 **<-6 或者 >6** 时，意味着输入任何值得到的激活值都是差不多的，这样会丢失部分的信息。
    - 比如：输入 100 和输出 10000 经过 Sigmoid 的激活值几乎都是等于 1 的，但是输入的数据之间相差 100 倍的信息就丢失了。
- 对于 Sigmoid 函数而言，输入值在 **[-6, 6]** 之间输出值才会有**明显差异**，输入值在 **[-3, 3]** 之间才会**有比较好的效果**。
- 通过上述导数图像，我们发现**导数数值范围是 (0, 0.25)**，当输入 <-6 或者 >6 时，sigmoid 激活函数图像的**导数接近为 0**，此时**网络参数将更新极其缓慢**，**或者无法更新**。
- 一般来说， sigmoid 网络在 **5 层之内就**会产生**梯度消失**现象。而且，该激活函数并不是以 0 为中心的，所以在实践中这种激活函数使用的很少。**sigmoid函数一般只用于二分类的输出层**。

```python
import torch
import matplotlib.pyplot as plt

# 创建画布和坐标轴
_, axes = plt.subplots(1, 2)
# 函数图像
x = torch.linspace(-20, 20, 1000)
# 输入值x通过sigmoid函数转换成激活值y
y = torch.sigmoid(x)
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('Sigmoid 函数图像')

# 导数图像
x = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.sigmoid(x).sum().backward()
# x.detach()：输入值x的数值
# x.grad：计算梯度，求导
axes[1].plot(x.detach(), x.grad)
axes[1].grid()
axes[1].set_title('Sigmoid 导数图像')
plt.show()
```

- 案例演示：[**04_activation_sigmoid.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day03/04_activation_sigmoid.py)（激活函数_Sigmoid图解）

# 10_Tanh激活函数介绍

- Tanh 激活函数公式：$f(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$
- Tanh 激活函数求导公式：$f'(x) = (\frac{1 - e^{-2x}}{1 + e^{-2x}})' = 1 - f^2(x)$
- Tanh 激活函数的函数图像
    
    ![Tanh激活函数的函数图像](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day3/Tanh%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E7%9A%84%E5%87%BD%E6%95%B0%E5%9B%BE%E5%83%8F.png)
    
    Tanh激活函数的函数图像
    
- Tanh 函数将**输入映射到 (-1, 1) 之间**，图像以 0 为中心，在 0 点对称，当输入 大概<-3 或者 >3 时将被映射为 -1 或者 1。**其导数值范围 (0, 1)**，当输入的值大概 <-3 或者 > 3 时，其导数近似 0。
- 与 Sigmoid 相比，它是**以 0 为中心的**，且梯度相对于 Sigmoid 大，使得其收敛速度要比 Sigmoid 快，减少迭代次数。然而，从图中可以看出，Tanh 两侧的导数也为 0，同样会造成梯度消失。
- 若使用时可在隐藏层使用 Tanh 函数，在输出层使用 Sigmoid 函数。

```python
# 创建画布和坐标轴
_, axes = plt.subplots(1, 2)
# 函数图像
x = torch.linspace(-20, 20, 1000)
y = torch.tanh(x)
axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('Tanh 函数图像')
# 导数图像
x = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.tanh(x).sum().backward()
axes[1].plot(x.detach(), x.grad)
axes[1].grid()
axes[1].set_title('Tanh 导数图像')
plt.show()
```

- 案例演示：[**05_activation_tanh.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day03/05_activation_tanh.py)（激活函数_Tanh图解）

# 11_ReLU激活函数介绍

- ReLU 激活函数公式：$f(x) = max(0, x)$
- ReLU 激活函数求导公式：$f'(x) = 0 或 1$
- ReLU 激活函数的函数图像
    
    ![ReLU激活函数的函数图像](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day3/ReLU%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E7%9A%84%E5%87%BD%E6%95%B0%E5%9B%BE%E5%83%8F.png)
    
    ReLU激活函数的函数图像
    
- ReLU 激活函数将小于 0 的值映射为 0，而大于 0 的值则保持不变，它更加重视正信号，而忽略负信号，这种激活函数运算更为简单，能够提高模型的训练效率。
- 当 x<0 时，ReLU 导数为 0，而当 x>0 时，则不存在饱和问题。所以，ReLU 能够在 x>0 时保持梯度不衰减，从而缓解梯度消失问题。然而，随着训练的推进，部分输入会落入小于 0 区域，导致对应权重无法更新。这种现象被称为“神经元死亡”。
- ReLU 是目前最常用的激活函数。与 Sigmoid 相比，ReLU 的优势是：
    - 采用 Sigmoid 函数，计算量大（指数运算），反向传播求误差梯度时，计算量相对大，而采用 RelU 激活函数，整个过程的计算量节省很多。
    - Sigmoid 函数反向传播时，很容易就会出现梯度消失的情况，从而无法完成深层网络的训练。
    - RelU 会使一部分神经元的输出为 0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。

```python
# 创建画布和坐标轴
_, axes = plt.subplots(1, 2)
# 函数图像
x = torch.linspace(-20, 20, 1000)
y = torch.relu(x)axes[0].plot(x, y)
axes[0].grid()
axes[0].set_title('ReLU 函数图像')
# 导数图像
x = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.relu(x).sum().backward()
axes[1].plot(x.detach(), x.grad)
axes[1].grid()
axes[1].set_title('ReLU 导数图像')
plt.show()
```

- 案例演示：[**06_activation_relu.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day03/06_activation_relu.py)（激活函数_ReLU图解）

# 12_Softmax激活函数介绍

- Softmax 用于多分类过程中，它是二分类函数 Sigmoid 在多分类上的推广，目的是将多分类的结果以概率的形式展现出来。
- Softmax 激活函数公式：$softmax(z_i) = \frac{e^{z_i}}{\sum_je^{z_j}}$

![Softmax激活函数](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day3/Softmax%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.png)

Softmax激活函数

- Softmax 就是将网络输出的 logits 通过 softmax 函数，就映射成为 (0, 1) 的值，而这些值的累和为 1（满足概率的性质），那么我们将它理解成概率，选取概率最大（也就是值对应最大的）节点，作为我们的预测目标类别。

```python
import torch

scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
# dim = 0,按行计算
probabilities = torch.softmax(scores, dim=0)
print(probabilities)
```

- 其他常见的激活函数
    
    ![其他常见的激活函数](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day3/%E5%85%B6%E4%BB%96%E5%B8%B8%E8%A7%81%E7%9A%84%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.png)
    
    其他常见的激活函数
    
- 激活函数的选择方法
    - 对于隐藏层
        1. 优先选择 ReLU 激活函数
        2. 如果 ReLU 效果不好，那么尝试其他激活，如 Leaky ReLU 等。
        3. 如果你使用了 ReLU，需要注意一下 Dead ReLU 问题，避免出现 0 梯度从而导致过多的神经元死亡。
        4. 少用使用 Sigmoid 激活函数，可以尝试使用 Tanh 激活函数
    - 对于输出层
        1. 二分类问题选择 Sigmoid 激活函数
        2. 多分类问题选择 Softmax 激活函数
        3. 回归问题选择 identity 激活函数
- 案例演示：[**07_activation_softmax.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day03/07_activation_softmax.py)（激活函数_Softmax图解）