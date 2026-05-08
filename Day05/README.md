# 神经网络与深度学习笔记_Day5

@ZZHow(ZZHow1024)

参考课程：

【**黑马程序员AI大模型《神经网络与深度学习》全套视频课程，涵盖Pytorch深度学习框架、BP神经网络、CNN图像分类算法及RNN文本生成算法**】

[[**https://www.bilibili.com/video/BV1c5yrBcEEX**](https://www.bilibili.com/video/BV1c5yrBcEEX)]

# 01_梯度相关知识点回顾

- 见 Day4

# 02_指数移动加权平均简介

- 梯度下降的优化方法
    - 梯度下降优化算法中，可能会碰到以下情况：
        1. 碰到平缓区域，梯度值较小，参数优化变慢。
        2. 碰到 “鞍点” ，梯度为 0，参数无法优化。
        3. 碰到局部最小值，参数不是最优。
    - 对于这些问题，出现了一些对梯度下降算法的优化方法，例如：Momentum、AdaGrad、RMSprop、Adam 等。
    
    ![梯度下降优化算法问题](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day5/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%E9%97%AE%E9%A2%98.png)
    
    梯度下降优化算法问题
    
- 指梯度下降的优化方法-指数加权平均
    - **指数移动加权平均**则是参考各数值，并且各数值的权重都不同，距离越远的数字对平均数计算的贡献就越小（权重较小），距离越近则对平均数的计算贡献就越大（权重越大）。
    - 比如：明天气温怎么样，和昨天气温有很大关系，而和一个月前的气温关系就小一些。
    - 计算公式可以用下面的式子来表示：$S_t = \begin{cases}
       Y_1 &\text{t = 0} \\
       β × S_{t-1} + (1 - β) × Y_t &\text{t > 0}
    \end{cases}$
        - $S_t$ 表示指数加权平均值。
        - $Y_t$ 表示 $t$ 时刻的值。
        - $β$ 调节权重系数，该值越大平均数越平缓。
    - 对比 $β$ 为 0.5 和 0.9 时的结果，从中可以看出：
        - 指数加权平均绘制出的气温变化曲线更加平缓。
        - $β$ 的值越大，则绘制出的折线越加平缓，波动越小。（$1-β$ 越小，$t$ 时刻的 $S_t$ 越不依赖 $Y_t$ 的值）。
        - $β$ 值一般默认都是 0.9。
- 案例演示：[**01_exponential_moving_average.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/01_exponential_moving_average.py)（演示指数移动加权平均）

# 03_梯度下降优化方法_动量法

- 梯度计算公式：$S_t = β × S_{t−1} + (1−β) × G_t$
- 参数更新公式：$W_t = W_{t − 1} − ηS_t$
    - $s_t$ 是当前时刻指数加权平均梯度值。
    - $s_{t-1}$ 是历史指数加权平均梯度值。
    - $g_t$ 是当前时刻的梯度值。
    - $β$ 是调节权重系数，通常取 0.9 或 0.99。
    - $η$ 是学习率。
    - $w_t$ 是当前时刻模型权重参数。
- 假设：权重 $β$ 为 0.9，例如：
    - 第一次梯度值：$S_1 = G_1 = W_1$
    - 第二次梯度值：$S_2 = 0.9 × S_1 + G_2 × 0.1$
    - 第三次梯度值：$S_3 = 0.9 × S_2 + G_3 × 0.1$
    - 第四次梯度值：$S_4 = 0.9 × S_3 + G_4 × 0.1$
        1. $W$ 表示初始梯度。
        2. $G$ 表示当前轮数计算出的梯度值。
        3. $S$ 表示历史梯度移动加权平均值。
- 梯度下降公式中梯度的计算，就不再是当前时刻t的梯度值，而是历史梯度值的指数移动加权平均值。
    - 公式修改为：$W_t = W_{t-1} - η × S_t$
        - $W_t$：当前时刻模型权重参数。
        - $S_t$：当前时刻指数加权平均梯度值。
        - $η$：学习率。
- Monmentum 优化方法在一定程度上克服 “平缓”、”鞍点” 的问题
    - 当处于鞍点位置时，由于当前的梯度为 0，参数无法更新。但是 Momentum 动量梯度下降算法已经在先前积累了一些梯度值，很有可能使得跨过鞍点。
    - 由于 mini-batch 普通的梯度下降算法，每次选取少数的样本梯度确定前进方向，可能会出现震荡，使得训练时间变长。Momentum 使用移动加权平均，平滑了梯度的变化，使得前进方向更加平缓，有利于加快训练过程。
- 案例演示：[**02_gradient_descent_optimization.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/02_gradient_descent_optimization.py)（梯度下降优化方法）

# 04_梯度下降优化方法_AdaGrad

- AdaGrad 通过对不同的参数分量使用不同的学习率，**AdaGrad 的学习率总体会逐渐减小**。
    - 其计算步骤如下：
        1. 初始化学习率 $η$、初始化参数 $W$、小常数 $σ = 1e-10$
        2. 初始化梯度累计变量 $S = 0$
        3. 从训练集中采样 $m$ 个样本的小批量，计算梯度 $G_t$
        4. 累积平方梯度：$S_t = S_{t-1} + G_t ⊙ G_t$，$⊙$ 表示各个分量相乘
        5. 学习率 $η$ 的计算公式如下：$η = \frac{η}{\sqrt{S_t} + σ}$
        6. 权重参数更新公式如下：$W_t = W_{t-1} - \frac{η}{\sqrt{S_t} + σ} × G_t$
        7. 重复 3-7 步骤
    - **AdaGrad 缺点是可能会使得学习率过早、过量的降低，导致模型训练后期学习率太小，较难找到最优解**。
- 案例演示：[**02_gradient_descent_optimization.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/02_gradient_descent_optimization.py)（梯度下降优化方法）

# 05_梯度下降优化方法_RMSProp

- **RMSProp 优化算法是对 AdaGrad 的优化**。最主要的不同是其使用**指数加权平均梯度**替换历史梯度的平方和。
- 其计算过程如下：
    1. 初始化学习率 $η$、初始化权重参数 $W$、小常数 $σ = 1e-10$
    2. 初始化梯度累计变量 $S = 0$
    3. 从训练集中采样 $m$ 个样本的小批量，计算梯度 $G_t$
    4. 使用指数加权平均累计历史梯度，$⊙$ 表示各个分量相乘，公式如下：$S_t = β × S_{t-1} + (1 - β) G_t ⊙ G_t$
    5. 学习率 $η$ 的计算公式如下：$η = \frac{η}{\sqrt{S_t} + σ}$
    6. 权重参数更新公式如下：$W_t = W_{t-1} - \frac{η}{\sqrt{S_t} + σ} × G_t$
    7. 重复 3-7 步骤
- 案例演示：[**02_gradient_descent_optimization.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/02_gradient_descent_optimization.py)（梯度下降优化方法）

# 08_梯度下降优化方法_Adam

- Momentum 使用指数加权平均计算当前的梯度值。
- AdaGrad、RMSProp 使用自适应的学习率。
- Adam 优化算法（Adaptive Moment Estimation，自适应矩估计）将 Momentum 和 RMSProp 算法结合在一起。
    - **修正梯度**：使⽤梯度的指数加权平均。
    - **修正学习率**：使⽤梯度平⽅的指数加权平均。
- 原理：Adam 是结合了 **Momentum** 和 **RMSProp** 优化算法的优点的自适应学习率算法。它计算了梯度的一阶矩（平均值）和二阶矩（梯度的方差）的自适应估计，从而动态调整学习率。
- 梯度计算公式：
    - $m_t = β_1 m_{t-1} + (1 - β_1) g_t$
    - $s_t = β_2s_{t-1} + (1 - β_2)gt^2$
    - $\hat{m}_t = \frac{m_t}{1 - β^t_1}$, $\hat{s_t} = \frac{s_t}{1 - β^t_2}$
- 权重参数更新公式：$w_t = w_{t-1} - \frac{η}{\sqrt{\hat{s_t}} + σ}\hat{m_t}$
- 其中，$m_t$ 是梯度的一阶矩估计，$s_t$ 是梯度的二阶矩估计，$\hat{m_t}$ 和 $\hat{s_t}$ 是偏差校正后的估计。
- 案例演示：[**02_gradient_descent_optimization.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/02_gradient_descent_optimization.py)（梯度下降优化方法）

# 09_梯度下降优化方法_总结

- 梯度下降算法优化的**目的**：梯度下降优化算法中，可能会碰到平缓区域，“鞍点”等问题。
- 梯度下降算法的优化**有哪些**：动量法，AdaGrad，RMSProp，Adam。
- **如何选择**梯度下降优化方法
    - 简单任务和较小的模型：SGD，动量法
    - 复杂任务或者有大量数据：Adam
    - 需要处理稀疏数据或者文本数据：AdaGrad，RMSProp

# 10_学习率优化_背景介绍

- 为什么要进行学习率优化：在训练神经网络时，一般情况下学习率都会随着训练而变化。这主要是由于，在神经网络训练的后期，如果**学习率过高，会造成loss的振荡**，但是如果**学习率减小的过慢，又会造成收敛变慢**的情况。
- 结论：
    - 采用较小的学习率，梯度下降的速度慢
    - 采用较大的学习率，梯度下降太快越过了最小值点，导致震荡，甚至不收敛（梯度爆炸）。
- 各学习率的收敛情况
    
    ![各学习率的收敛情况](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day5/%E5%90%84%E5%AD%A6%E4%B9%A0%E7%8E%87%E7%9A%84%E6%94%B6%E6%95%9B%E6%83%85%E5%86%B5.png)
    
    各学习率的收敛情况
    
- 案例演示：[**03_learning_rate_impact.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/03_learning_rate_impact.py)（演示学习率对梯度的影响）

# 11_学习率衰减策略_等间隔学习率衰减

- API
    
    ```python
    # optimizer：优化器对象
    # step_size：调整间隔数=50
    # gamma：调整系数=0.5
    # 调整方式：lr = lr * gamma
    lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
    ```
    
- 案例演示：[**04_learning_rate_scheduling.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/04_learning_rate_scheduling.py)（演示学习率衰减策略）

# 12_学习率衰减策略_指定间隔学习率衰减

- API
    
    ```python
    # optimizer：优化器对象
    # milestones：设定调整轮次:[50, 125, 160]
    # gamma：调整系数=0.5
    # 调整方式：lr = lr * gamma
    optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    ```
    
- 案例演示：[**04_learning_rate_scheduling.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/04_learning_rate_scheduling.py)（演示学习率衰减策略）

# 13_学习率衰减策略_指数间隔学习率衰减

- API
    
    ```python
    # optimizer：优化器对象
    # gamma：指数的底
    # 调整方式：lr = lr ∗ gamma ** epoch
    optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    ```
    
- 案例演示：[**04_learning_rate_scheduling.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/04_learning_rate_scheduling.py)（演示学习率衰减策略）

# 14_学习率衰减策略_总结

- 学习率衰减策略对比
    
    
    | **方法** | **等间隔学习率衰减 (Step Decay)** | **指定间隔学习率衰减 (Exponential Decay)** | **指数学习率衰减 (Exponential Moving Average Decay)** |
    | --- | --- | --- | --- |
    | **衰减方式** | 固定步长衰减 | 指定步长衰减 | 平滑指数衰减，历史平均考虑 |
    | **实现难度** | 简单易实现 | 相对简单，容易调整 | 需要额外历史计算，较复杂 |
    | **适用场景** | 大型数据集、较为简单的任务 | 对训练平稳性要求较高的任务 | 高精度训练，避免过快收敛 |
    | **优点** | 直观，易于调试，适用于大批量数据 | 易于调试，稳定训练过程 | 平滑且考虑历史更新，收敛稳定性较强 |
    | **缺点** | 学习率变化较大，可能跳过最优点 | 在某些情况下可能衰减过快，导致优化提前停滞 | 超参数调节较为复杂，可能需要更多的计算资源 |

# 15_正则化_dropout(随机失活)介绍

- 正则化
    - 在设计机器学习算法时希望在新样本上的泛化能力强。许多机器学习算法都采用相关的策略来减小测试误差，这些策略被统称为**正则化**。
    - 神经网络强大的表示能力经常遇到过拟合，所以需要使用不同形式的正则化策略。
    - 目前在深度学习中使用较多的策略有**范数惩罚**、**DropOut**、**特殊的网络层**等。
- Dropout 正则化
    - 在神经网络中模型参数较多，在数据量不足的情况下，很容易过拟合。Dropout（随机失活）是一个简单有效的正则化方法。
    - 在训练过程中，Dropout 的实现是**让神经元以超参数p的概率停止工作或者激活被置为 0，未被置为 0 的进行缩放，缩放比例为** $\frac{1}{(1-p)}$。训练过程可以认为是对完整的神经网络的一些子集进行训练，每次基于输入数据只更新子网络的参数。
    - 在实际应用中，Dropout 参数 p 的概率通常取值在 0.2 到 0.5 之间
        - 对于较小的模型或较复杂的任务，丢弃率可以选择 0.3 或更小。
        - 对于非常深的网络，较大的丢弃率（如 0.5 或 0.6）可能会有效防止过拟合。
        - 实际应用中，通常会在全连接层（激活函数后）之后添加 Dropout 层。
    - **在测试过程中，随机失活不起作用**
        - 在测试阶段，使用所有的神经元进行预测，以获得更稳定的结果。
        - 直接使用训练好的模型进行测试，由于所有的神经元都参与计算，输出的期望值会比训练阶段高。测试阶段的期望输出是 E[x_test] = x。
        - 测试/推理模式：`model.eval()`。
    - **缩放的必要性**
        - 在训练阶段，将参与计算的神经元的输出除以 (1-p)。
        - 经过 Dropout 后的期望输出变为 E[x_dropout] = [(1-p) * x] / (1-p) = x，与测试阶段的期望输出一致。
        - 训练模型：`model.train()`。

# 16_正则化_dropout(随机失活)代码演示

- 案例演示：[**05_dropout_regularization.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/05_dropout_regularization.py)（Dropout 随机失活）

# 17_正则化_批量归一化(BN)介绍

- 先对数据标准化，再对数据重构（缩放+平移）。
- 公式：$f(x) = λ · \frac{x - E(x)}{\sqrt{Var(x)} + ε} + β$
    1. $λ$ 和 $β$ 是可学习的参数，它相当于对标准化后的值做了一个线性变换，λ 为系数，β 为偏置；
    2. $ε$ 通常指为 1e-5，避免分母为 0；
    3. $E(x)$ 表示变量的均值；
    4. $Var(x)$ 表示变量的方差；
- 批量归一化层在**计算机视觉领域**使用较多。
- 归一化的步骤：
    1. 计算均值和方差
    2. 标准化
    3. 缩放和平移
- 批量归一化的作用：
    - **减少内部协方差偏移**：通过对每层的输入进行标准化，减少了输入数据分布的变化，从而加速了训练过程，并使得网络在训练过程中更加稳定。
    - **加速训练**：
        - 在没有批量归一化的情况下，神经网络的训练通常会很慢，尤其是深度网络。因为在每层的训练过程中，输入数据的分布（特别是前几层）会不断变化，这会导致网络学习速度缓慢。
        - 批量归一化通过确保每层的输入数据在训练时分布稳定，有效减少了这种变化，从而加速了训练过程。
    - **起到正则化作用**：批量归一化可以视作一种正则化方法，因为它引入了对训练样本的噪声（不同批次的统计信息不同，批次较小的均值和方差估计会更加不准确），使得模型不容易依赖特定的输入特征，从而起到一定的正则化效果，减少了对其他正则化技术 (如 Dropout)的需求。
    - **提升泛化能力**：由于其正则化效果，批量归一化能帮助网络在测试集上取得更好的性能。

# 18_正则化_批量归一化(BN)代码实现

- 案例演示：[**06_batch_normalization.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/06_batch_normalization.py)（批量归一化）

# 19_ANN案例_手机价格分类_需求介绍

- 需求分析
    - 小明创办了一家手机公司，他不知道如何估算手机产品的价格。为了解决这个问题，他收集了多家公司的手机销售数据。该数据为二手手机的各个性能的数据，最后根据这些性能得到 4 个价格区间，作为这些二手手机售出的价格区间。
    - 我们需要帮助小明找出手机的功能（例如：RAM 等）与其售价之间的某种关系。我们可以使用机器学习的方法来解决这个问题，也可以构建一个全连接的网络。
    - 在这个问题中，我们不需要预测实际价格，而是一个价格范围，它的范围使用 0、1、2、3 来表示，所以该问题也是一个分类问题。
- 按照四个步骤来完成这个任务：
    1. 准备训练集数据
    2. 构建要使用的模型
    3. 模型训练
    4. 模型预测评估
- 案例演示：[**07_ann_phone_price_classification.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/07_ann_phone_price_classification.py)（ANN案例_手机价格分类案例）

# 20_ANN案例_手机价格分类_准备数据集

- 导入所需的工具包
    
    ```python
    # 导入相关模块
    import torch
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import time
    ```
    
- 构建数据集
    - 数据共有 2000 条，其中 1600 条数据作为训练集，400 条数据用作测试集。使用 sklearn 的数据集划分工作来完成。并使用 PyTorch 的 TensorDataset 来将数据集构建为 Dataset 对象，方便构造数据集加载对象。
    
    ```python
    # 构建数据集
    def create_dataset():
        # 使用pandas读取数据
        data = pd.read_csv('data/手机价格预测.csv')
        # 特征值和目标值
        x, y = data.iloc[:, :-1], data.iloc[:, -1]
        # 类型转换：特征值，目标值
        x = x.astype(np.float32)
        y = y.astype(np.int64)
        # 数据集划分
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=88)
        # 构建数据集，转换为 PyTorch 的形式
        train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.tensor(y_train.values))
        valid_dataset = TensorDataset(torch.from_numpy(x_valid.values), torch.tensor(y_valid.values))
        # 返回结果
        return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y))
    ```
    
- 获取数据的结果
    
    ```python
    if __name__ == '__main__':
    	# 获取数据
    	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
    	print("输入特征数：", input_dim)
    	print("分类个数：", class_num)
    ```
    
    - 输出结果为：
        
        ```
        输入特征数：20
        分类个数：4
        ```
        
- 案例演示：[**07_ann_phone_price_classification.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day05/07_ann_phone_price_classification.py)（ANN案例_手机价格分类案例）