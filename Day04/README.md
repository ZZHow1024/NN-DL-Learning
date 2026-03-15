# 神经网络与深度学习笔记_Day4

@ZZHow(ZZHow1024)

参考课程：

【**黑马程序员AI大模型《神经网络与深度学习》全套视频课程，涵盖Pytorch深度学习框架、BP神经网络、CNN图像分类算法及RNN文本生成算法**】

[[**https://www.bilibili.com/video/BV1c5yrBcEEX**](https://www.bilibili.com/video/BV1c5yrBcEEX)]

# 01_参数初始化_介绍

- 我们在构建网络之后，网络中的参数是需要初始化的。我们需要初始化的参数主要有**权重**和**偏置**，**偏置一般初始化为 0 即可**，而对权重的初始化则会更加重要。
- 参数初始化的作用
    - **防止梯度消失或爆炸**：初始权重值过大或过小会导致梯度在反向传播中指数级增大或缩小。
    - **提高收敛速度**：合理的初始化使得网络的激活值分布适中，有助于梯度高效更新。
    - **保持对称性破除**：权重的初始化需要打破对称性，否则网络的学习能力会受到限制。
- 参数初始化
    - 随机初始化
        - 均匀分布初始化 (`nn.init.uniform_()`)：权重参数初始化从区间均匀随机取值，默认区间为 $(0，1）$。可以设置为在 $(-\frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}})$ 均匀分布中生成当前神经元的权重，其中 $d$ 为神经元的输入数量。
        - 正态分布初始化 (`nn.init.normal_()`)：随机初始化从均值为 0，标准差是 1 的高斯分布中取样，使用一些很小的值对参数 W 进行初始化。
        - **优点**：能有效打破对称性。
        - **缺点**：随机选择范围不当可能导致梯度问题。
        - **适用场景**：浅层网络或低复杂度模型。隐藏层 1-3 层，总层数不超过 5 层。
    - 全 0 初始化 (`nn.init.zeros_()`)：将神经网络中的所有权重参数初始化为 0。
        - **优点**：实现简单。
        - **缺点**：无法打破对称性，所有神经元更新方向相同，无法有效训练。
        - **适用场景**：几乎不使用，仅用于偏置项的初始化。
    - 全 1 初始化 (`nn.init.ones_()`)：将神经网络中的所有权重参数初始化为 1。
        - **优点**：实现简单。
        - **缺点**
            - 无法打破对称性，所有神经元更新方向相同，无法有效训练。
            - 会导致激活值在网络中呈指数增长，容易出现梯度爆炸。
        - **适用场景**
            - 测试或调试：比如验证神经网络是否能正常前向传播和反向传播。
            - 特殊模型结构：某些稀疏网络或特定的自定义网络中可能需要手动设置部分参数为 1。
            - 偏置初始化：偶尔可以将偏置初始化为小的正值（如 0.1），但很少用 1 作为偏置的初始值。
    - 固定值初始化 (`nn.init.constant_()`)：将神经网络中的所有权重参数初始化为某个固定值。
        - **优点**：实现简单。
        - **缺点**
            - 无法打破对称性，所有神经元更新方向相同，无法有效训练。
            - 初始权重过大或过小可能导致梯度爆炸或梯度消失。
        - **适用场景：**测试或调试。
    - kaiming 初始化，也叫做 HE 初始化：HE 初始化分为正态分布的 HE 初始化、均匀分布的 HE 初始化。
        - 正态分布的 HE 初始化 (`nn.init.kaiming_normal_()`)：它是从 $[0, std]$ 中抽取样本的，$std = sqrt(\frac{2}{fan_in})$。
        - 均匀分布的 HE 初始化 (`nn.init.kaiming_uniform_()`)：它从 $[-limit，limit]$ 中的均匀分布中抽取样本，limit 是 $sqrt(\frac{6}{fan\_in})$。
        - $fan\_in$ 输入层神经元的个数。
        - **优点**：适合 ReLU，能保持梯度稳定。
        - **缺点**：对非 ReLU 激活函数效果一般。
        - **适用场景**：深度网络（10 层及以上），使用 ReLU、Leaky ReLU 激活函数。
    - Xavier 初始化，也叫做 Glorot 初始化：该方法也有两种，一种是正态分布的 Xavier 初始化、一种是均匀分布的 Xavier 初始化。
        - 正态化的 Xavier 初始化 (`nn.init.xavier_normal_()`)：它是从 $[0, std]$ 中抽取样本的，$std = sqrt(\frac{2}{fan\_in + fan\_out)}$。
        - 均匀分布的 Xavier 初始化 (`nn.init.xavier_uniform_()`)：$[-limit，limit]$ 中的均匀分布中抽取样本，limit 是 $sqrt(\frac{6}{fan\_in + fan\_out)}$。
        - $fan\_in$ 是输入层神经元的个数，$fan\_out$ 是输出层神经元个数。
        - **优点**：适用于 Sigmoid、Tanh 等激活函数，解决梯度消失问题。
        - **缺点**：对 ReLU 等激活函数表现欠佳。
        - **适用场景**：深度网络（10 层及以上），使用 Sigmoid 或 Tanh 激活函数。

# 02_参数初始化_代码演示

- 参数初始化选择
    - 激活函数的选择：根据激活函数的类型选择对应的初始化方法。
        - Sigmoid / Tanh：Xavier 初始化。
        - ReLU / Leaky ReLU：Kaiming 初始化。
    - 网络的深度
        - 浅层网络：随机初始化即可。
        - 深层网络：需要考虑方差平衡，如 Xavier 或 Kaiming 初始化。
- 案例演示：[**01_parameter_initialization.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day04/01_parameter_initialization.py)（参数初始化介绍）

# 03_神经网络_搭建流程介绍

- 神经网络搭建和参数计算
    - 在 PyTorch 中定义深度神经网络其实就是层堆叠的过程，继承自 nn.Module，实现两个方法：
        - `__init__` 方法中定义网络中的层结构，主要是全连接层，并进行初始化。
        - `forward` 方法，在实例化模型的时候，底层会自动调用该函数。该函数中为初始化定义的 layer 传入数据，进行前向传播等。

# 04_神经网络_搭建代码实现

- 构建如下图所示的神经网络模型
    
    ![神经网络模型](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day4/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B.png)
    
    神经网络模型
    
    - 编码设计如下：
        1. 第 1 个隐藏层：权重初始化采用标准化的 Xavier 初始化 激活函数使用 Sigmoid。
        2. 第 2 个隐藏层：权重初始化采用标准化的 HE 初始化 激活函数采用 ReLU。
        3. out 输出层线性层假若多分类，采用 Softmax 做数据归一化。
- 案例演示：[**02_neural_network_construction.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day04/02_neural_network_construction.py)（神经网络搭建）

# 05_神经网络_模型训练

- 案例演示：[**02_neural_network_construction.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day04/02_neural_network_construction.py)（神经网络搭建）

# 06_神经网络_总结

- 神经网络搭建和参数计算
    - 神经网络的输入数据是为 $[batch_size, in_features]$ 的张量经过网络处理后获取了 $[batch_size, out_features]$ 的输出张量。
    - 在上述例子中，batchsize=5, infeatures=3, out_features=2，结果如下所示：
        - mydata.shape → torch.Size([5, 3])
        - output.shape → torch.Size([5, 2])
    - 模型参数的计算：
        1. 以第一个隐层为例：该隐层有 3 个神经元，每个神经元的参数为：4 个 $(w1, w2, w3, b1)$，所以一共用 $3 × 4 = 12$ 个参数。
        2. **输入数据**和**网络权重**是两个不同的事！
- 神经网络的优缺点
    - 优点
        - 精度⾼，性能优于其他的机器学习⽅法，甚⾄在某些领域超过了⼈类。
        - 可以近似任意的⾮线性函数。
        - 近年来在学界和业界受到了热捧，有⼤量的框架和库可供调。
    - 缺点
        - ⿊箱，很难解释模型是怎么⼯作的。
        - 训练时间⻓，需要⼤量的计算⼒。
        - ⽹络结构复杂，需要调整超参数。
        - ⼩数据集上表现不佳，容易发⽣过拟合。

# 07_损失函数_多分类交叉熵损失介绍

- 什么是损失函数：在深度学习中，损失函数是**用来衡量模型参数的质量的函数**，衡量的方式是比较网络输出和真实输出的差异。
- 损失函数在不同的文献中名称是不一样的，主要有以下几种命名方式：
    - 损失函数 (loss function)
    - 代价函数 (cost function)
    - 目标函数 (objective function)
    - 误差函数 (error function)
- 多分类任务损失函数
    - 在多分类任务通常使用 Softmax 将 logits 转换为概率的形式，所以多分类的交叉熵损失也叫做 Softmax 损失，它的计算方法是：$L = -\sum^n_{i=1}y_ilog(S(f_\theta(x_i)))$
    - 其中：
        - $y$ 是样本 $x$ 属于某一个类别的真实概率。
        - 而 $f(x)$ 是样本属于某一类别的预测分数。
        - $S$ 是 $Softmax$ 激活函数,将属于某一类别的预测分数转换成概率。
        - $L$ 用来衡量真实值 $y$ 和预测值 $f(x)$ 之间差异性的损失结果。
    
    ![多分类任务](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day4/%E5%A4%9A%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1.png)
    
    多分类任务
    
    - 上图中的交叉熵损失为：$-(0log(0.10) + 1log(0.7) + 0log(0.2)) = -log0.7$
    - 从概率角度理解，我们的目的是最小化正确类别所对应的预测概率的对数的负值（损失值最小）。
- 在 PyTorch 中使用 `nn.CrossEntropyLoss()` 实现。
- 案例演示：[**03_cross_entropy_loss.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day04/03_cross_entropy_loss.py)（多分类任务交叉熵损失函数）

# 08_损失函数_二分类交叉熵损失介绍

- 在处理二分类任务时，我们不再使用 Softmax 激活函数，而是使用 Sigmoid 激活函数，那损失函数也相应的进行调整，使用二分类的交叉熵损失函数：$L = -ylog\hat{y} - (1 - y)log(1 - \hat{y})$
- 其中：
    - $y$ 是样本 $x$ 属于某一个类别的真实概率。
    - 而 $\hat{y}$ 是样本属于某一类别的预测概率。
    - $L$ 用来衡量真实值 $y$ 与预测值 $\hat{y}$ 之间差异性的损失结果。
- 在 PyTorch 中实现时使用 `nn.BCELoss()`。
- 案例演示：[**04_binary_cross_entropy_loss.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day04/04_binary_cross_entropy_loss.py)（二分类任务损失函数）

# 09_损失函数_MAE损失函数介绍

- Mean absolute loss(MAE)也被称为 L1 Loss，是以绝对误差作为距离。
- 损失函数公式：$L = \frac{1}{n}\sum^n_{i=1}|y_i - f_\theta(x_i)|$
- 特点是：
    1. 由于 L1 loss 具有稀疏性，为了惩罚较大的值，因此常常将其作为正则项添加到其他 loss 中作为约束。
    2. L1 loss 的最大问题是梯度在零点不平滑，导致会跳过极小值。
- 在 PyTorch 中使用 `nn.L1Loss()` 实现。
- 案例演示：[**05_regression_loss_functions.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day04/05_regression_loss_functions.py)（回归任务_损失函数介绍）

# 10_损失函数_MSE损失函数介绍

- Mean Squared Loss/ Quadratic Loss(MSE loss)也被称为L2 Loss，或欧氏距离，它以误差的平方和的均值作为距离。
- 损失函数公式：$L = \frac{1}{n}\sum^n_{i=1}(y_i - f_\theta(x_i))^2$
- 特点：
    - L2 Loss 也常常作为正则项。
    - 当预测值与目标值相差很大时，梯度容易爆炸。
- 在 PyTorch 中使用 `nn.MSELoss()` 实现。
- 案例演示：[**05_regression_loss_functions.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day04/05_regression_loss_functions.py)（回归任务_损失函数介绍）

# 11_损失函数_Smooth L1损失函数介绍

- Smooth L1 说的是光滑之后的 L1。
- 损失函数公式：$smooth_{l_1}(x) = \begin{cases}
   0.5x^2 &\text{if } |x| < 1 \\
   |x| - 0.5 &\text{otherwise}
\end{cases}$
- 其中：$𝑥=f(x)−y$ 为真实值和预测值的差值。
- L1、L2 与 SmoothL1 图像
    
    ![L1L2与SmoothL1图像](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day4/L1L2%E4%B8%8ESmoothL1%E5%9B%BE%E5%83%8F.png)
    
    L1L2与SmoothL1图像
    
- 该函数实际上就是一个分段函数
    - 在 $[-1, 1]$ 之间实际上就是 L2 损失，这样解决了 L1 的不光滑问题。
    - 在 $[-1, 1]$ 区间外，实际上就是 L1 损失，这样就解决了离群点梯度爆炸的问题。
- 在 PyTorch 中使用 `nn.SmoothL1Loss()` 实现。
- 案例演示：[**05_regression_loss_functions.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day04/05_regression_loss_functions.py)（回归任务_损失函数介绍）

# 12_梯度下降算法回顾

- 梯度下降法是一种寻找使损失函数最小化的方法。从数学角度来看，梯度的方向是函数增长速度最快的方向，那么梯度的反方向就是函数减少最快的方向，所以有：$w^{new}_{ij} = w^{old}_{ij} - \eta\frac{∂E}{∂w_{ij}}$
- 其中，$η$ 是学习率，如果学习率太小，那么每次训练之后得到的效果都太小，增大训练的时间成本。如果学习率太大，那就有可能直接跳过最优解，进入无限的训练中。解决的方法就是，学习率也需要随着训练的进行而变化。
- 在进行模型训练时，有三个基础的概念：
    - **Epoch**：使用全部数据对模型进行一次完整训练，训练轮次。
    - **Batch_size**：使用训练集中的小部分样本对模型权重进行以此反向传播的参数更新，每次训练每批次样本数量。
    - **Iteration**：使用一个 Batch 数据对模型进行一次参数更新的过程。
- 假设数据集有 50000 个训练样本，现在选择 Batch Size = 256 对模型进行训练。
    - 每个 Epoch 要训练的图片数量：50000
    - 训练集具有的 Batch 个数：50000 / 256 + 1 = 196
    - 每个 Epoch 具有的 Iteration 个数：196
    - 10个 Epoch 具有的 Iteration 个数：1960
- 在深度学习中，梯度下降的几种方式的根本区别就在于 Batch Size 不同。
    
    
    | 梯度下降方式 | Training Set Size | Batch Size | Number of Batches |
    | --- | --- | --- | --- |
    | BGD（全梯度下降） | N | N | 1 |
    | SGD（随机梯度下降） | N | 1 | N |
    | Mini-Batch（小批量梯度下降） | N | B | N / B + 1 |
    - 注：上表中 Mini-Batch 的 Batch 个数为 N / B + 1 是针对未整除的情况。整除则是 N / B。

# 13_反向传播

- **前向传播**：指的是数据输入到神经网络中，逐层向前传输，一直运算到输出层为止。
- **反向传播（Back Propagation）**：利用损失函数 ERROR 值，从后往前，结合梯度下降算法，依次求各个参数的偏导，并进行参数更新。
    
    ![前向传播与反向传播](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day4/%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%E4%B8%8E%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD.png)
    
    前向传播与反向传播
    
- 反向传播算法利用链式法则对神经网络中的各个节点的权重进行更新。