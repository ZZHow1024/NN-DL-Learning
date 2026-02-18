# 神经网络与深度学习笔记_Day2

@ZZHow(ZZHow1024)

参考课程：

【**黑马程序员AI大模型《神经网络与深度学习》全套视频课程，涵盖Pytorch深度学习框架、BP神经网络、CNN图像分类算法及RNN文本生成算法**】

[[**https://www.bilibili.com/video/BV1c5yrBcEEX**](https://www.bilibili.com/video/BV1c5yrBcEEX)]

# 01_今日内容大纲介绍

- 张量运算
- 张量运算函数
- 张量索引操作
- 张量形状操作
- 张量拼接操作
- 自动微分模块
- PyTorch 构建回归横型

# 02_张量与NumPy之间相互转换

- 张量转换为 NumPy 数组
    - 使用 `Tensor.numpy` 函数可以将张量转换为 ndarray 数组，但是共享内存，可以使用 copy 函数避免共享。
    
    ```python
    # 对象拷贝避免共享内存
    data_tensor = torch.tensor([2, 3, 4])
    # 使用张量对象中的 numpy 函数进行转换，通过 copy 方法拷贝对象
    data_numpy = data_tensor.numpy().copy()
    print(type(data_tensor))
    print(type (data_numpy))
    # 注意：data_tensor 和 data_numpy 此时不共享内存
    # 修改其中的一个，另外一个不会发生改变
    # data_tensor[0] = 100
    data_numpy[0] = 100
    print(data_tensor)
    print(data_numpy)
    ```
    
- NumPy 数组转换沩张量
    - 使用 `from_numpy` 可以将 ndarray 数组转换为Tensor，默认共享内存，使用 copy 函数避免共享。
        
        ```python
        data_numpy = np.array([2, 3, 4])
        # 将 numpy 数组转换为张量类型
        # 1. torch.from_numpy(ndarray)
        data_tensor = torch.from_numpy(data_numpy)
        # nunpy 和 tensor 共享内存
        # data_numpy[0] = 100
        data_tensor[0] = 100
        print(data_tensor)
        print(data_numpy)
        ```
        
    - 使用 `torch.tensor` 可以将 ndarray 数组转换为 Tensor，默认不共享内存。
        
        ```python
        # 2. torch.tensor(ndarray)
        data_numpy = np.array([2, 3, 4]) 
        data_tensor = torch.tensor(data_numpy) 
        # numpy 和 tensor 不共享内存
        # data_numpy[0] = 100
        data_tensor[0] = 100
        print(data_tensor)
        print(data_numpy)
        ```
        
- 标量张量和数字转换
    - 对于只有一个元素的张量，使用 `item()` 函数将该值从张量中提取出来。
    
    ```python
    # 当张量只包含一个元素时，可以通过 item() 函数提取出该值
    data = torch.tensor([39, ])
    print(data.item())
    data = torch.tensor(30)
    print(data.item())
    ```
    
- 案例演示：[**01_tensor_numpy_conversion.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day02/01_tensor_numpy_conversion.py)（张量和numpy之间相互转换）

# 03_张量的基本运算

- 加减乘除取负号
    - add、sub、mul、div、neg
    - add_、sub_、mul_、div_、neg_（其中带下划线的版本会修改原数据）
- 案例演示：[**02_tensor_basic_operations.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day02/02_tensor_basic_operations.py)（张量的基本运算）

# 04_张量点乘和矩阵乘法

- 点乘运算
    - 点乘指 (Hadamard) 的是相同形状的张量对应位置的元素相乘，使用 mul 和运算符 * 实现。
    - 例如：
        
        $A = \begin{bmatrix}
           1 & 2 \\
           3 & 4
        \end{bmatrix}$, $B = \begin{bmatrix}
           5 & 6 \\
           7 & 8
        \end{bmatrix}$
        
        则 $A$, $B$ 的 Hadamard 积：$A · B = \begin{bmatrix}
           1 × 5 & 2 × 6 \\
           3 × 7 & 4 × 8
        \end{bmatrix} = \begin{bmatrix}
           5 & 12 \\
           21 & 32
        \end{bmatrix}$
        
- 矩阵乘法运算
    - 矩阵乘法运算要求第一个矩阵 shape：$(n, m)$，第二个矩阵 shape：$(m, p)$，两个矩阵点积运算 shape 为：$(n, p)$。
        1. 运算符 @ 用于进行两个矩阵的乘积运算。
        2. `torch.matmul` 对进行乘积运算的两矩阵形状没有限定。对于输入的 shape 不同的张量，对应的最后几个维度必须符合矩阵运算规则。
- 案例演示：[**03_tensor_dot_matmul.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day02/03_tensor_dot_matmul.py)（张量的点乘和矩阵乘法）

# 05_张量的常用运算函数

- PyTorch 为每个张量封装很多实用的计算函数
    - 均值：`mean()`
    - 平方根：`sqrt()`
    - 求和：`sum()`
    - 指数计算：`pow()`
    - 对数计算：`log()`、`log2()`、`log10()`
    - 等等
- 案例演示：[**04_tensor_utility_functions.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day02/04_tensor_utility_functions.py)（张量的常用运算函数）

# 06_张量的索引操作

- 索引操作
    - 我们在操作张量时，经常需要去获取某些元素就进行处理或者修改操作，在这里我们需要了解在 torch 中的索引操作。
    - 准备数据
        
        ```python
        import torch
        # 随机生成数据
        data = torch.randint(0, 10, [4, 5])
        print(data)
        ```
        
    - 简单行列
        
        ```python
        print(data[0])
        print(data[:, 0])
        ```
        
    - 列表索引
        
        ```python
        # 返回(0, 1)、(1, 2)两个位置的元素
        print(data[[O, 1], [1, 2]])
         
        # 返回 0、1 行的 1、2 列共 4 个元素
        print(data[[[O], [1]], [1, 2]])
        ```
        
    - 范围索引
        
        ```python
        # 前 3 行的前 2 列数据
        print(data[:3, :2])
        
        # 第 2 行到最后的前 2 列数据
        print(data[2:, :2])
        ```
        
    - 布尔索引
        
        ```python
        # 第三列大于 5 的行数据
        print(data[data[:, 2] > 5])
        
        # 第二行大于 5 的列数据
        print(data[:, data[1] > 5])
        ```
        
    - 多维索引
        
        ```python
        data = torch.randint(0, 10, [3, 4, 5])
        print(data)
        # 获取 0 轴上的第一个数据
        print(data[O, :, :]) 
        # 获取 1 轴上的第一个数据
        print(data[:, 0, :])
        # 获取 2 轴上的第一个数据
        print(data[:, :, 0])
        ```
        
- 案例演示：[**05_tensor_indexing.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day02/05_tensor_indexing.py)（张量的索引操作）

# 09-12_张量的形状操作

- `reshape()` 在不改变张量内容的前提下，对其形状做改变
- `unsqueeze()` 在指定的轴上增加一个(1)维度，等价于：升维
- `squeeze()` 删除所有为 1 的维度，等价于：降维
- `transpose()` 一次只能交换 2 个维度
- `permute()` 一次可以同时交换多个维度
- `view()` 只能修改连续的张量的形状，连续张量=内存中存储顺序和在张量中显示的顺序相同
- `contiguous()` 把不连续的张量 -> 连续的张量，即：基于张量中显示的顺序，修改内存中的存储顺序
- `is_contiguous()` 判断张量是否是连续的
- 案例演示：[**06_tensor_reshaping.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day02/06_tensor_reshaping.py)（张量的形状操作）

# 13_张量的拼接操作

- `torch.cat()` 函数可以将多个张量根据指定的维度拼接起来，**不改变维度数**。
- `torch.stack()` 函数会在一个新的维度上连接一系列张量，这会**增加一个新维度**，并且所有输入张量的形状必须完全相同。
- 案例演示：[**07_tensor_concatenation.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day02/07_tensor_concatenation.py)（张量的拼接）

# 14_自动微分模块

- 训练神经网络时，最常用的算法就是反向传播。在该算法中，参数（模型权重）会根据损失函数关于对应参数的梯度进行调整。为了计算这些梯度，PyTorch 内置了名 torch.autograd 的微分模块。它支持任意计算图的自动梯度计算。
    
    ![自动微分模块](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day2/%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E6%A8%A1%E5%9D%97.png)
    
    自动微分模块
    

# 15_自动微分模块案例_更新一次参数

- 梯度基本计算
    - PyTorch 不支持向量张量对向量张量的求导，只支持标量张量对向量张量的求导
        - x 如果是张量，y 必须是标量（一个值）才可以进行求导
    - 计算梯度：y.backvard()，**y 是一个标量**
    - 获取 x 点的梯度值：x.grad，**会累加上一次的梯度值**
- 梯度基本计算
    - 案例演示：[**08_autograd_introduction.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day02/08_autograd_introduction.py)（自动微分模块入门案例）

# 16_自动微分模块案例_循环更新参数

- 梯度下降法求最优解
    - 梯度下降法公式：w = w - r * grad（r 是学习率，grad 是梯度值）
    - 清空上一次的梯度值：`x.grad.zero_()`
- 案例演示：[**09_autograd_examples.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day02/09_autograd_examples.py)（自动微分模块案例）