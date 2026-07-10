# 神经网络与深度学习笔记_Day6

@ZZHow(ZZHow1024)

参考课程：

【**黑马程序员AI大模型《神经网络与深度学习》全套视频课程，涵盖Pytorch深度学习框架、BP神经网络、CNN图像分类算法及RNN文本生成算法**】

[[**https://www.bilibili.com/video/BV1c5yrBcEEX**](https://www.bilibili.com/video/BV1c5yrBcEEX)]

# 01_ANN案例_手机价格分类_准备数据集(回顾)

- 见 Day5

# 02_ANN案例_手机价格分类_搭建神经网络

- 构建分类网络模型
    - 构建全连接神经网络来进行手机价格分类，该网络主要由三个线性层来构建，使用 ReLU 激活函数。
    - 网络共有 3 个全连接层，具体信息如下：
        - 第一层：输入为维度为 20，输出维度为：128
        - 第二层：输入为维度为 128，输出维度为：256
        - 第三层：输入为维度为 256，输出维度为：4
        
        ```python
        # 构建网络模型
        class PhonePriceModel(nn.Module):
        	def __init__(self, input_dim, output_dim):
        		super(PhonePriceModel, self).__init__()
        		# 1. 第一层: 输入为维度为 20, 输出维度为: 128
        		self.linear1 = nn.Linear(input_dim, 128)
        		# 2. 第二层: 输入为维度为 128, 输出维度为: 256
        		self.linear2 = nn.Linear(128, 256)
        		# 3. 第三层: 输入为维度为 256, 输出维度为: 4
        		self.linear3 = nn.Linear(256, output_dim)
        
        		def forward(self, x):
        			# 前向传播过程
        			x = torch.relu(self.linear1(x))
        			x = torch.relu(self.linear2(x))
        			output = self.linear3(x)
        			# 获取数据结果
        			return output
        ```
        
    - 模型实例化：
        
        ```python
        if __name__ == '__main__':
        	# 模型实例化
        	model = PhonePriceModel(input_dim, class_num)
        	summary(model, input_size=(input_dim,), batch_size=16)
        ```
        
- 案例演示：[**01_ann_phone_price_classification.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day06/01_ann_phone_price_classification.py)（ANN案例_手机价格分类案例）

# 03_ANN案例_手机价格分类_模型训练

- 网络编写完成之后，需要编写训练函数。所谓的训练函数，指的是输入数据读取、送入网络、计算损失、更新参数的流程，该流程较为固定。使用的是多分类交叉生损失函数、使用 SGD 优化方法。最终将训练好的模型持久化到磁盘中。
    
    ```python
    # 模型训练过程
    def train(train_dataset,input_dim,class_num,):
    	# 固定随机数种子
    	torch.manual_seed(0)
    	# 初始化数据加载器
    	dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    	# 初始化模型
    	model = PhonePriceModel(input_dim, class_num)
    	# 损失函数
    	criterion = nn.CrossEntropyLoss()
    	# 优化方法
    	optimizer = optim.SGD(model.parameters(), lr=1e-3)
    	# 训练轮数
    	num_epoch = 50
    ```
    
- 编写训练函数
    
    ```python
    # 遍历每个轮次的数据
    for epoch_idx in range(num_epoch):
    	# 训练时间
    	start = time.time()
    	# 计算损失
    	total_loss = 0.0
    	total_num = 0
    	# 遍历每个 batch 数据进行处理
    	for x, y in dataloader:
    		# 将数据送入网络中进行预测
    		output = model(x)
    		# 计算损失
    		loss = criterion(output, y)
    		# 梯度清零
    		optimizer.zero_grad()
    		# 反向传播
    		loss.backward()
    		# 参数更新
    		optimizer.step()
    		# 损失计算
    		total_num += 1
    		total_loss += loss.item()
    	# 打印损失变换结果
    	print('epoch: %4s loss: %.2f, time: %.2fs' %(epoch_idx + 1, total_loss / total_num, time.time() - start))
    # 模型保存
    torch.save(model.state_dict(), 'model/phone.pth')
    ```
    
- 调用训练函数
    
    ```python
    if __name__ == '__main__':
    	# 获取数据
    	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
    	# 模型训练过程
    	train(train_dataset, input_dim, class_num)
    ```
    
- 案例演示：[**01_ann_phone_price_classification.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day06/01_ann_phone_price_classification.py)（ANN案例_手机价格分类案例）

# 04_ANN案例_手机价格分类_模型测试

- 编写评估函数
    - 使用训练好的模型，对未知的样本的进行预测的过程。使用前面单独划分出来的验证集来进行评估。
    
    ```python
    def test(valid_dataset, input_dim, class_num):
    	# 加载模型和训练好的网络参数
    	model = PhonePriceModel(input_dim, class_num)
    	model.load_state_dict(torch.load('model/phone.pth'))
    
    	# 构建加载器
    	dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    
    	# 评估测试集
    	correct = 0
    	# 遍历测试集中的数据
    	for x, y in dataloader:
    		# 将其送入网络中
    		output = model(x)
    		# 获取类别结果
    		y_pred = torch.argmax(output, dim=1)
    		# 获取预测正确的个数
    		correct += (y_pred == y).sum()
    	# 求预测精度
    	print('Acc: %.5f' % (correct.item() / len(valid_dataset)))
    ```
    
- 调用评估函数
    
    ```python
    if __name__ == '__main__':
    	# 获取数据
    	train_dataset, valid_dataset, input_dim, class_num = create_dataset()
    	# 模型预测结果
    	test(valid_dataset, input_dim, class_num)
    ```
    
- 案例演示：[**01_ann_phone_price_classification.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day06/01_ann_phone_price_classification.py)（ANN案例_手机价格分类案例）

# 05_ANN案例_手机价格分类_调优思路

- 可以通过以下方面进行调优：
    1. 对输入数据进行标准化
    2. 调整优化方法
    3. 调整学习率
    4. 增加批量归一化层
    5. 增加网络层数、神经元个数
    6. 增加训练轮数
    7. 等等
- 进行下如下调整：
    1. 优化方法由 SGD 调整为 Adam
    2. 学习率由 1e-3 调整为 1e-4
    3. 对数据进行标准化
    4. 增加网络深度（增加网络参数量）
- 案例演示：[**02_ann_phone_price_optimized.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day06/02_ann_phone_price_optimized.py)（ANN案例_手机价格分类案例_优化版）

# 06_图像相关知识介绍

- 图像基本概念
    - 图像是由像素点组成的，每个像素点的取值范围为：[0, 255]。像素值越接近于 0，颜色越暗，接近于黑色；像素值越接近于 255，颜色越亮，接近于白色。
    - 在深度学习中，我们使用的图像大多是彩色图，彩色图由 RGB 3 个通道组成。
- 图像分类
    
    
    | 图像类型 | 通道数 | 像素值范围 | 主要特点 | 常见用途 |
    | --- | --- | --- | --- | --- |
    | **二值图像** | 1 通道 | 0 或 1 | 每个像素只有黑与白两种值 | 形态学操作、二值化、轮廓检测 |
    | **灰度图像** | 1 通道 | 0 到 255 | 每个像素表示灰度（亮度） | 图像预处理、物体检测、人脸识别 |
    | **索引图像** | 1 通道 | 0 到 255（索引） | 像素值为颜色表的索引，颜色表决定实际颜色 | 存储压缩、较少颜色的图像表示 |
    | **RGB图像** | 3 通道（R、G、B） | 0 到 255 | 每个像素由红、绿、蓝三个通道组成 | 普通彩色图像显示、图像处理与分析 |
- API
    - `imshow()`：基于 HWC 展示图像
    - `imread()`：读取图像，获取 HWC
    - `imsave()`：基于 HWC 保存图片
- 案例演示：[**03_data_visualization.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day06/03_data_visualization.py)（绘制图像）

# 08_CNN概述介绍

- 卷积神经网络（Convolutional Neural Network）是含有卷积层的神经网络. 卷积层的作用就是用来自动学习、提取图像的特征。
- CNN 网络主要由三部分构成：**卷积层**、**池化层**和**全连接层**构成。
    1. **卷积层**负责提取图像中的局部特征；
    2. **池化层**用来大幅降低参数量级（降维）；
    3. **全连接层**用来输出想要的结果。
- CNN 要做的事情是：给定一张图片，是车还是马未知，是什么车也未知，现在需要模型判断这张图片里具体是一个什么东西，总之输出一个结果：如果是车，那是什么车。
    - 数据输入层：对数据做一些处理，比如去均值（各维度都减对应维度的均值，使得输入数据各个维度都中心化为 0，避免数据过多偏差，影响训练效果）、归一化（把所有的数据都归一到同样的范围）、PCA 等等。CNN 只对训练集做“去均值”这一步。
    - 卷积层 (CONV)：线性乘积求和，提取图像中的局部特征。
    - 激励层 (RELU)：ReLU 激活函数，输入数据转换成输出数据。
    - 池化层 (POOL)：取区域平均值或最大值，大幅降低参数量级（降维）。
    - 全连接层 (FC)：输出 CNN 模型预测结果。
- 卷积神经网络应用
    - **图像分类**：最常见的应用，例如识别图片中的物体类别。
    - **目标检测**：检测图像中物体的位置和类别。
    - **图像分割**：将图像分成多个区域，用于语义分割。
    - **人脸识别**：识别图像中的人脸。
    - **医学图像分析**：用于检测医学图像中的异常（如癌症检测、骨折检测等）。
    - **自动驾驶**：用于识别交通标志、车辆、行人。
- CNN 中的经典算法/网络架构
    - **LeNet-5**：作为最早的 CNN 架构之一，证明了 CNN 在图像识别任务上的有效性，为后续的 CNN 发展奠定了基础。
    - **AlexNet**：显著提升了 ImageNet 图像分类的准确率，证明了深度学习在计算机视觉领域的潜力，并推动了深度学习的快速发展。
    - **VGGNet**：探索了网络深度对性能的影响，证明了更深的网络可以提取更抽象和更具表达力的特征。
    - **GoogLeNet (Inception)**：提出了 Inception 模块，在提高性能的同时减少了计算量，为后续的网络架构设计提供了新的思路。
    - **ResNet**：解决了深度网络训练困难的问题，使得可以训练更深的网络，从而显著提高了模型的性能。
    - **DenseNet**：通过密集连接（Dense Connectivity）在网络中各层之间建立了直接的连接，即每一层都接收前面所有层的输出作为输入。这种设计增强了特征传递和梯度流动，避免了梯度消失问题，并提高了信息的利用率。

# 09_卷积层_计算规则介绍

- 卷积计算
    1. input 表示输入的图像。
    2. filter 表示卷积核，也叫做卷积核（滤波矩阵）。
        - 一组固定的权重，因为每个神经元的多个权重固定，所以又可以看做一个恒定的滤波器 filter。
        - 非严格意义上来讲，下图中红框框起来的部分便可以理解为一个滤波器，即带着**一组固定权重的神经元**。多个滤波器叠加便成了卷积层。
        - 一个卷积核就是一个神经元。
    3. input 经过 filter 得到输出为最右侧的图像，该图叫做特征图。
    - 卷积运算本质上就是在卷积核和输入数据的局部区域间做点积。
        
        ![卷积计算过程](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day6/%E5%8D%B7%E7%A7%AF%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B.png)
        
        卷积计算过程
        
- 卷积层的主要作用如下：
    - **特征提取**：卷积层的主要作用是从输入图像中提取低级特征（如边缘、角点、纹理等）。通过多个卷积层的堆叠，网络能够逐渐从低级特征到高级特征（如物体的形状、区域等）进行学习。
    - **权重共享**：在卷积层中，同一个卷积核在整个输入图像上共享权重，这使得卷积层的参数数量大大减少，减少了计算量并提高了训练效率。
    - **局部连接**：卷积层中的每个神经元仅与输入图像的一个小局部区域相连，这称为**局部感受野**，这种局部连接方式更符合图像的空间结构，有助于捕捉图像中的局部特征。
    - **空间不变性**：由于卷积操作是局部的并且采用权重共享，卷积层在处理图像时具有**平移不变性**。也就是说，不论物体出现在图像的哪个位置，卷积层都能有效地检测到这些物体的特征。

# 10_卷积层_填充(Padding)介绍

- 通过上面的卷积计算过程，最终的特征图比原始图像小很多，如果想要保持经过卷积后的图像大小不变，可以在原图周围添加 padding 来实现。
- padding（填充）操作用于处理卷积时图像边缘的像素。
- 其目的是在输入图像的边界周围添加额外的像素（通常是零），从而解决卷积操作时边缘信息丢失的问题。
- Padding 的主要作用
    - **保持空间维度**：如果不使用 Padding，每次卷积操作后，特征图的尺寸都会缩小。多次卷积后，特征图会变得非常小，可能会丢失重要的边缘信息。Padding 可以帮助维持输出特征图的尺寸与输入相同或接近相同。
    - **保留边缘信息**：图像边缘的像素在卷积过程中参与的计算次数较少，这意味着边缘信息在特征提取过程中容易丢失。Padding 通过在边缘添加额外的像素，增加了边缘像素的参与度，从而更好地保留了边缘信息。
    - **提高性能**：Padding 有助于避免由于特征图尺寸快速缩小而导致的信息丢失，从而提高模型的性能，尤其是在处理较小的图像或需要进行多层卷积时。
- Padding的类型
    - **Valid Padding (No Padding)**：不进行任何填充。卷积核只在输入图像的有效区域内滑动。输出尺寸会缩小。
    - **Same Padding**：添加足够的填充，使得输出特征图的尺寸与输入相同。
    - **Full Padding**：尽可能多地添加填充，使得卷积核的每个元素都至少在输入图像上滑动一次。输出尺寸会增大。
- **Padding 的选择**：取决于具体的应用场景和网络架构
    - **Valid Padding**：适用于不需要保持输出尺寸的场景，或者输入图像足够大，边缘信息丢失不重要的情况。
    - **Same Padding**：广泛应用于各种CNN架构中，因为它可以保持特征图的尺寸，方便网络设计和计算。
    - **Full Padding**：较少使用，因为它会增加计算量，并且可能会在边缘引入一些伪影。

# 11_卷积层_步长(Stride)介绍

- stride（步长）指的是**卷积核在图像上滑动时的步伐大小**，即每次卷积时卷积核在图像中向右（或向下）移动的像素数。步长直接影响卷积操作后输出特征图的尺寸，以及计算量和模型的特征提取能力。
- Stride 的作用
    - **降低计算复杂度**：更大的步长意味着卷积核移动的次数更少，从而减少了计算量，并加快了训练和推理速度。
    - **减 1 长越大，生成的特征图尺寸越小**。这类似于池化的降维效果。
    - **增大感受野**：虽然更大的步长会减小特征图的尺寸，但它同时也会增大每个神经元在输入数据上的感受野。这意味着每个神经元能够捕捉到更大范围的输入信息。
- Stride 的选择：取决于具体的应用场景和网络架构
    - **Stride = 1**：这是最常见的设置，尤其是在网络的早期层。它允许保留更多的空间细节。
    - **Stride > 1**：通常用于减小特征图的尺寸和增大感受野，例如在网络的后期层或需要进行快速降维时。常见的设置包括 stride=2 或 stride=4。

# 12_卷积层_多通道卷积计算

- 计算方法
    1. 当输入有多个通道（Channel），例如 RGB 三个通道，此时要求卷积核需要拥有相同的通道数（图像有多少通道，每个卷积核就有多少通道）。
    2. 每个卷积核通道与对应的输入图像的各个通道进行卷积。
    3. 将每个通道的卷积结果按位相加得到最终的特征图。
    
    ![多通道卷积计算方法](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day6/%E5%A4%9A%E9%80%9A%E9%81%93%E5%8D%B7%E7%A7%AF%E8%AE%A1%E7%AE%97%E6%96%B9%E6%B3%95.png)
    
    多通道卷积计算方法
    

# 13_卷积层_多卷积核卷积计算

- 计算方法
    - 两个神经元，意味着有两个滤波器。
    - 数据窗口每次移动两个步长取 3*3 的局部数据，即 stride=2。
    - zero-padding=1。输入数据由`5*5*3`变为`7*7*3`。
    - 左边是输入（7*7*3 中，7*7 代表图像的像素 / 长宽，3 代表 R、G、B 三个颜色通道）。
    - 中间部分是两个不同的滤波器 Filter w0、Filter w1。
    - 最右边则是两个不同的输出。
    
    ![多卷积核卷积计算方法](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day6/%E5%A4%9A%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%8D%B7%E7%A7%AF%E8%AE%A1%E7%AE%97%E6%96%B9%E6%B3%95.gif)
    
    多卷积核卷积计算方法
    

# 14_卷积层_特征图(FeatureMap)计算规则

- 输出特征图的大小与以下参数息息相关
    - size：卷积核/过滤器大小，一般会选择为奇数，比如有 1×1 、3×3、5**×**5
    - Padding：零填充的方式
    - Stride：步长
- 计算方法
    - 输入图像大小：W × W
    - 卷积核大小：F × F
    - Stride：S
    - Padding：P
    - 输出图像大小：N × N
    - **计算公式**：$N = \frac{W - F + 2P}{S} + 1$
- 以下图为例
    - 图像大小：5 × 5
    - 卷积核大小：3 × 3
    - Stride：1
    - Padding：1
    - N = (5 - 3 + 2) / 1 + 1 = 5，即得到的特征图大小为：5 x 5

# 15_卷积层_API介绍

- API
    
    ```python
    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    ```
    
    - in_channels：输入图像的通道数
    - out_channels：输出图像的通道数
    - kernel_size：卷积核的大小
    - stride：步长
    - padding：填充
- 案例演示：[**04_convolution_layers_api.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day06/04_convolution_layers_api.py)（卷积层API介绍）

# 16_池化层_介绍

- 池化层计算
    - 池化层 (Pooling) 降低维度，从而减少计算量、减少内存消耗，并提高模型的鲁棒性。
    - 池化层通常位于卷积层之后，它通过对卷积层输出的特征图进行下采样，保留最重要的特征信息，同时丢弃一些不重要的细节。
    
    ![池化过程](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day6/%E6%B1%A0%E5%8C%96%E8%BF%87%E7%A8%8B.png)
    
    池化过程
    
- Padding：填充
- Stride：步长
- 多通道池化层计算
    - 在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各个通道的输入相加。这意味着池化层的输出和输入的通道数是相等。

# 17_池化层_API介绍

- API
    - 最大池化
        
        ```python
        nn.MaxPool2d(kernel_size, stride, padding)
        ```
        
        - kernel_size：池化核大小
        - stride：步长
        - padding：填充
    - 平均池化
        
        ```python
        nn.AvgPool2d(kernel_size, stride, padding)
        ```
        
        - kernel_size：池化核大小
        - stride：步长
        - padding：填充
- 案例演示：[**05_pooling_layers_api.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day06/05_pooling_layers_api.py)（池化层API介绍）