# 神经网络与深度学习笔记_Day1

@ZZHow(ZZHow1024)

参考课程：

【**黑马程序员AI大模型《神经网络与深度学习》全套视频课程，涵盖Pytorch深度学习框架、BP神经网络、CNN图像分类算法及RNN文本生成算法**】

[[**https://www.bilibili.com/video/BV1c5yrBcEEX**](https://www.bilibili.com/video/BV1c5yrBcEEX)]

# 01_深度学习_知识框架介绍

![知识框架.png](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day1/%E7%9F%A5%E8%AF%86%E6%A1%86%E6%9E%B6.png)

- 机器学习在处理**图像**和**文本数据**方面，能力较弱。
- **CNN**，卷积神经网络 (Convolutional Neural Network)
    - 结构
        - 卷积层
        - 池化层
        - 全连接层
    - 案例：图像分类，CIFAR10 数据集
    - 进阶：CV，计算机视觉 (Computer Vision)
- **RNN**，循环神经网络 (Recurrent NN)
    - 结构
        - 词嵌入层
        - 循环网络层
        - 全连接层
    - 案例：歌词 AI 生成器
    - 进阶：NLP，自然语言处理 (Natural Language Processing)
- **ANN**，人工神经网络 (Artificial neural network)
    - 结构
        - 输入层 → 1 层
        - 隐藏层 → N 层
        - 输出层 → 1 层
- 标量，向量，矢量 → 张量：Tensor
- 工具
    - **PyTorch（当前主流）**
    - TensorFlow（趋于过时）

# 02_深度学习_简介

![深度学习发展](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day1/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%8F%91%E5%B1%95.jpeg)

深度学习发展

- **机器学习**是实现**人工智能**的一种途径。
- 深度学习是机器学习的一个子集，也就是说深度学习是实现机器学习的一种方法。
- 深度学习是机器学习中一种**基于对数据进行特征学习**的算法。
- 深度学习是**基于人工神经网络**，**深度**是指**网络中使用多层**，每层都通过**非线性变换**处理数据，并逐渐提取出更复杂、更抽象的特征。

![深度学习与机器学习对比](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day1/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AF%B9%E6%AF%94.png)

深度学习与机器学习对比

- 传统机器学习算法依赖人工设计特征，并进行特征提取：而**深度学习**方法不需要人工，而是依**赖算法自动提取特征**。
- 深度学习通过模仿人脑的神经网络来处理和分析复杂的数据，从大量数据中自动提取复杂特征。这也是深度学习被看做**黑盒子**，可解释性差的原因。
- 深度学习**尤其擅长处理高维数据**，如图像、语音和文本。

# 03_深度学习_特点

1. **多层非线性变换**：深度学习模型由多个层次组成，每一层都应用非线性激活函数对输入数据进行变换。较低的层级通常捕提到简单的特征（如边缘、颜色等），而更高的层级则可以识别更复杂的模式（如物体或面部识别）。
2. **自动特征提取**：与传统机器学习算法不同，深度学习能够自动从原始数据中学习到有用的特征，而不需要人工特征工程。这使得深度学习在许多领域中表现出色。
3. **大数据和计算能力**：深度学习模型通常需要大量的标注数据和强大的计算资源（如 GPU）来进行训练。大数据和高性能计算使得深度学习在图像识别、自然语言处理等领域取得了显著突破。
4. **可解释性差**：深度学习模型内部的运作机制相对不透明，被称为“黑箱”，这意味着理解模型为什么做出特定决策可能会比较困难。这对某些应用场景来说是一个挑战。

# 04_深度学习_常用模型介绍

- 卷积神经网络 (Convolutional Neural Networks, CNN)
    - 结构
        - 输入层
        - 隐藏层
            - 卷积层
            - 池化层
            - 全连接层
        - 输出层
    - 应用：图像 → CV
    - 主要用于图像处理任务，如图像分类、目标检测、图像分割等。
    - 特点是使用卷积层来自动提取图像中的局部特征，并通过池化层减少参数数量，提高计算效率。
- 循环神经网络 (Recurrent Neural Networks, RNN)
    - 应用：文本 → NLP
    - 适用于处理序列数据, 例如自然语言处理 (NLP) 、语音识别等。
    - RNN 具有记忆功能，可以处理输入数据的时间依赖性，但标准 RNN 难以捕捉长期依赖关系。
- 自编码器 (Autoencoders)
    - 适用于处理序列数据，例如自然语言处理 (NLP) 、语音识别等。
    - RNN 具有记忆功能，可以处理输入数据的时间依赖性，但标准 RNN 难以捕捉长期依赖关系。
- 生成对抗网络 (Generative Adversarial Networks, GAN)
    - 包含两个子网络：生成器和判别器。生成器负责创建看起来真实的假样本，而判别器则试图区分真假样本。
    - GAN 广泛应用于图像生成、视频合成等领域。
- Transformer
    - 主要用于自然语言处理 (NLP) 任务，尤其是机器翻译、文本生成等。
    - Transformer 摒弃了传统的递归结构，采用自注意机制 (self-attention)，使得它能够并行处理整个句子的信息，在机器翻译、文本摘要等任务中表现出色。
- 深度强化学习 (Deep Reinforcement Learning, DRL)
- 图神经网络 (GNN, Graph Neural Network)

# 05_深度学习_应用场景介绍

- **计算机视觉 (Computer Vision)**
    - 图像分类：将图像分为不同的类别。常用于人脸识别、物体检测等。
        - 自动标注社交媒体照片、医疗影像中的病变检测。
    - 目标检测 (Object Detection) ：在图像或视频中定位并分类多个对象。
        - 自动驾驶中的行人检测、监控视频中的入侵检测。
    - 面部识别：通过分析面部特征进行身份验证或分类。
        - 手机解锁、安防监控系统。
    - 图像生成：基于输入生成新的图像，如风格转换、图像超分辨率等。
        - 艺术风格迁移、老旧照片修复。
- **自然语言处理 (Natural Language Processing, NLP)**
    - 机器翻译：使用深度学习模型将一种语言的文本自动翻译成另一种语言。
        - Google 翻译、实时语音翻译。
    - 情感分析：分析文本中的情感倾向，如正面、负面或中性。
        - 社交媒体监控、产品评论分析。
    - 文本生成：生成符合语法和语义的自然语言文本。
        - 自动写作助手、新闻生成。
    - 语音识别：将语音转化为文字。
        - 智能助手（如 Siri、Alexa）、自动字幕生成。
    - 聊天机器人（Chatbot）：通过深度学习理解用户输入并生成合理的回应。
        - 客服机器人、虚拟助手（如 GPT 类模型）。
- **推荐系统 (Recommendation Systems)**
    - 电影、音乐推荐：根据用户历史的评分和行为，推荐相关的电影、音乐或电视剧。
        - Netflix、 Spotify 的个性化推荐。
    - 电商推荐：根据用户的购买历史和浏览习惯推荐商品。
        - 亚马逊、淘宝的商品推荐系统。
    - 社交媒体推荐：分析用户的社交行为，推荐相关内容或朋友。
        - Facebook、 Instagram 的内容推荐。
- **多模态大模型**

# 06_深度学习_发展史介绍

- 人工智能发展历史
    - 符号主义（20 世纪 50-70）
        - **专家系统**占主导
        - 1950：图灵设计国际象棋程序
        - 1962：IBM Arthur Samuel 的跳棋程序战胜人类高手（人工智能第一次浪潮）
    - 统计主义（20 世纪 80-2000）
        - 主要用**统计模型**解决问题
        - 1993：Vapnik 提出 SVM
        - 1997：IBM 深蓝战胜卡斯帕罗夫（人工智能第二次浪潮）
    - 神经网络（21 世纪初期）
        - **神经网络**、深度学习流派
        - 2012：AlexNet 深度学习的开山之作
        - 2016：Google AlphaGO 战胜李世石（人工智能第三次浪潮）
    - 大规模预训练模型（2017-至今）
        - **大规模预训练模型**
        - 2017：自然语言处理 NLP 的 Transformer 框架出现
        - 2018：Bert 和 GPT 的出现
        - 2022：ChatGPT 的出现，进入到大模型 AIGC 发展的阶段
- 深度学习发展历史
    - 早期探索（1940s-1980s）
        - 20 世纪 40 年代：沃尔特•皮茨 (Walter Pitts) 和沃伦• 麦卡洛克 (Warren McCulloch) 等开始模仿生物神经系统来构建计算模型，如 McCulloch-Pitts **神经元**。
        - 1957：弗兰克•罗森布拉特 (Frank Rosenblatt) 提出**感知器概念**，能够进行简单的二分类任务。
        - 1960 年代末：出现了多层感知器 (MLP)，但当时由于**计算能力和数据量的限制**，这些模型的应用受到很大限制。
    - 挑战与瓶颈（1980s-1990s）
        - 1986：David Rumelhart等人发表了关于**反向传播 (Backpropagation, BP) 算法**的研究成果，使得多层神经网络能够通过梯度下降优化参数，解决复杂的非线性问题。
        - 由于**计算资源的限制**以及对复杂数据 (如图像和语音) 的处理能力较弱，深度学习未能广泛应用。
    - 复兴与突破（2000s-2010s）
        - 2006：杰弗里•辛顿 (Geoffrey Hinton) 和其团队提出了深度置信网络 (DBN)，标志着深度学习的复兴。
        - 2012：亚历克斯•克里泽夫斯基 (Alex Krizhevsky) 等人设计的**卷积神经网络 (CNN)**，AlexNet 在 ImageNet 图像识别挑战赛中取得巨大成功。
    - 爆发期（2016-至今）
        - 2016：Google AlphaGO 战胜李世石（人工智能第三次浪潮）。
        - 2017：自然语言处理 NLP 的 Transformer 框架出现。
        - 2018：Bert 和 GPT 的出现。
        - 2022：ChatGPT 的出现，进入到大模型 AIGC 发展的阶段。

# 07_PyTorch框架简介

[PyTorch Foundation](https://pytorch.org/)

- 什么是 PyTorch
    - PyTorch 一个基于 Python 语言的深度学习框架，它将数据封装成张量 (Tensor) 来进行处理。
    - PyTorch 提供了灵活且高效的工具，用于构建、训练和部署机器学习和深度学习模型。
    - PyTorch 广泛应用于学术研究和工业界，特别是在计算机视觉、自然语言处理、强化学习等领域。
- PyTorch 的安装：`pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple`
- PyTorch 特点
    - 类似于 NumPy 的张量计算
    - 自动微分系统
    - 深度学习库
    - 动态计算图
    - GPU 加速（CUDA 支持）
    - 支持多种应用场景
    - 跨平台支持
- PyTorch 发展历史
    - 2016 年 Facebook 正式发布了 PyTorch 的第一个版本。
    - 2018 年 PyTorch 发布了 1.0 版本，标志着其正式进入生产级应用阶段。

# 08_PyTorch_张量基本创建方式

- 什么是张量
    - PyTorch 中的张量就是元素为同一种数据类型的多维矩阵。在 PyTorch 中，张量以“类”的形式封装起来，对张量的一些运算、处理的方法被封装在类中。
    - PyTorch 张量与 NumPy 数组类似，但 PyTorch 的张量具有 GPU 加速的能力（通过 CUDA），这使得深度学习模型能够高效地在 GPU 上运行。
    - PyTorch 提供了对张量的强大支持，可以进行高效的数值计算、矩阵操作、自动求导等。
    - 张量是 PyTorch 中的核心数据抽象，PyTorch 支持各种张量子类型。通常地，一维张量称为向量/矢量 (vector)，二维张量称为矩阵 (matrix)。
- 张量基本创建方式
    - `torch.tensor` 根据指定数据创建张量。
    - `torch.Tensor` 根据形状创建张量, 其也可用来创建指定数据的张量。
    - `torch.IntTensor`、`torch.FloatTensor`、`torch.DoubleTensor` 创建指定类型的张量。
- 案例演示：[**01_tensor_creation_basics.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day01/01_tensor_creation_basics.py)（张量的基本创建方式）

# 09_PyTorch_创建线性和随机张量

- 创建线性和随机张量
    - `torch.arange()` 和 `torch.linspace()` 创建线性张量。
    - `torch.random.initial_seed()` 和 `torch.random.manual_seed()` 随机种子设置。
    - `torch.rand` / `randn()` 创建随机浮点类型张量。
    - `torch.randint(low, high, size=())` 创建随机整数类型张量。
- 案例演示：[**02_linear_random_tensors.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day01/02_linear_random_tensors.py)（创建线性和随机张量）

# 10_PyTorch_创建全0_1_指定值张量

- 创建 0、1、指定值张量
    - torch.ones 和 torch.ones_like 创建全 1 张量
    - torch.zeros 和 torch.zeros_like 创建全 0 张量
    - torch.full 和 torch.full_like 创建全为指定值张量
- 案例演示：[**03_constant_tensors.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day01/03_constant_tensors.py)（创建全0_1_指定值张量）

# 11_PyTorch_元素类型转换

- 张量元素类型转换
    - data.type(torch.DoubleTensor)
    - data.half / double / float / short / int / long()
- 案例演示：[**04_tensor_type_conversion.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day01/04_tensor_type_conversion.py)（张量类型转换）