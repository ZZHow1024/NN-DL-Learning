# 神经网络与深度学习笔记_Day7

@ZZHow(ZZHow1024)

参考课程：

【**黑马程序员AI大模型《神经网络与深度学习》全套视频课程，涵盖Pytorch深度学习框架、BP神经网络、CNN图像分类算法及RNN文本生成算法**】

[[**https://www.bilibili.com/video/BV1c5yrBcEEX**](https://www.bilibili.com/video/BV1c5yrBcEEX)]

# 01_今日内容大纲介绍

- CNN 图像分类案例
    - 加载数据集 → CIFAR10 数据集
    - 构建神经网络分类模型
    - 模型训练
    - 模型评估
    - 模型优化
- RNN 入门
    - 生成杰伦歌词案例

# 02_CNN图像分类案例_准备数据集

- 卷积神经网络案例
    - 使用前面学习到的知识来构建一个卷积神经网络，并训练该网络实现图像分类。要完成这个案例，需要学习的内容如下：
        - 了解 CIFAR10 数据集
        - 搭建卷积神经网络
        - 编写训练函数
        - 编写预测函数
- CIFAR10 数据集：有 5 万张训练图像、1 万张测试图像、10 个类别、每个类别有 6k 个图像，图像大小 32×32×3。下图列举了 10 个类，每一类随机展示了 10 张图片。
- 案例演示：[**01_cnn_image_classification.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/01_cnn_image_classification.py)（CNN案例_图像分类）

# 03_CNN图像分类案例_搭建神经网络_思路分析

- 搭建图像分类网络
    - 网络结构
        1. 输入形状：32×32。
        2. 第一个卷积层输入 3 个 Channel，输出 6 个 Channel，Kernel Size 为 3×3。
        3. 第一个池化层输入 30×30，输出 15×15，Kernel Size 为 2×2，Stride 为 2。
        4. 第二个卷积层输入 6 个 Channel，输出 16 个 Channel，Kernel Size 为 3×3。
        5. 第二个池化层输入 13×13，输出 6×6，Kernel Size 为 2×2，Stride 为 2。
        6. 第一个全连接层输入 576 维，输出 120 维。
        7. 第二个全连接层输入 120 维，输出 84 维。
        8. 最后的输出层输入 84 维，输出 10 维。
        9. 在每个卷积计算之后应用 ReLU 激活函数来给网络增加非线性因素。
    - 网络结构图
        
        ![网络结构图](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day7/%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E5%9B%BE.png)
        
        网络结构图
        

# 04_CNN图像分类案例_搭建神经网络_代码实现

- 卷积层参数计算公式：$卷积层参数量 = 输入通道数 × 卷积核尺寸 × 卷积核数量 + 卷积核数量$
- 代码
    
    ```python
    # 搭建卷积神经网络
    class CNNModel(nn.Module):
        # 1. 初始化父类成员并搭建神经网络
        def __init__(self):
            # 1.1 初始化父类成员
            super(CNNModel, self).__init__()
    
            # 1.2 搭建神经网络
            # 第 1 个卷积层，输入 3 通道，输出 6 通道，卷积核 3*3，步长 1，填充 0
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
            # 第 1 个池化层，窗口 3*3，步长 2，填充 0
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
            # 第 2 个卷积层，输入 6 通道，输出 16通道，卷积核 3*3，步长 1，填充 0
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
            # 第 2 个池化层，窗口 2*2，步长 2，填充 0
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
            # 第 1 个全连接层
            self.linear1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)
            # 第 2 个全连接层
            self.linear2 = nn.Linear(in_features=120, out_features=84)
    
            # 输出层
            self.output = nn.Linear(in_features=84, out_features=10)
    
        # 2. 定义前向传播
        def forward(self, x):
            # 第 1 层：卷积层（加权求和） + 激励层（激活函数） + 池化层（降维）
            x = self.pool1(torch.relu(self.conv1(x)))
    
            # 第 2 层：卷积层（加权求和） + 激励层（激活函数） + 池化层（降维）
            x = self.pool2(torch.relu(self.conv2(x)))
    
            # 全连接层只能处理二维数据，要将数据进行拉平
            # 参数 1：样本数（行）；参数 2：特征数（列），-1 表示自动计算
            x = x.reshape(x.shape[0], -1)
            # print(f'x.shape: {x.shape}')
    
            # 第 3 层：全连接层（加权求和） + 激励层（激活函数）
            x = torch.relu(self.linear1(x))
    
            # 第 4 层：全连接层（加权求和） + 激励层（激活函数）
            x = torch.relu(self.linear2(x))
    
            # 第 5 层：全连接层（加权求和）输出层
            return self.output(x)
    ```
    
- 案例演示：[**01_cnn_image_classification.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/01_cnn_image_classification.py)（CNN案例_图像分类）

# 05_CNN图像分类案例_模型训练

- 代码
    
    ```python
    # 模型训练
    def train(train_dataset):
        # 1. 创建数据加载器
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
        # 2. 创建模型对象
        model = CNNModel().to(device)
    
        # 3. 创建损失函数对象
        criterion = nn.CrossEntropyLoss().to(device)
    
        # 4. 创建优化器对象
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
        # 5. 开始每轮的训练动作
        epochs = 10
        for epoch in range(epochs):
            total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                # 切换模式
                model.train()
                # 模型预测
                y_pred = model(x)
                # 计算损失
                loss = criterion(y_pred, y)
                # 梯度清零 + 反向传播 + 参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 统计预测正确的样本个数
                total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
                # 统计当前批次的总损失
                total_loss += loss.item() * len(y)
                # 统计当前批次总样本个数
                total_samples += len(y)
    
            # 打印该轮训练完毕，打印该轮的训练信息
            print(
                f'epoch: {epoch + 1}, loss: {total_loss / total_samples:.3f}, accuracy: {total_correct / total_samples:.3f}, time: {time.time() - start}s')
    
        # 6. 保存模型
        torch.save(model.state_dict(), os.path.join('model', 'cnn-model.pth'))
    ```
    
- 案例演示：[**01_cnn_image_classification.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/01_cnn_image_classification.py)（CNN案例_图像分类）

# 06_CNN图像分类案例_模型测试

- 代码
    
    ```python
    # 模型测试
    def evaluate(test_dataset):
        # 1. 创建测试集数据加载器
        dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
        # 2. 创建模型对象
        model = CNNModel().to(device)
    
        # 3. 加载模型参数
        model.load_state_dict(torch.load(os.path.join('model', 'cnn-model.pth')))
    
        # 4. 统计预测正确的样本个数、总样本个数
        total_correct, total_samples = 0, 0
    
        # 5. 遍历数据加载，获取到每批次的数据
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            # 切换模型模式
            model.eval()
            # 模型预测
            y_pred = model(x)
            # 通过 argmax() 获得预测的分类索引
            y_pred = torch.argmax(y_pred, dim=-1)
            # 统计预测正确的样本个数
            total_correct += (y_pred == y).sum()
            # 统计总样本个数
            total_samples += len(y)
    
        # 6. 打印正确率
        print(f'Accuracy: {total_correct / total_samples:.3f}')
    ```
    
- 案例演示：[**01_cnn_image_classification.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/01_cnn_image_classification.py)（CNN案例_图像分类）

# 07_CNN图像分类案例_优化及总结

- 掌握模型构建流程
    - 加载数据集
    - 模型构建
    - 模型训练
    - 模型测试
- 优化方法
    - 增加卷积核输出通道数
    - 增加全连接层的参数量
    - 调整学习率
    - 调整优化方法
    - 修改激活函数
    - …
- [**02_cnn_image_classification_optimized.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/02_cnn_image_classification_optimized.py)（CNN案例_图像分类_优化版）

# 08_RNN介绍

- RNN 介绍
    - 循环神经网络（Recurrent Neural Network, RNN）是一种**专门处理序列数据的神经网络**。与传统的前馈神经网络不同，RNN具有“循环”结构，能够处理和记住前面时间步的信息，使其特别适用于时间序列数据或有时序依赖的任务。
    - 需要明确什么是**序列数据**，时间序列数据是指在不同时间点上收集到的数据，这类数据反映了某一事物、现象等随时间的变化状态或程度。这是时间序列数据的定义，当然这里也可以不是时间，比如文字序列，但总归序列数据有一个特点——**后面的数据跟前面的数据有关系**。
- RNN 的应用
    - 自然语言处理（NLP）：文本生成、语言建模、机器翻译、情感分析等。
    - 时间序列预测：股市预测、气象预测、传感器数据分析等。
    - 语音识别：将语音信号转换为文字。
    - 音乐生成：通过学习音乐的时序模式来生成新乐曲。
- 自然语言处理概述
    - 自然语言处理（Nature Language Processing, NLP）研究的主要是通过计算机算法来理解自然语言。对于自然语言来说，处理的数据主要就是人类的语言，例如：汉语、英语、法语等，该类型的数据不像我们前面接触过的结构化数据、或者图像数据可以很方便的进行数值化。

# 09_词嵌入层_解释

- 词嵌入层作用
    - **词嵌入层的作用就是将文本转换为向量**。
    - 词嵌入层在 RNN 中的作用有**输入表示**、**降低维度**和**捕捉语义相似性**。
    - 词嵌入层首先会根据输入的词的数量构建一个**词向量矩阵**，例如: 我们有 100 个词，每个
    - 希望转换成 128 维度的向量，那么构建的矩阵形状即为：100*128，输入的每个词都对应了一个该矩阵中的一个向量。
- 词嵌入层工作流程
    - **初始化词向量**：词嵌入层的初始词向量通常会使用随机初始化或者通过加载预训练的词向量（如 Word2Vec 或 GloVe）进行初始化。
    - **输入索引**：每个单词在词汇表中都有一个唯一的索引。输入文本（例如一个句子）会先被分词，然后每个单词会被转换为相应的索引。
    - **查找词向量**：词嵌入层将这些单词索引映射为对应的词向量。这些词向量是一个低维稠密向量，表示该词的语义。
    - **输入到 RNN**：这些词向量作为 RNN 的输入，RNN 处理它们并根据上下文生成一个序列的输出。

# 10_词嵌入层_API演示

- 词嵌入层使用
    - 在 PyTorch 中，使用 `nn.Embedding` 词嵌入层来实现输入词的向量化。
        
        ```python
        nn.Embedding(num_embeddings, embedding_dim)
        ```
        
    - `nn.Embedding` 对象构建时，最主要有两个参数
        - `num_embeddings` 表示**词的数量**。
        - `embedding_dim` 表示**用多少维的向量来表示每个词**。
    - 将词转换为词向量的步骤
        1. 先将语料进行分词，构建词与索引的映射，我们可以把这个映射叫做词表，词表中每个词都对应了一个唯一的索引。
        2. 然后使用 nn.Embedding 构建词嵌入矩阵，词索引对应的向量即为该词对应的数值化后的向量表示。
- [**03_word_embedding_demo.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/03_word_embedding_demo.py)（词嵌入层演示）

# 11_RNN层(循环网络层)_简介

- 文本数据是具有序列特性的
    - 例如：“我爱你”，这串文本就是具有序列关系的，“爱”需要在“我”之后，“你”需要在“爱”之后，如果颠倒了顺序，那么可能就会表达不同的意思。
    - 为了表示出数据的序列关系，需要使用循环神经网络 (Recurrent Nearal Networks, RNN) 来对数据进行建模，RNN 是一个作用于处理带有序列特点的样本数据。
- RNN 计算过程
    
    ![RNN计算过程](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0_Day7/RNN%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B.png)
    
    RNN计算过程
    
    - h 表示隐藏状态，保存了**序列数据中的历史信息**，并将这些信息传递给下一个时间步，从而允许 RNN 处理和预测序列数据中的元素。
    - 每一次的输入包含两个值：上一个时间步的隐藏状态、当前状态的输入值 x。
    - 每一次的输出也会包含两个值：当前时间步的隐藏状态、当前时间步的预测结果 y。
    - 隐藏状态作用
        1. 记忆功能：隐藏状态就像 RNN 的记忆，它能够在不同的时间步之间传递信息。当一个新的输入进入网络时，当前的隐藏状态会结合这个新输入来生成新的隐藏状态。
        2. 上下文理解：由于隐藏状态携带了过去的信息，它可以用于理解和生成与上下文相关的输出。这对于语言模型、机器翻译等任务尤其重要。
        3. 连接不同时间步：隐藏状态通过网络内部的循环连接将各个时间步连接起来，使得网络可以处理变长的序列数据。
    - 实际上只有一个神经元，“我爱你”三个字是重复输入到同一个神经元中。
- RNN 神经元内部计算
    - **计算隐藏状态**：每个时间步的隐藏状态 $h_t$ 是根据当前输入 $x_t$ 和前一时刻的隐藏状态 $h_{t-1}$ 计算的。
        - 公式：$h_t = tanh(W_{ih}x_t + b_{ih} + W_{hh}h_{t-1} + b_{hh})$
    - 上述公式中
        - $W_{ih}$ 表示输入数据的权重
        - $b_{ih}$ 表示输入数据的偏置
        - $W_{hh}$ 表示输入隐藏状态的权重
        - $b_{hh}$ 表示输入隐藏状态的偏置
        - $h_{t-1}$ 表示输入隐藏状态
        - $h_t$ 表示输出隐藏状态
        - 最后对输出的结果使用 tanh 激活函数进行计算，得到该神经元你的输出隐藏状态。
    - **计算当前时刻的输出**：网络的输出 $y_t$ 是当前时刻的隐藏状态经过一个线性变换得到的。
        - 公式：$y_t = W_{hy}h_t + b_y$
    - 上述公式中
        1. $y_t$ 是当前时刻的输出（通常是一个向量，表示当前时刻的预测值，RNN 层的预测值）
        2. $h_t$ 是当前时刻的隐藏状态
        3. $W_{hy}$ 是从隐藏状态到输出的权重矩阵
        4. $b_y$ 是输出层的偏置项
    - **词汇表映射**：输出 $y_t$ 是一个**向量**，该向量经过**全连接层**后输出得到最终预测结果 $Y_{pred}$，$Y_{pred}$ 中每个元素代表当前时刻生成词汇表中某个词的得分（或概率，通过激活函数如 Softmax）。**词汇表有多少个词，Ypred 就有多少个元素值，最大元素值对应的词就是当前时刻预测生成的词**。
- 神经元工作机制总结
    - **接收输入**：每个 RNN 神经元接收来自输入数据 $x_t$ 和前一时刻的隐藏状态 $h_{t-1}$。
    - **更新隐藏状态**：神经元通过一个加权和（由权重矩阵和偏置项组成）更新当前时刻的隐藏状态 $h_t$，该隐藏状态包含了来自过去的记忆以及当前输入的信息。
    - **输出计算**：基于当前隐藏状态 $h_t$，神经元生成当前时刻的输出 $y_t$，该输出可以用于任务的最终预测。

# 12_RNN层(循环网络层)_API演示

- API介绍
    
    ```python
    RNN = torch.nn.RNN(input_size,hidden_size, num_layers)
    ```
    
- 参数意义是：
    - `input_size`：输入数据的维度，一般设为词向量的维度；
    - `hidden_size`：隐藏层 h 的维度，也是当前层神经元的输出维度；
    - `num_layers`: 隐藏层 h 的层数，默认为 1。
    - 将 RNN 实例化就可以将数据送入进行处理。
- 输入数据和输出结果
    - 将 RNN 实例化就可以将数据送入其中进行处理。
    
    ```python
    output, hn = RNN(x, h0)
    ```
    
    - 输入数据：输入主要包括词嵌入的 x、初始的隐藏层 h0。
        - x 的表示形式为 [seq_len, batch, input_size]，即 [句子的长度, batch的大小, 词向量的维度]。
        - h0 的表示形式为[num_layers, batch, hidden_size]，即[隐藏层的层数, batch的大, 隐藏层 h 的维数]。
    - 输出结果：主要包括输出结果 output，最后一层的 hn。
        - output 的表示形式与输入 x 类似，为[seq_len, batch, hidden_size]，即 [句子的长度, batch 的大小, 输出向量的维度]。
        - hn 的表示形式与输入 h0 一样，为[num_layers, batch, hidden_size]，即[隐藏层的层数, batch 的大, 隐藏层 h 的维度]。
- 案例演示：[**04_rnn_layer_introduction.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/04_rnn_layer_introduction.py)（RNN层简介）

# 13_RNN_AI歌词生成器案例_构建词表

- 项目需求
    - 文本生成任务是一种常见的自然语言处理任务，输入一个开始词能够预测出后面的词序列。本案例将会使用循环神经网络来实现周杰伦歌词生成任务。
- 数据集
    - 收集了周杰伦从第一张专辑《Jay》到第十张专辑《跨时代》中的歌词，来训练神经网络模型，当模型训练好后，我们就可以用这个模型来创作歌词。
    - 数据集共有 5819 行文本。
- 获取数据集并构建词表
    - 在进行自然语言处理任务之前，首要做的就是构建词表。
    - 词表是将数据进行分词，然后给每一个词分配一个唯一的编号，便于我们送入词嵌入层获取每个词的词向量。
- 对周杰伦歌词的数据进行处理构建词表，整体流程是：获取文本数据、分词并进行去重、构建词表。
- 案例演示：[**05_rnn_lyrics_generator.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/05_rnn_lyrics_generator.py)（RNN_AI歌词生成器）

# 14_RNN_AI歌词生成器案例_构建数据集

- 在训练的时候，为了便于读取语料，并送入网络，所以我们会构建一个 Dataset 对象。
- 案例演示：[**05_rnn_lyrics_generator.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/05_rnn_lyrics_generator.py)（RNN_AI歌词生成器）

# 15_RNN_AI歌词生成器案例_搭建神经网络

- 用于实现《歌词生成》的网络模型，主要包含了三个层
    - **词嵌入层**：用于将语料转换为词向量。
    - **循环网络层**：提取句子语义。
    - **全连接层**：输出对词典中每个词的预测概率。
- 案例演示：[**05_rnn_lyrics_generator.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/05_rnn_lyrics_generator.py)（RNN_AI歌词生成器）

# 16_RNN_AI歌词生成器案例_模型训练

- 前面的准备工作完成之后，可以编写训练函数。训练函数主要负责编写数据迭代、送入网络、计算损失、反向传播、更新参数，其流程基本较为固定。
- 由于要实现文本生成，文本生成本质上，输入一串文本，预测下一个文本，也属于分类问题，所以，使用多分类交叉熵损失函数。优化方法有 SGB、AdaGrad、Adam 等，在这里我们选择学习率、梯度自适应的 Adam 算法作为我们的优化方法。
- 训练完成之后，使用 torch.save 方法将模型持久化存储。
- 案例演示：[**05_rnn_lyrics_generator.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/05_rnn_lyrics_generator.py)（RNN_AI歌词生成器）

# 17_RNN_AI歌词生成器案例_模型测试

- 从磁盘加载训练好的模型，进行预测。预测函数，输入第一个指定的词，将该词输入网路，预测出下一个词，再将预测的出的词再次送入网络，预测出下一个词，以此类推，直到预测出指定长度的内容。
- 案例演示：[**05_rnn_lyrics_generator.py**](https://github.com/ZZHow1024/NN-DL-Learning/blob/main/Day07/05_rnn_lyrics_generator.py)（RNN_AI歌词生成器）