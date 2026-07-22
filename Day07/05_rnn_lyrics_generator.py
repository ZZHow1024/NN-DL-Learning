"""
案例：RNN 案例，基于结论歌词训练模型，用给定的起始词，结合长度，进行 AI 歌词生成。

实现步骤：
    1. 获取数据，进行分词，获取词表
    2. 数据预处理，构建数据集
    3. 搭建 RNN 神经网络
    4. 模型训练
    5. 模型预测
"""
import os

import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

# 自动获取当前环境的最佳加速器
if hasattr(torch, "accelerator") and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
elif torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"训练设备：{device}")


# 1. 获取数据，进行分词，获取词表
def build_vocab():
    # 1.1 定义变量，记录去重后所有词、每行文本分词结果
    unique_words, all_words = [], []

    # 1.2 遍历数据集，获取每行文本
    for line in open(os.path.join('data', 'jaychou_lyrics.txt'), 'r', encoding='utf-8'):
        # 获取每行歌词，进行分词
        words = jieba.lcut(line)
        # 所有分词结果记录到 all_words 中
        all_words.append(words)
        # 遍历分词结果，去重后，添加到 unique_words 中
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    # 1.3 统计语料中词的数量
    word_count = len(unique_words)

    # 1.4 构建词表（字典形式），key 是词，value 是词的索引
    word_to_index = {word: i for i, word in enumerate(unique_words)}

    # 1.5 歌词文本用词表索引表示
    corpus_idx = []

    # 1.6 遍历每一行的分词结果
    for words in all_words:
        # 定义变量，记录词索引表
        tmp = []
        # 获取每一行的词，并获取相应的索引
        for word in words:
            tmp.append(word_to_index[word])
        # 在每行之间，添加空格隔开
        tmp.append(word_to_index[' '])
        # 获取文档中每个词的索引，添加到 corpus_idx 中
        corpus_idx.extend(tmp)

    # 1.7 返回结果
    return unique_words, word_to_index, word_count, corpus_idx


# 2. 数据预处理，构建数据集
class LyricsDataset(torch.utils.data.Dataset):
    # 2.1 初始化词索引等
    def __init__(self, corpus_idx, num_chars):
        # 文档数据中词的索引
        self.corpus_idx = corpus_idx
        # 每个句子中词的个数
        self.num_chars = num_chars
        # 文档数据中词的数量（不去重）
        self.word_count = len(self.corpus_idx)
        # 句子数量
        self.number = self.word_count // self.num_chars

    # 2.2 获取句子数量（使用 len(obj) 时自动调用）
    def __len__(self):
        return self.number

    # 2.3 通过词索引获取词（使用 obj[index] 时自动调用）
    def __getitem__(self, idx):
        # 当前样本的起始索引
        start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        # 当前样本的结束索引
        end = start + self.num_chars
        # 输入值
        x = self.corpus_idx[start:end]
        # 输出值
        y = self.corpus_idx[start + 1: end + 1]
        # 返回输入值和输出值
        return torch.tensor(x), torch.tensor(y)


# 3. 搭建 RNN 神经网络
class RNNModel(nn.Module):
    # 3.1 初始化方法
    def __init__(self, unique_word_count):
        # 初始化父类成员
        super(RNNModel, self).__init__()
        # 初始化词嵌入层，
        # 参数 1：语料中词的数量；参数 2：词向量的维度
        self.ebd = nn.Embedding(unique_word_count, 128)
        # 循环网络层
        # 参数 1：词向量维度；参数 2：隐藏层维度；参数 3：网络层数
        self.rnn = nn.RNN(128, 256, 1)
        # 输出层
        # 参数 1：特征向量维度；参数 2：词表中词的个数
        self.out = nn.Linear(256, unique_word_count)

    # 3.2 前向传播方法
    def forward(self, inputs, hidden):
        # 初始化（词嵌入层处理）
        # 返回值 (batch 句子的数量, 句子的长度, 词向量维度)
        embed = self.ebd(inputs)
        # RNN 处理
        # 参数 (句子的长度, batch 句子的数量, 隐藏层维度)
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)
        # 全连接层
        # 输入维度：(seq_len 句子数量 * batch, 词向量维度 256)
        # 输出维度：(seq_len 句子数量 * batch, 词表中词的个数)
        output = self.out(output.reshape(shape=(-1, output.shape[-1])))
        # 返回结果
        return output, hidden

    # 3.3 隐藏层初始化方法
    def init_hidden(self, batch_size):
        # 隐藏层初始化
        # [网络层数, batch, 隐藏层向量维度]
        return torch.zeros(1, batch_size, 256).to(device)


# 4. 模型训练
def train():
    # 4.1 构建词典
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 4.2 获取数据集
    lyrics_dataset = LyricsDataset(corpus_idx, 32)
    # 4.3 初始化模型
    model = RNNModel(unique_word_count).to(device)
    # 4.4 创建数据加载器对象
    # 参数 1：数据集对象；参数 2：批次大小；参数 3：是否打乱数据
    lyrics_dataloader = DataLoader(lyrics_dataset, batch_size=5, shuffle=True)
    # 4.5 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 4.6 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 4.7 模型训练
    epochs = 10
    for epoch in range(epochs):
        # 训练总损失，迭代次数，训练总损失
        total_loss, count, start = 0.0, 0, time.time()
        # 获取每个
        for inputs, labels in lyrics_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 切换模型状态
            model.train()
            # 获取隐藏层初始值
            hidden = model.init_hidden(5)
            # 模型计算
            outputs, hidden = model(inputs, hidden)
            # 计算损失
            labels = torch.transpose(labels, 0, 1).reshape(shape=(-1,))
            loss = criterion(outputs, labels)
            # 梯度清零 + 反向传播 + 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累计损失
            total_loss += loss.item()
            count += 1
        # 打印本轮训练信息
        print(
            f'epoch {epoch}, loss {total_loss / count:.3f}, lr {optimizer.param_groups[0]["lr"]}, time: {time.time() - start:.2f}s')
    # 4.8 保存模型
    torch.save(model.state_dict(), os.path.join('model', 'rnn-model.pth'))


# 5. 模型预测
def evaluate(start_word, sentence_length):
    # 5.1 构建词典
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()

    # 5.2 获取模型
    model = RNNModel(unique_word_count).to(device)
    model.eval()

    # 5.3 加载模型参数
    model.load_state_dict(torch.load(os.path.join('model', 'rnn-model.pth')))

    # 5.4 获取隐藏层初始值
    hidden = model.init_hidden(1)

    # 5.5 将输入的开始词转换成索引
    word_idx = word_to_index[start_word]

    # 5.6 定义列表，存放产生的词的索引
    generate_sentence = [word_idx]

    # 5.7 遍历句子长度，获取到每一个词
    for i in range(sentence_length):
        # 模型预测
        # 参数：(batch, word, hidden)
        output, hidden = model(torch.tensor([[word_idx]]).to(device), hidden)
        # 获取预测结果
        word_idx = torch.argmax(output).item()
        # 把预测结果添加到列表
        generate_sentence.append(word_idx)

    # 5.8 打印生成的句子
    print('\n\n' + '=' * 30)
    for word_idx in generate_sentence:
        print(unique_words[word_idx], end='')
    print('\n' + '=' * 30)


if __name__ == '__main__':
    # 1. 获取数据，进行分词，获取词表
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    print(f'词的数量：{word_count}')
    print(f'去重后的词：{unique_words}')
    print(f'每个词的索引：{word_to_index}')
    print(f'文档中每个词对应的索引：{corpus_idx}')

    # 2. 数据预处理，构建数据集
    dataset = LyricsDataset(corpus_idx, 5)
    print(f'句子数量：{len(dataset)}')

    # 3. 搭建 RNN 神经网络
    model = RNNModel(word_count)
    # 查看参数
    for name, parameter in model.named_parameters():
        print(f'参数名称：{name}，参数维度：{parameter.shape}')

    # 4. 模型训练
    # train()

    # 5. 模型预测
    evaluate('星星', 50)
