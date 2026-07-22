"""
案例：演示词嵌入层的 API 应用

RNN：Recurrent Neural Network，循环神经网络，主要处理序列数据。
    组成：
        词潜入层
        循环网络层
        输出层

序列数据：后边数据对前边数据有依赖，例如天气预测、股市分析、文本生成。

词嵌入层介绍：把词转为词向量。
"""
import torch
import jieba
import torch.nn as nn


# 演示：词嵌入层的 API（词 -> 词向量）
def demo():
    # 1. 定义一句话
    text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'

    # 2. 使用 jieba 模块进行分词
    words = jieba.lcut(text)
    print(f'分词结果：{words}')

    # 3. 创建词嵌入层
    # 参数 1：词表大小；参数 2：词向量的维度
    embed = nn.Embedding(len(words), 4)

    # 4. 获取每个词对象的下标索引
    for i, word in enumerate(words):
        # 5. 把词索引转成词向量
        word_vector = embed(torch.tensor(i))
        print(f'词：{word}\t\t词向量：{word_vector}')


if __name__ == '__main__':
    demo()
