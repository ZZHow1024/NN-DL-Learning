"""
案例：ANN（人工神经网络）案例——手机价格分类

背景：基于于机的 20 列特征 -> 预测手机的价格区间（4 个区问），可以用机器学习做，也可以用深度学习做（推荐）

ANN 案例的实现步骤：
    1. 构建数据集
    2. 搭建神经网络
    3. 模型训练
    4. 模型测试

优化思路
    1. 优化方法：SGD -> Adam
    2. 学习率：0.001 -> 0.0001
    3. 对数据进行标准化
    4. 增加网络的深度，每层的神经元数量
    5. 调整训练的轮数
    ...
"""
import os

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time

from torchsummary import summary


# 1. 构建数据集
def create_dateset():
    # 1.1 加载 CSV 文件数据集
    data = pd.read_csv(os.path.join('data', '手机价格预测.csv'))
    print(data.head())
    print(f'data.shape = {data.shape}')

    # 1.2 获取 x 特征列和 y 标签列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    print(f'x.shape = {x.shape}')
    print(f'y.shape = {y.shape}')

    # 1.3 把特征列转成浮点型
    x = x.astype('float32')

    # 1.4 切分训练集和测试集
    # 参数 1：特征
    # 参数 2：标签
    # 参数 3：测试集所占比例
    # 参数 4：随机种子
    # 参数 5：样本的分布（参考 y 的类别进行抽取数据）
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    # 1.5 把数据集封装成张量数据集，思路：数据 -> 张量 Tensor -> 数据集 TensorDataset -> 数据加载器 DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))

    # 1.6 返回对象
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y_train))


# 2. 搭建神经网络
class PhonePriceANN(nn.Module):
    # 2.1 在 init 魔法方法中初始化父类成员及搭建神经网络
    def __init__(self, input_dim, output_dim):
        # 2.1.1 初始化父类成员
        super(PhonePriceANN, self).__init__()
        # 2.1.2 搭建神经网络
        # 隐藏层 1
        self.linear1 = nn.Linear(input_dim, 128)
        # 隐藏层 2
        self.linear2 = nn.Linear(128, 256)
        # 隐藏层 3
        self.output = nn.Linear(256, output_dim)

    # 2.2 定义前向传播方法 forward()
    def forward(self, x):
        # 2.2.1 隐藏层 1：加权求和 + 激活函数 (ReLU)
        x = torch.relu(self.linear1(x))
        # 2.2.2 隐藏层 2：加权求和 + 激活函数 (ReLU)
        x = torch.relu(self.linear2(x))
        # 2.2.3 隐藏层 3：加权求和 + 激活函数 (Softmax)
        # 由于多分类交叉熵损失函数 CrossEntropyLoss() = Softmax + 损失计算，因此省略 Softmax 激活函数
        return self.output(x)


# 3. 模型训练
def train(train_dataset, input_dim, output_dim):
    # 3.1 创建数据加载器
    # 参数 1：数据集对象（1600 条）；参数 2：每批次的数据条数；参数 3：是否打乱数据
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3.2 创建神经网络模型
    model = PhonePriceANN(input_dim, output_dim)

    # 3.3 定义损失函数（多分类交叉熵）
    criterion = nn.CrossEntropyLoss()

    # 3.4 创建优化器对象
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # 3.5 模型训练
    # 3.5.1 定义变量，记录训练的总轮数
    epochs = 50
    # 3.5.2 开始训练
    for epoch in range(epochs):
        # 定义变量，记录每次训练的损失值、训练批次数
        total_loss, batch_num = 0.0, 0
        # 定义变量，表示训练开始的时间
        start_time = time.time()
        for x, y in train_loader:
            # 切换模型状态
            model.train()
            # 模型预测
            y_pred = model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 梯度清零、反向传播、优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累加损失值
            total_loss += loss.item()
            batch_num += 1
        # 本轮训练结束，打印训练信息
        print(f'epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start_time:.2f}s')

    # 3.6 保存模型参数
    # 参数 1：模型对象的参数（权重矩阵、偏置矩阵）；参数 2：模型保存的文件名
    torch.save(model.state_dict(), os.path.join('model', 'phone.pth'))


# 4. 模型测试
def evaluate(test_dataset, input_dim, output_dim):
    # 4.1 创建神经网络分类对象
    model = PhonePriceANN(input_dim, output_dim)

    # 4.2 加载模型参数
    model.load_state_dict(torch.load(os.path.join('model', 'phone.pth')))

    # 4.3 创建测试集的数据加载器对象
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 4.4 定义变量，记录预测正确的样本个数
    correct = 0

    # 4.5 从数据加载器中获取每批次的数据
    for x, y in test_loader:
        # 切换模型状态
        model.eval()
        # 模型预测
        y_pred = model(x)
        # 根据加权求和得到类别
        # dim = 1 表示逐行处理
        y_pred = torch.argmax(y_pred, dim=1)
        print(f'y_pred: {y_pred}')
        # 统计预测正确的样本个数
        correct += y_pred.eq(y).sum()

    # 4.6 打印准确率
    print(f'accuracy: {correct / len(test_dataset):.4f}')


if __name__ == '__main__':
    # 准备数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dateset()
    print(f'训练集：{train_dataset}')
    print(f'测试集：{test_dataset}')
    print(f'输入特征数：{input_dim}')
    print(f'输出标签数：{output_dim}')

    # 搭建神经网络模型
    model = PhonePriceANN(input_dim, output_dim)
    # 计算模型参数
    # 参数 1：模型对象；参数 2：输入数据的形状(批次大小, 输入特征数)
    summary(model, input_size=(16, input_dim))

    # 模型训练
    train(train_dataset, input_dim, output_dim)

    # 模型测试
    evaluate(test_dataset, input_dim, output_dim)
