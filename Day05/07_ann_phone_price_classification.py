"""
案例：ANN（人工神经网络）案例——手机价格分类

背景：基于于机的 20 列特征 -> 预测手机的价格区间（4 个区问），可以用机器学习做，也可以用深度学习做（推荐）

ANN 案例的实现步骤：
    1. 构建数据集
    2. 搭建神经网络
    3. 模型训练
    4. 模型测试
"""
import os

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

# 3. 模型训练

# 4. 模型测试

if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dateset()
    print(f'训练集：{train_dataset}')
    print(f'测试集：{test_dataset}')
    print(f'输入特征数：{input_dim}')
    print(f'输出标签数：{output_dim}')
