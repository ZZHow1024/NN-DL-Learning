"""
案例：演示神经网络搭建流程。

深度学习案例的 4 个步骤：
    1. 准备数据
    2. 搭建神经网络
    3. 模型训练
    4. 模型测试

神经网络搭建流程：
    1. 定义一个类，继承：nn.Module
    2. 在 __init__() 方法中，搭建神经网络
    3. 在 forward() 方法中，完成：前向传播
"""
import torch
import torch.nn as nn
from torchsummary import summary


# 搭建神经网络，即自定义即成 nn.Module
class Net(nn.Module):
    # 1. 在 __init__() 魔法方法中完成初始化：父类初始化和神经网络搭建
    def __init__(self):
        # 1.1 初始化父类成员
        super().__init__()
        # 1.2 搭建神经网络 -> 隐藏层 + 输出层
        # 隐藏层 1：输出特称 3，输出特征 3
        self.linear1 = nn.Linear(3, 3)
        # 隐藏层 2：输出特称 3，输出特征 2
        self.linear2 = nn.Linear(3, 2)
        # 输出层：输出特称 2，输出特征 2
        self.output = nn.Linear(2, 2)

        # 1.3 对隐藏层进行参数初始化
        # 隐藏层 1
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        # 隐藏层 2
        nn.init.kaiming_normal(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    # 2. 前向传播：输入层 -> 隐藏层 -> 输出层
    def forward(self, x):
        # 2.1 第一层 隐藏层计算：加权求和 + 激活函数(Sigmoid)
        x = torch.sigmoid(self.linear1(x))
        # 2.2 第二层 隐藏层计算：加权求和 + 激活函数(ReLU)
        x = torch.relu(self.linear2(x))
        # 2.3 第三层 输出层计算：加权求和 + 激活函数(Softmax)
        x = torch.softmax(self.output(x), dim=-1) # dim=-1表示最后一个维度，按行处理

        # 2.4 返回预测值
        return x
