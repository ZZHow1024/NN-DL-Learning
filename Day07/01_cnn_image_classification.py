"""
案例：演示 CNN 图像分类综合案例

回顾：深度学习项日的步骤
    1. 准备数据集
        使用计算机视觉模块 torchvision 白带的 CIFAR10 数据集，包含 6W 张 (32,32,3) 的图片，5W 张训练集，1W 张测试集，10 个分类，每个分类 6K 张图片。
    2. 搭建（卷积）神经网络。
    3. 模型训练。
    4. 模型测试。

卷积层：
    提取图像的局部特征 -> 特征图
    计算方式：N = (W - F + 2P) // S + 1
    每个卷积核都是 1 个神经元

池化层：
    降维，有最大池化和平均池化
    池化只在 HW 上做调整，通道上不改变

优化思路：
    1. 增加卷积核的输出通道（卷积核的数量）
    2. 增加全连接层的参数量
    3. 调整学习率
    4. 调整优化方法
    5. 调整激活函数
    ...
"""
import os.path

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8

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


# 准备数据集
def create_dataset():
    # 1. 获取训练集
    # 参数 1：数据集路径；参数 2：是否是训练集；参数 3：数据预处理；参数 4：是否联网下载
    train_dataset = CIFAR10(root='data', train=True, transform=ToTensor(), download=True)

    # 2. 获取测试集
    test_dataset = CIFAR10(root='data', train=False, transform=ToTensor(), download=True)

    # 3. 返回数据集
    return train_dataset, test_dataset


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


if __name__ == '__main__':
    # 1. 准备数据集
    train_dataset, test_dataset = create_dataset()
    print(f'训练集：{train_dataset.data.shape}')
    print(f'测试集：{test_dataset.data.shape}')
    print(f'数据集类别：{train_dataset.class_to_idx}')

    # 2. 搭建卷积神经网络
    model = CNNModel()
    # 查看模型参数量
    # 参数 1：模型；参数 2：输入维度 (CHW)；参数 3：批次大小
    summary(model, input_size=(3, 32, 32), batch_size=BATCH_SIZE)

    # 3. 模型训练
    train(train_dataset)

    # 4. 模型测试
    evaluate(test_dataset)
