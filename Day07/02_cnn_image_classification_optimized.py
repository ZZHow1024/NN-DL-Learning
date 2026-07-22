"""
案例：演示 CNN 图像分类综合案例（优化版）

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
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 128, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(128 * 6 * 6, 2048)
        self.linear2 = nn.Linear(2048, 2048)
        self.out = nn.Linear(2048, 10)
        # Dropout层，p表示神经元被丢弃的概率
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # 由于最后一个批次可能不够 32，所以需要根据批次数量来 flatten
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        # dropout正则化
        # 训练集准确率远远高于测试准确率,模型产生了过拟合
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        return self.out(x)


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
    torch.save(model.state_dict(), os.path.join('model', 'cnn-model-optimized.pth'))


# 模型测试
def evaluate(test_dataset):
    # 1. 创建测试集数据加载器
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 创建模型对象
    model = CNNModel().to(device)

    # 3. 加载模型参数
    model.load_state_dict(torch.load(os.path.join('model', 'cnn-model-optimized.pth')))

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
