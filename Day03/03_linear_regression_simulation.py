import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn  # nn 模块中有平方损失函数和假设函数
from torch import optim  # optim 模块中有优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集
import matplotlib.pyplot as plt  # 可视化
import zhplot  # matplotlib 汉化库

zhplot.matplotlib_chineseize()  # matplotlib 汉化


# 定义函数，创建线性回归样本数据
def create_dataset():
    x, y, coef = make_regression(
        n_samples=100,  # 100 个样本点
        n_features=1,  # 1 个特征点
        noise=10,  # 噪声。噪声越大，样本点约分散
        coef=True,  # 是否返回系数
        bias=14.5,  # 偏置
        random_state=1  # 随机种子
    )

    # 把数据封装成张量对象
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    coef = torch.tensor(coef, dtype=torch.float)

    # 返回结果
    return x, y, coef


# 定义函数，模型训练
def train(x, y, coef):
    # 创建数据集对象，把 tensor -> 数据集对象 -> 数据加载器
    dataset = TensorDataset(x, y)
    # 创建数据加载器对象
    # 参数 1：数据集对象，参数 2：批次大小，参数 3：是否打乱数据（训练集打乱，测试集不打乱）
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # 创建初始的线性回归模型
    # 参数 1：输入特征维度，参数 2：输出特征维度
    model = nn.Linear(1, 1)
    # 创建损失函数对象
    criterion = nn.MSELoss()
    # 创建优化器对象
    # 参数 1：模型参数，参数 2：学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 具体的训练过程
    # 定义变量（训练轮数、每轮的平均损失值、训练总损失值 和 训练的样本数）
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    # 开始训练，按轮训练
    for epoch in range(epochs):
        # 从数据加载器中获取批次数据
        for train_x, train_y in dataloader:
            # 模型预测
            y_pred = model(train_x)
            # 计算损失值
            loss = criterion(y_pred, train_y.reshape(-1, 1))  # -1 自动计算行，1 列
            # 计算总损失
            total_loss += loss.item()
            total_sample += 1
            # 梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 把本轮的平均损失值添加到列表中
        loss_list.append(total_loss / total_sample)
        print(f'epoch = {epoch + 1}, loss = {total_loss / total_sample}')

    # 打印最终的训练结果
    print(f'{epochs} 轮的平均损失分别为 {loss_list}')
    print(f'模型参数，权重 = {model.weight}，偏置 = {model.bias}')

    # 绘制损失曲线
    plt.plot(range(epochs), loss_list)
    plt.title('损失值曲线变化图')
    plt.grid(True)
    plt.show()

    # 绘制样本点分布情况
    plt.scatter(x, y)
    # 100 个样本点的特征
    y_pred = torch.tensor(data=[v * model.weight + model.bias for v in x], dtype=torch.float)
    # 计算真实值
    y_true = x * coef + 14.5
    # 绘制预测值和真实值的折线图
    plt.title('预测值和真实值的折线图')
    plt.plot(x, y_pred, color='red', label='预测值')
    plt.plot(x, y_true, color='blue', label='真实值')
    plt.legend()  # 图例
    plt.grid(True)  # 网格
    plt.show()


if __name__ == '__main__':
    x, y, coef = create_dataset()
    print(f'x = {x}')
    print(f'y = {y}')
    print(f'coef = {coef}')

    train(x, y, coef)
