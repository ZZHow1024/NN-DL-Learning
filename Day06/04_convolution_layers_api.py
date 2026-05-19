"""
案例：演示卷积层的 API，用于提取图像的局部特征，获取特征图。

卷积神经网络介绍：
    概述：包含卷积层的神经网络，Convolutional neural network。
    组成：
        卷积层 (Convolutional)：用于提取图像的局部特征，结合卷积核（每个卷积核 = 1 个神经元）实现，处理后的结果为特征图。
        池化层 (Pooling)：用于降维，降采样。
        全连接层 (Full Connected, fc, linear)：用于预测结果，并输出结果的。
    特征图计算方式：
        N = (W - F + 2*P) / S + 1
        W：输入图像的大小
        F：卷积核的大小
        P：填充的大小
        S：步长
        N：输出图像的大小（特征图大小）
"""
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 演示：图像的加载、卷积和特征图的可视化操作
def demo01():
    # 1. 加载 RGB 真彩图
    img = plt.imread(os.path.join('data', 'image.jpg'))

    # 2. 打印图像信息
    print(f'img = {img}, shape: {img.shape}')

    # 3. 转换图像的形状 HWC -> CHW
    img2 = torch.tensor(img, dtype=torch.float)
    img2 = img2.permute(2, 0, 1)
    print(f'img2 = {img2}, shape: {img2.shape}')

    # 4. 因为这里只有 1 张图，再增加 1 个维度，(C, H, W) -> (1, C, H, W)，1 张 3 通道的 640*640 像素的图
    img3 = img2.unsqueeze(dim=0)
    print(f'img3 = {img3}, shape: {img3.shape}')

    # 5. 创建卷积层对象，提取特征图
    # 参数 1：输入图像的通道数；参数 2：输出图像的通道数；参数 3：卷积核的大小；参数 4：步长；参数 5：填充
    conv = nn.Conv2d(3, 4, 3, 2, 0)

    # 6. 具体的卷积计算
    conv_img = conv(img3)

    # 7. 打印卷积后的结果
    print(f'conv_img = {conv_img}, shape: {conv_img.shape}')

    # 8. 查看提取到的 4 个特征图
    img4 = conv_img[0]
    print(f'img4 = {img4}, shape: {img4.shape}')

    # 9. 转换图 CHW -> HWC
    img5 = img4.permute(1, 2, 0)
    print(f'img5 = {img5}, shape: {img5.shape}')

    # 10. 可视化第 1 个通道的特征图
    for i in range(4):
        feature1 = img5[:, :, i].detach().numpy()
        plt.imshow(feature1)
        plt.show()


if __name__ == '__main__':
    demo01()
