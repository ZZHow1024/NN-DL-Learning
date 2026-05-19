"""
案例：演示基础的图像操作

图像分类：
    二值图：1 通道，每个像素点由 1 或 0 组成
    灰度图：1 通道，每个像素点的范围：[0, 255]
    索引图：1 通道，每个像素点的范围：[0, 255]，像素点表示颜色表的案引
    RGB 真彩图：三通道，Red、Green、BLue（红绿蓝）

API：
    imshow()    基于 HWC 展示图像
    imread()    读取图像，获取 HWC
    imsave()    基于 HWC 保存图片
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import torch


# 演示：绘制全黑、全白图
def demo01():
    # 定义全黑图片
    # HWC：H 高度、W 宽度、C 通道
    img1 = np.zeros((200, 200, 3))

    # 绘制图像
    plt.imshow(img1)
    plt.axis('off')
    plt.show()

    # 定义全白图片
    img2 = torch.full(size=(200, 200, 3), fill_value=255)
    plt.imshow(img2)
    plt.axis('off')
    plt.show()


# 演示：加载图片
def demo02():
    print(f'演示：加载图片')
    # 加载图片
    img = plt.imread(os.path.join('data', 'image.jpg'))
    print(f'img1 = {img}')
    print(f'img1.shape = {img.shape}')

    # 保存图像
    plt.imsave(os.path.join('data', 'image_copy.jpg'), img)

    # 展示图像
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    demo01()
    demo02()
