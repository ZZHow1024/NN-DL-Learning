"""
案例：演示参数初始化的 7 种方式。

参数初始化的目的：
    1. 防止梯度消失或者梯度爆炸。
    2. 提高收敛速度。
    3. 打破对称性。

参数初始化的方式：
    无法打破对称性的：全 0、全 1、固定值
    可以打破对称性的：随机初始化、正态分布初始化、Kaiming 初始化、Xavier 初始化
"""
import torch.nn as nn


# 1. 均匀分布随机初始化
def test01():
    print('1. 均匀分布随机初始化')
    linear = nn.Linear(5, 3)
    # 从0-1均匀分布产生参数
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)
    print('-' * 10)


# 2. 固定初始化
def test02():
    print('\n2. 固定初始化')
    linear = nn.Linear(5, 3)
    nn.init.constant_(linear.weight, 5)
    print(linear.weight.data)
    print('-' * 10)


# 3. 全 0 初始化
def test03():
    print('\n3. 全 0 初始化')
    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight)
    print(linear.weight.data)
    print('-' * 10)


# 4. 全 1 初始化
def test04():
    print('\n4. 全 1 初始化')
    linear = nn.Linear(5, 3)
    nn.init.ones_(linear.weight)
    print(linear.weight.data)
    print('-' * 10)


# 5. 正态分布随机初始化
def test05():
    print('\n5. 正态分布随机初始化')
    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight, mean=0, std=1)
    print(linear.weight.data)
    print('-' * 10)


# 6. Kaiming 初始化
def test06():
    print('\n6. Kaiming 初始化')
    # Kaiming 正态分布初始化
    print('Kaiming 正态分布初始化')
    linear = nn.Linear(5, 3)
    nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
    print(linear.weight.data)

    # Kaiming 均匀分布初始化
    print('Kaiming 均匀分布初始化')
    linear = nn.Linear(5, 3)
    nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
    print(linear.weight.data)
    print('-' * 10)


# 7. Xavier 初始化
def test07():
    print('\n7. Xavier 初始化')
    # Xavier 正态分布初始化
    print('Xavier 正态分布初始化')
    linear = nn.Linear(5, 3)
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight.data)

    # Xavier 均匀分布初始化
    print('Xavier 正态分布初始化')
    linear = nn.Linear(5, 3)
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight.data)


if __name__ == '__main__':
    test01()
    test02()
    test03()
    test04()
    test05()
    test06()
    test07()
