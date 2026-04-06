"""
案例：演示梯度下降优化方法。

梯度下降相关介绍：
    概述：梯度下降是结合本次损失函数的导数（作为梯度）基于学习率 来更新权重的。
    公式：W新 = W旧 - 学习率 * 梯度
    存在的问题：
        1. 遇到严缓区域，梯度下降（权重更新）可能会慢。
        2. 可能会遇到鞍点（梯度为 0）。
        3. 可能会遇到局部最小值。
    解决思路：从上述的学习率或者梯度入手，进行优化，于是有了：动量法 Momentum，白适应学习率 AdaGrad，RMSProp，綜合衡量：Adam。

动量法 Momentum：
    动量法公式：St = β * St-1 + (1 - β) * Gt
    解释：
        St：本次的指数移动加权严均结果。
        B：调节权重系数，越大，数据越严缓，历史指数秘动加权平均比重越大，本次梯度权重越小。
        St-1：历史的指数移动加权平均结果。
        Gt：本次计算出的梯度（不考虑历史梯度）。
    加入动量法后的梯度更新公式：W新 = W旧 - 学习率 * St

自适应学习率：AdaGrad
    公式：
        累计平方梯度：St = St-1 + Gt * Gt
            解释：
                St：累计平方梯度
                St-1：历史累计平方梯度
                Gt：本次的梯度
        学习率：η = η / (sqrt(St) + σ)
            解释：σ = 1e-10，目的：防止分母变成 0
        梯度下降公式：W新 = W旧 - 调整后的学习率 * Gt
    缺点：可能会导致学习率过早，过量的降低，导致模型后期学习率太小，较难找到最优解。

均方根传播：RMSProp (Root Mean Square Propagation) -> 可以看做是 AdaGrad 做的优化，加入调和权重系数。
    公式：
        指数加权平均累计历史平方梯度：St = β * St-1 + (1 - β) * Gt * Gt
            解释：
                St：累计平方梯度
                St-1：历史累计平方梯度
                Gt：本次的梯度
                β：调和权重系数
        学习率：η = η / (sqrt(St) + σ)
            解释：σ = 1e-10，目的：防止分母变成 0
        梯度下降公式：W新 = W旧 - 调整后的学习率 * Gt
        优点：RMSProp 通过引入衰减系数 β，控制历史梯度对历史梯度信息获取的多少。

自适应矩估计：Adam (Adaptive Moment Estimation)
    思路：既优化学习率，又优化梯度。
    公式：
        一阶矩：算平均
            Mt = β1 * Mt-1 + (1 - β1) * Gt          充当梯度
            St = β2 * St-1 + (1 - β2) * Gt * Gt     充当学习率
        二阶矩：梯度的方差
            Mt^ = Mt / (1 - β1 ^ t)
            St^ = St / (q - β2 ^ t)
        权重更新公式：W新 = W旧 - 学习率 / (sqrt(St^) + σ) * Mt^
    简单解释：Adam = RMSProp + Momentum
"""
import torch
import torch.optim as optim


# 1. 演示：梯度下降优化方法 -> 动量法 (Momentum)
def demo01_momentum():
    print('演示：梯度下降优化方法 -> 动量法 (Momentum)')
    # 1.1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 1.2 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 1.3 创建优化器 -> 基于 SGD（随机梯度下降），加入参数 momentum 就是动量法
    # 参数 1：待优化的参数列表；参数 2：学习率；参数 3：动量参数（默认是 0）
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)
    # 1.4 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w = {w}, grad = {w.grad}')

    # 1.5 重复上述的步骤，第二次更新权重参数
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w = {w}, grad = {w.grad}')
    print('-' * 10)


# 2. 演示：梯度下降优化方法 -> 自适应学习率 (AdaGrad)
def demo02_adagrad():
    print('\n演示：梯度下降优化方法 -> 自适应学习率 (AdaGrad)')
    # 2.1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2.2 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 2.3 创建优化器 -> 基于 AdaGrad（自适应学习率）
    # 参数 1：待优化的参数列表；参数 2：学习率
    optimizer = optim.Adagrad(params=[w], lr=0.01)
    # 2.4 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w = {w}, grad = {w.grad}')

    # 2.5 重复上述的步骤，第二次更新权重参数
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w = {w}, grad = {w.grad}')
    print('-' * 10)


# 3. 演示：梯度下降优化方法 -> 均方根传播 (RMSProp)
def demo03_rmsprop():
    print('\n演示：梯度下降优化方法 -> 均方根传播 (RMSProp)')
    # 3.1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 3.2 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 3.3 创建优化器 -> 基于 RMSProp（均方根传播）
    # 参数 1：待优化的参数列表；参数 2：学习率
    optimizer = optim.RMSprop(params=[w], lr=0.01)
    # 3.4 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w = {w}, grad = {w.grad}')

    # 3.5 重复上述的步骤，第二次更新权重参数
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w = {w}, grad = {w.grad}')
    print('-' * 10)


# 4. 演示：梯度下降优化方法 -> 自适应矩估计 (Adam)
def demo04_adam():
    print('\n演示：梯度下降优化方法 -> 自适应矩估计 (Adam)')
    # 3.1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 3.2 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 3.3 创建优化器 -> 基于 Adam（自适应矩估计）自适应矩估计
    # 参数 1：待优化的参数列表；参数 2：学习率；参数 3：betas=(梯度用的衰减系数, 学习率用的衰减系数)
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.99))
    # 3.4 计算梯度值：梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w = {w}, grad = {w.grad}')

    # 3.5 重复上述的步骤，第二次更新权重参数
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w = {w}, grad = {w.grad}')


if __name__ == '__main__':
    demo01_momentum()
    demo02_adagrad()
    demo03_rmsprop()
    demo04_adam()
