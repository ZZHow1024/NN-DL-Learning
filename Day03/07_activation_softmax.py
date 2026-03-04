"""
案例：演示 Softmax 激活函数。

Softmax 激活函数：将多分类的结果以概率的形式展示，且概率和相加为 1，最终选取概率值最大的分类作为最终结果。
"""
import torch

scores = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
# dim = 0，按行计算
probabilities = torch.softmax(scores, dim=0)
print(probabilities)
