import numpy as np

# 创建数据集X,y
X_a = np.random.rand(100,1)
X_b = np.random.rand(100,1)
y = 3 + 5*X_a + 6*X_b + np.random.randn(100,1)
# 给X添加恒为1的一列，为了求W0截距项
X = np.c_[np.ones((100,1)),X_a,X_b]

# 超参数 学习率η 迭代次数
learning_rate = 0.0001
n_iterator = 10000

"""
    梯度下降流程:
        1. 随机θ
        2. 计算梯度
        3. 调整θ g>0 theta变小 g<0 theta变大
        4. 判断是否收敛
"""
# 随机theta 标准正态分布
theta = np.random.randn(3, 1)
# 使用较大的迭代次数判断theta是否收敛，当迭代次数增加并不会为theta带来收益时，我们就认为收敛了
for _ in range(n_iterator):
    # 计算出每个维度对应的梯度 3*1 分别对应W0 W1 W2
    gradient = X.T.dot(X.dot(theta) - y)
    # 调整theta θt+1=θt - η*gradient
    theta = theta - learning_rate*gradient
print(theta)
