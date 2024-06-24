import numpy as np

# 数据集X，y
X_a = np.random.rand(100,1)
X_b = np.random.rand(100,1)
y = 3 + 5*X_a + 6*X_b + np.random.randn(100,1)

X = np.c_[np.ones((100,1)),X_a,X_b]
# 超参数 学习率 迭代的轮次 样本总数(随机梯度下降每个轮次中的批次为样本总数)
learning_rate = 0.0001
n_epoch = 10000
m = 100

# 随机θ 我们的特征项有3项 θ也应该为3项
theta = np.random.randn(3,1)
for _ in range(n_epoch):
    for _ in range(m):
        # 属于有放回的采样
        random_index = np.random.randint(m)
        # 包头不包尾 x和y要一一对应
        x_random = X[random_index:random_index+1]
        y_random = y[random_index:random_index+1]
        # 计算某个维度的梯度gradient
        gradient = x_random.T.dot(x_random.dot(theta)-y_random)
        # 调整θ
        theta = theta - learning_rate*gradient
print(theta)


