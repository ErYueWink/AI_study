import numpy as np

# 训练集X，y
X_a = np.random.rand(100,1)
X_b = np.random.rand(100,1)
# 真实的y y=θ0+θ1x1+...+Θnxn+error (考虑截距项的情况下)
y = 3 + 5*X_a + 6*X_b + np.random.randn(100,1)
# 为X添加恒为1的一列 计算W0截距
X = np.c_[np.ones((100,1)),X_a,X_b]

# 超参数 学习率 迭代轮次 样本总数
learning_rate = 0.0001
n_epoch = 10000
m = 100

# 初始化θ x有3个特征值
theta = np.random.randn(3,1)
for _ in range(n_epoch):
    """
        在代码实现随机梯度下降中我们使用的方式是有放回的采样，即有可能使用到相同的样本去调整θ,有些样本不会参与到调整θ中
        我们应该怎么解决这一问题呢?
    """
    # 在每个轮次的分批次迭代之前打乱数组顺序 保证随机性
    arr = np.arange(len(X))
    np.random.shuffle(arr)
    X = X[arr]
    y = y[arr]
    # 分批次迭代 在随机梯度下降中批次迭代多少次是由样本数决定的
    for a in range(m):
        x_random = X[a:a+1]
        y_random = y[a:a+1]
        # 计算梯度gradient
        gradient = x_random.T.dot(x_random.dot(theta)-y_random)
        # 调整θ θt+1=Θt-η*gradient
        theta = theta-learning_rate*gradient

print(theta)
