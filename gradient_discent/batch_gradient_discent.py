import numpy as np

# 全量梯度下降

# 超参数
n_iterator = 10000
# 指定训练集X，y
X_a = np.random.rand(100,1)
X_b = np.random.rand(100,1)
y = 5 + 6 * X_a + 3 * X_b + np.random.randn(100,1)
t0,t1 = 5,500

X = np.c_[np.ones((100,1)),X_a,X_b]

# 创建函数，随着迭代次数不断增加，学习率不断减小，更有可能趋近最优解
def learning_rate_schedule(t):
    return t0/(t+t1)

# 创建θ 标准正态分布
theta = np.random.randn(3,1)
# 不会设置阈值，直接设置超参数，迭代次数，迭代次数到了，并且当迭代次数调整也不会给θ带来收益时，我们就认为收敛了
for n in range(n_iterator):
    # 计算梯度
    gradient =X.T.dot(X.dot(theta)-y)
    # 调整theta
    learning_rate = learning_rate_schedule(n)
    theta = theta - learning_rate*gradient

print(theta)


