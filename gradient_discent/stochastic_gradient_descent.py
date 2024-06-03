import numpy as np

# 随机梯度下降

# 超参数
m = 100
t0,t1 = 5,500
n_epochs = 10000

# 定义数据集，X，y
X_a = np.random.rand(100,1) # [0,1)
X_b = np.random.rand(100,1) # [0,2)
y = 5 + 3 * X_a + 6 * X_b + np.random.randn(100,1)

X = np.c_[np.ones((100,1)),X_a,X_b]

# 定义函数动态更改学习率
def learning_rate_schedule(t):
    return t0 / (t+t1)

# 定义θ标准正态分布
theta = np.random.randn(3,1)


for epoch in range(n_epochs):
    # 打乱X，y顺序，既要保证随机性，也要保证可以取到所有样本
    arr = np.arange(len(X))
    np.random.shuffle(arr)
    X = X[arr]
    y = y[arr]
    for n in range(m):
        learning_rate = learning_rate_schedule(n*m+n)
        # 计算梯度
        x_stochastic = X[n:n+1]
        y_stochastic = y[n:n+1]
        # 100 * 3 3*1 100*1 100*3 3*100  100*1 3*1
        gradient = x_stochastic.T.dot(x_stochastic.dot(theta)-y_stochastic)
        theta = theta - learning_rate * gradient

print(theta)