import numpy as np

# 小批量梯度下降

# 超参数
m = 100
batch_size = 10
number_size = int(m/batch_size)
n_epochs = 10000
t0,t1 = 5,500

# 训练集，X，y
X_a = 3 * np.random.rand(100,1) # [0,3)
X_b = 5 * np.random.rand(100,1) # [0,5)
y = 5 + 6 * X_a + 3 * X_b + np.random.randn(100,1)

X = np.c_[np.ones((100,1)),X_a,X_b]

def learning_rate_schedule(t):
    return t0 / (t+t1)

# 标准正态分布
theta = np.random.randn(3,1)

for epoch in range(n_epochs):
    arr = np.arange(len(X))
    np.random.shuffle(arr)
    X = X[arr]
    y = y[arr]
    for n in range(number_size):
        learning_rate = learning_rate_schedule(epoch*m + n)
        x_mini = X[n*batch_size:n*batch_size+batch_size]
        y_mini = y[n*batch_size:n*batch_size+batch_size]
        gradients = x_mini.T.dot(x_mini.dot(theta)-y_mini)
        theta = theta - learning_rate*gradients

print(theta)
