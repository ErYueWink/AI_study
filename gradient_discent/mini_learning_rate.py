import numpy as np

"""
    代码改进 随着迭代次数不断增多 学习率不断减小
    在迭代次数不断增多时 我们会距离最优解越来越近 我们应该减小步长 会更有可能找到最优解
"""

# 数据集X，y
X_a = np.random.rand(100,1)
X_b = np.random.rand(100,1)
y = 3 + 5*X_a + 6*X_b + np.random.randn(100,1)

X = np.c_[np.ones((100,1)),X_a,X_b]

# 超参数
t0,t1 = 5,500
m = 100
batch_size = 10
number_batches = int(m/batch_size)
n_epochs = 10000

# 初始化θ X有三个特征
theta = np.random.randn(3,1)

# 定义函数 随着迭代次数不断增多学习率不断减小
def learning_rate(t):
    return t0/(t+t1)

for epoch in range(n_epochs):
    # 在每个轮次分批次迭代前打乱数组顺序 保证随机性
    # 打乱数组顺序确保每条样本可以参与到调整theta中
    arr = np.arange(len(X))
    np.random.shuffle(arr)
    X = X[arr]
    y = y[arr]
    for a in range(number_batches):
        x_random = X[a*batch_size:a*batch_size+batch_size]
        y_random = y[a*batch_size:a*batch_size+batch_size]
        rate = learning_rate(epoch*m+a)
        # 计算梯度
        gradient = x_random.T.dot(x_random.dot(theta)-y_random)
        # 调整θ θt+1=θt-η*gradient
        theta = theta - rate * gradient

print(theta)
