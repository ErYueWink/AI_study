import numpy as np

# 数据集X,y
X_a = np.random.rand(100,1)
X_b = np.random.rand(100,1)
y = 3 + 5*X_a + 6 * X_b + np.random.randn(100,1)

X = np.c_[np.ones((100,1)),X_a,X_b]

# 超参数 学习率 迭代轮次 每批次需要的数据量 样本总数
learning_rate = 0.0001
n_epoch = 10000
m = 100
batch_size = 10
number_batches = int(m/batch_size)

# 初始化θ X有三个维度
theta = np.random.randn(3,1)
for _ in range(n_epoch):
    for _ in range(number_batches):
        # 属于有放回的采样
        random_index = np.random.randint(number_batches)
        x_random = X[random_index:random_index+batch_size]
        y_random = y[random_index:random_index+batch_size]
        # 计算梯度gradient
        gradient = x_random.T.dot(x_random.dot(theta)-y_random)
        # 调整θ
        theta = theta - learning_rate * gradient

print(theta)
