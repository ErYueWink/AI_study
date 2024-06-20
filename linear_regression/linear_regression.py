import numpy as np

# 训练集X,y
X_a = 2 * np.random.rand(100,1)
X_b = 2 * np.random.rand(100,1)
# 计算y值
# 我们假设误差服务期望为0 方差为某个定值的正态分布
y = 3 + 5 * X_a + 7*X_b + np.random.randn(100,1)
print(y)
# 为X添加恒为1的一列，求W0截距
X = np.c_[np.ones((100,1)),X_a,X_b]
# 根据推导出的解析解公式求解θ
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
"""
    [[3.13710358]
    [4.92589159]
    [6.94505671]]
"""
print(theta)

# 给定测试集数据
# 怎么训练模型模型就具备什么功能，在训练集中我们的X是有两个特征的，X_model中也应该有两个特征(维度)
X_model = np.array([[0,0],
                    [2,3]])
# 计算预测值
y_predict = X.dot(theta)
print(y_predict)