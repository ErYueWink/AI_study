import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale


# 准备数据集
data = load_breast_cancer()
# 取出数据集X，y
X,y = scale(data['data'][:,:2]),data['target']

logistic = LogisticRegression(fit_intercept=False)
# 计算θ
logistic.fit(X,y)
w1 = logistic.coef_[0,0]
w2 = logistic.coef_[0,1]

# 计算y_hat
# 已知w1,w2的情况下，传进来数据的x，返回数据的y_predict
def p_theta_function(features,w1,w2):
    z = w1*features[0] + w2*features[1]
    return 1/(1+np.exp(-z))

# 传入一份已知的数据X，y 如果已知w1,w2的情况下，计算出这份数据的损失
def loss_function(samples_features,samples_label,w1,w2):
    result = 0
    for feature,label in zip(samples_features,samples_label):
        # 一条样本的y_hat
        p_result = p_theta_function(feature,w1,w2)
        loss_result = -1*label*np.log(p_result)-(1-label)*np.log(1-p_result)
        result += loss_result
    return result

theta1_space = np.linspace(w1-0.6,w1+0.6,50)
theta2_space = np.linspace(w2-0.6,w2+0.6,50)

result1_ = np.array([loss_function(X,y,i,w2) for i in theta1_space])
result2_ = np.array([loss_function(X,y,w1,i) for i in theta2_space])

fig = plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
plt.plot(theta1_space,result1_)

plt.subplot(2,2,2)
plt.plot(theta2_space,result2_)


# 探索两个参数和损失函数之间的变换关系
plt.subplot(2,2,3)
theta1_grid,theta2_grid = np.meshgrid(theta1_space,theta2_space)
loss_grid = loss_function(X,y,theta1_grid,theta2_grid)
# 等高线 x轴，y轴，纵轴
plt.contour(theta1_grid,theta2_grid,loss_grid)

plt.subplot(2,2,4)
plt.contour(theta1_grid,theta2_grid,loss_grid,30)

# 第二张画布
fig2 = plt.figure(figsize=(8,6))
ax = Axes3D(fig2)
fig2.add_axes(ax)
ax.plot_surface(theta1_grid, theta2_grid, loss_grid)
plt.show()


