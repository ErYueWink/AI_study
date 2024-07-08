from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('./data/shuju.csv')
print(data.head(n=5))

x = data['bochang']
y = data['xiangduiguangqiang']

x_train = np.array(x).reshape(-1,1)
print(x_train)
y_train = np.array(y)
print(y_train)

# 定义 KNN 回归模型，这里我们选择 K=3
knn_regressor = KNeighborsRegressor(n_neighbors=3)

# 使用训练数据拟合模型
knn_regressor.fit(x_train, y_train)
# 准备一些测试数据
x_test = np.arange(1548, 1552, 0.9).reshape(-1, 1)
# 预测
y_pred = knn_regressor.predict(x_test)
print(y_pred)
#
# # 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='darkorange', label='Training data')
plt.plot(x_test, y_pred, color='navy', label='Prediction')
plt.xlabel('bochang')
plt.ylabel('xiangduiguangqiang')
plt.title('xiangduiguangqiang')
plt.legend()
plt.grid(True)
plt.show()