# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# 创建示例数据
# 假设我们有5个房屋的数据，其中X是房屋的面积，y是房屋的价格
X_train = np.array([[80], [100], [120], [150], [200]])  # 房屋面积（单位：平方米）
print(X_train)
y_train = np.array([300000, 350000, 400000, 450000, 500000])  # 房屋价格（单位：人民币）

# 定义 KNN 回归模型，这里我们选择 K=3
knn_regressor = KNeighborsRegressor(n_neighbors=3)

# 使用训练数据拟合模型
knn_regressor.fit(X_train, y_train)

# 生成一些测试数据（房屋面积）
X_test = np.arange(80, 201, 10).reshape(-1, 1)

# 使用模型进行预测
y_pred = knn_regressor.predict(X_test)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='darkorange', label='Training data')
plt.plot(X_test, y_pred, color='navy', label='Prediction')
plt.xlabel('House Area (sqm)')
plt.ylabel('House Price (RMB)')
plt.title('KNN Regression: House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()