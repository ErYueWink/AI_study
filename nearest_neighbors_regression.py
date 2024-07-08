# 最近邻回归
# 导入所需要的库
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
# 决策树 梯度升级 可以用于拟合非线性数据
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv('./data/shuju.csv')
print(data.head(n=6))

# EDA数据探索
# plt.hist(np.log1p(data['xiangduiguangqiang']))
# plt.show()

x = data.drop('xiangduiguangqiang',axis=1)
print(x)
y = data['xiangduiguangqiang']
print(y)
# 对x和y的数据进行切割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,train_size=0.85)

# 使用均值归一化和方差归一化提高模型精度
lin = StandardScaler(with_mean=True,with_std=True).fit(x_train)
x_train_scaler = lin.transform(x_train)
x_test_scaler = lin.transform(x_test)

# 多项式升维度
poly = PolynomialFeatures(degree=2,include_bias=False)
x_train_scaler = poly.fit_transform(x_train_scaler)
x_test_scaler = poly.fit_transform(x_test_scaler)

# 训练模型
linear = LinearRegression()
linear.fit(x_train_scaler,y_train)
y_test_hat = linear.predict(x_test_scaler)
print("LinearRegression线性算法拟合数据start")
rmse_train = np.sqrt(mean_squared_error(y_true=y_train,y_pred=linear.predict(x_train_scaler)))
print(rmse_train)
rmse_test = np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_test_hat))
print(rmse_test)
print("LinearRegression线性算法拟合数据end")

print("Ridge岭回归拟合数据start")
ridge = Ridge(alpha=0.4)
ridge.fit(x_train_scaler,y_train)
y_train_ridge = ridge.predict(x_train_scaler)
y_test_ridge = ridge.predict(x_test_scaler)
rmse_train = np.sqrt(mean_squared_error(y_true=y_train,y_pred=y_train_ridge))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_test_ridge))
print(rmse_train)
print(rmse_test)
print("Ridge岭回归拟合数据end")

print("GradientBoostingRegressor拟合数据start")
gradient = GradientBoostingRegressor()
gradient.fit(x_train_scaler,y_train)
y_train_gradient = gradient.predict(x_train_scaler)
y_test_gradient = gradient.predict(x_test_scaler)
rmse_train = np.sqrt(mean_squared_error(y_true=y_train,y_pred=y_train_gradient))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_test_gradient))
print(rmse_train)
print(rmse_test)
print("GradientBoostingRegressor拟合数据end")

print("KNeighborsRegressor拟合数据start")
kn = KNeighborsRegressor()
kn.fit(x_train_scaler,y_train)
y_train_kn = kn.predict(x_train_scaler)
y_test_kn = kn.predict(x_test_scaler)
rmse_train = np.sqrt(mean_squared_error(y_true=y_train,y_pred=y_train_kn))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_test_kn))
print(rmse_train)
print(rmse_test)
print("KNeighborsRegressor拟合数据end")




