import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 线性算法训练模型
from sklearn.linear_model import Ridge
# 决策树 梯度升级 可以用于拟合非线性数据
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
# 读取数据
data = pd.read_csv('./data/insurance.csv')
print(data.head(n=6))

# EDA 数据探索
plt.hist(np.log(data['charges']))
plt.show()
# 对data数据中非值类型的数据做离散化处理
data = pd.get_dummies(data=data,dtype=int)
print(data.head(n=6))

# 数据中charges作为目标变量，即y，其余变量作为自变量X1...Xn，顺便用fillna()将空值补充为0
x = data.drop('charges',axis=1)
y = data['charges']
print(x)
print(y)

x.fillna(0,inplace=True)
y.fillna(0,inplace=True)
print(x.head(n=6))
print(y.head(n=6))


# 对X和y进行切割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,train_size=0.85)

# 使用标准归一化提高模型精度 均值归一化和方差归一化
scaler = StandardScaler(with_mean=True,with_std=True).fit(x_train)
x_train_scaler = scaler.transform(x_train)
x_test_scaler = scaler.transform(x_test)

# 多项式升维度
poly = PolynomialFeatures(degree=2,include_bias=False)
x_train_scaler = poly.fit_transform(x_train_scaler)
x_test_scaler = poly.fit_transform(x_test_scaler)

# 训练模型
lin = LinearRegression()
lin.fit(x_train_scaler,np.log1p(y_train))
y_test_predict = lin.predict(x_test_scaler)

# 计算rmse
print("LinearRegression线性算法求解Loss start")
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train),y_pred=lin.predict(x_train_scaler))) # y经过log变换后计算出的rmse
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test),y_pred=y_test_predict)) # y经过log变换后计算出的rmse
rmse_train = np.sqrt(mean_squared_error(y_true=y_train,y_pred=np.exp(lin.predict(x_train_scaler)))) # y没有经过log变换后的rmse
rmse_test = np.sqrt(mean_squared_error(y_true=y_test,y_pred=np.exp(y_test_predict))) # y没有经过log变换后的rmse
print(log_rmse_train)
print(log_rmse_test)
print(rmse_train)
print(rmse_test)
print("LinearRegression线性算法求解Loss end")

ridge = Ridge(alpha=0.4)
ridge.fit(x_train_scaler,np.log1p(y_train))
y_ridge_train_predict = ridge.predict(x_train_scaler)
y_ridge_test_predict = ridge.predict(x_test_scaler)
print("Ridge线性算法求解Loss start")
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train),y_pred=y_ridge_train_predict)) # y经过log变换后计算出的rmse
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test),y_pred=y_ridge_test_predict)) # y经过log变换后计算出的rmse
rmse_train = np.sqrt(mean_squared_error(y_true=y_train,y_pred=y_ridge_train_predict)) # y没有经过log变换后的rmse
rmse_test = np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_ridge_test_predict)) # y没有经过log变换后的rmse
print(log_rmse_train)
print(log_rmse_test)
print(rmse_train)
print(rmse_test)
print("Ridge线性算法求解Loss end")

grand = GradientBoostingRegressor()
grand.fit(x_train_scaler,np.log1p(y_train))
y_grand_train_predict = grand.predict(x_train_scaler)
y_grand_test_predict = grand.predict(x_test_scaler)
print("GradientBoostingRegressor决策树求解Loss start")
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train),y_pred=y_grand_train_predict)) # y经过log变换后计算出的rmse
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test),y_pred=y_grand_test_predict)) # y经过log变换后计算出的rmse
rmse_train = np.sqrt(mean_squared_error(y_true=y_train,y_pred=y_grand_train_predict)) # y没有经过log变换后的rmse
rmse_test = np.sqrt(mean_squared_error(y_true=y_test,y_pred=y_ridge_test_predict)) # y没有经过log变换后的rmse
print(log_rmse_train)
print(log_rmse_test)
print(rmse_train)
print(rmse_test)
print("GradientBoostingRegressor线性算法求解Loss end")