import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
# 决策树 梯度提升 可以用于拟合非线性的数据
from sklearn.ensemble import GradientBoostingRegressor
# 读取数据
data = pd.read_csv('./data/insurance.csv',sep=",")
data.head(n=6)
print(data.head(n=6))

#  EDA数据探索
# 绘图分析y的变化规律看看是否发生左偏或右偏
# plt.hist(data['charges']) # 发生右偏 对其矫正
plt.hist(np.log(data['charges']),bins=20)
plt.show()

# 特征工程 get_dummies()可以将数据非数值类型的值做离散化
"""
    data:要做离散化的数据 dtype:离散化之后的值类型 默认bool
"""
data = pd.get_dummies(data=data,dtype=int)
print(data.head())

# 获取表中的charges花销一列作为目标变量y，其余列作为自变量X1...Xn，顺便使用fillna()将空值部分自动填充为0
    # 移除数据表中的charges 剩下数据作为自变量X1...Xn axis:针对于列操作
x = data.drop('charges',axis=1)
y = data['charges']

x.fillna(0,inplace=True)
y.fillna(0,inplace=True)

print(x.tail())
print(y.head())

# 将数据集X，y切分为训练集和测试集
"""
    :arg:
        :test_size 测试集占数据集的30%
        :train_size 训练集占数据集的70%
"""
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8)

# 使用标准归一化提高模型准确率 均值归一化和方差归一化 fit()进行拟合
scaler = StandardScaler(with_mean=True,with_std=True).fit(x_train)
x_train_scaler = scaler.transform(x_train) # 训练集归一化后的数据
x_test_scaler = scaler.transform(x_test) # 测试集归一化后的数据
print(x_train_scaler)

# 使用多元线性回归对其进行多项式升维判断是否为线性
poly_features = PolynomialFeatures(degree=2,include_bias=False) # 升为二阶并且不考虑截距项
x_train_scaler = poly_features.fit_transform(x_train_scaler)
x_test_scaler = poly_features.fit_transform(x_test_scaler)

# 模型训练
lin_reg = LinearRegression()
# 进行拟合 计算出截距项
lin_reg.fit(x_train_scaler,np.log1p(y_train))
y_predict = lin_reg.predict(x_test_scaler)

# 训练集的rmse y经过log变换的
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train),y_pred=lin_reg.predict(x_train_scaler)))
print(log_rmse_train)
# 测试集的rmse
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test),y_pred=y_predict))
print(log_rmse_test)

ridge = Ridge(alpha=5.0)
ridge.fit(x_train_scaler,np.log1p(y_train))
y_predict = ridge.predict(x_test_scaler)

log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train),y_pred=ridge.predict(x_train_scaler)))
print(log_rmse_train)
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test),y_pred=y_predict))
print(log_rmse_test)

# 使用决策树 可以拟合非线性数据
"""
    对于非线性数据我们使用线性算法进行拟合训练出的模型并不是很好
"""
boost = GradientBoostingRegressor()
# 计算截距项和截距
boost.fit(x_train_scaler,np.log1p(y_train))
# 预测
y_boost_predict = boost.predict(x_test_scaler)
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train),y_pred=boost.predict(x_train_scaler)))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test),y_pred=y_boost_predict))
rmse_train = np.sqrt(mean_squared_error(y_true=y_train,y_pred=np.exp(boost.predict(x_train_scaler))))
rmse_test = np.sqrt(mean_squared_error(y_true=y_test,y_pred=np.exp(y_boost_predict)))
print(log_rmse_train)
print(log_rmse_test)
print(rmse_train)
print(rmse_test)


