import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 设置随机数种子
np.random.seed(42)
# 样本总量
m = 100
# 数据集 X，y
X = 6 * np.random.rand(m,1) - 3 # X (-3,3)
y = 0.5*X**2 + X + 2 + np.random.randn(m,1)

# 数据集X，y中前80条数据作为训练集 后20条数据作为测试集
x_train = X[:80]
x_test = X[80:]
y_train = y[:80]
y_test = y[80:]

# 将X，y的图画出来 可以看出来是非线性的
plt.plot(X,y,'b.')
# plt.show()

# 准备字典数据 进行多项式升维
dict = {1:'g-',2:'r+',10:'y*'}
# 遍历字典数据
for n in dict:
    # degree:几阶 include_bias:是否考虑截距项 考虑截距项会给X添加恒为1的一列
    poly_reg = PolynomialFeatures(degree=n,include_bias=True)
    # 进行多项式升维后X训练集、测试集的结果
    x_poly_train = poly_reg.fit_transform(x_train)
    x_poly_test = poly_reg.fit_transform(x_test)
    print(x_train[0])
    print(x_poly_train[0])
    print(x_test[0])
    print(x_poly_test[0])

    # 计算y_hat
    line_reg = LinearRegression(fit_intercept=False)
    line_reg.fit(x_poly_train,y_train)
    print(line_reg.intercept_,line_reg.coef_)

    y_train_predict = line_reg.predict(x_poly_train)
    y_test_predict = line_reg.predict(x_poly_test)
    plt.plot(x_poly_train[:,1],y_train_predict,dict[n])

    print(mean_squared_error(y_train,y_train_predict))
    print(mean_squared_error(y_test,y_test_predict))

plt.show()
