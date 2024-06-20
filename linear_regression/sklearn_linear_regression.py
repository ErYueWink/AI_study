import numpy as np
from sklearn.linear_model import LinearRegression

X_a = 2 * np.random.rand(100,1)
X_b = 2 * np.random.rand(100,1)
# 这里不需要我们手动添加恒为1的一列，sklearn会帮助我们添加
X_new = np.c_[X_a,X_b]
y = 3 + 5*X_a + 6*X_b + np.random.randn(100,1)
reg = LinearRegression(fit_intercept=True)
"""
    :arg:
        fit_intercept=True/False
        True:考虑截距项 default
        False:不考虑截距项
            一般都是考虑截距项的不考虑截距项的话误差会比较大
    :return:
        LinearRegression
"""
# 底层通过解析解的形式帮我们求解出θ
reg.fit(X_new,y)
print(reg.intercept_,reg.coef_)
"""
    reg.intercept_:截距项 W0
    reg.coef_:其他参数 W1...Wn
"""
# 使用我们训练好的模型去做预测
X_model = np.array([[0,0],
                    [3,3]])
# 计算y_hat
y_hat = reg.predict(X_model)
print(y_hat)