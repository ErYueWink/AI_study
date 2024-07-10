import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

list = datasets.load_iris()
print(list.keys())
print(list['DESCR'])
print(list['feature_names'])
X = list['data'][:,3:]
print(X)
print(list['target'])
# y有三个类别 对y进行二分类 list['target'] == 2为true时返回1 否则返回0
y = (list['target'] == 2).astype(int)
print(y)
binary_classifier = LogisticRegression(solver='sag',max_iter=1000) # 使用梯度下降
binary_classifier.fit(X,y)
# 生成测试集数据
X_test = np.linspace(0,3,1000).reshape(-1,1)
y_predict = binary_classifier.predict_proba(X_test) # 查看每条样本的预测值
print(y_predict)
y_hat = binary_classifier.predict(X_test)
print(y_hat) # 直接查看预测结果
