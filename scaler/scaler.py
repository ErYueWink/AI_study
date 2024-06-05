from sklearn.preprocessing import StandardScaler
import numpy as np

# 归一化
temp = np.array([[1],[2],[3],[4],[5]])
temp.reshape(-1,1) # 针对于列操作
print(temp)
scaler = StandardScaler() # StandardScaler默认就是使用均值归一化
# 计算均值和标准差
scaler.fit(temp)
print(scaler.transform(temp)) # 均值为0 方差为1