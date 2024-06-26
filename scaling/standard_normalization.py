from sklearn.preprocessing import StandardScaler
import numpy as np
# 标准归一化
scaler = StandardScaler()
data = np.array([1,2,3,4,50001])
# 针对于列操作
scaler.fit(data.reshape(-1,1))
print(scaler.mean_) # 均值
print(scaler.var_) # 方差(标准差的平方)
print(scaler.transform(data.reshape(-1,1))) # data标准归一化计算之后的结果
