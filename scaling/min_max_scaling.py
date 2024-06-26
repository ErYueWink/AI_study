from sklearn.preprocessing import MinMaxScaler
import numpy as np

from scaler.scaler import temp

# 最大值最小值归一化
scaler = MinMaxScaler()
arr = np.array([1,2,3,5,5])
# 使用最大值最小值归一化计算 获得最大值最小值归一化后的结果
a = scaler.fit_transform(temp.reshape(-1,1))
"""
[[0.  ]
 [0.25]
 [0.5 ]
 [0.75]
 [1.  ]]
 最大值最小值归一化的优点是一定会把数值归到0-1之间
 缺点是，如果有一个离群值，会使得一个数值为1，其它的数值都几乎为0，受离群值影响比较大
"""
print(a)