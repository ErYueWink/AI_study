import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('./data/insurance.csv')
data.head(n=6)
print(data.head(n=6))

# EDA数据探索 向右偏
# plt.hist(data['charges'])
# plt.show()
plt.hist(np.log(data['charges']))
plt.show()
# 性别对保险花销预测的影响不大
# sns.kdeplot(data.loc[data.sex=='male','charges'],shade=True,label='male')
# sns.kdeplot(data.loc[data.sex=='female','charges'],shade=True,label='female')
# plt.show()

sns.kdeplot(data.loc[data.children==0,'charges'],shade=True,label='children1')
sns.kdeplot(data.loc[data.children==1,'charges'],shade=True,label='children1')
sns.kdeplot(data.loc[data.children==2,'charges'],shade=True,label='children1')
sns.kdeplot(data.loc[data.children==3,'charges'],shade=True,label='children1')
sns.kdeplot(data.loc[data.children==4,'charges'],shade=True,label='children1')
sns.kdeplot(data.loc[data.children==5,'charges'],shade=True,label='children1')
plt.show()
