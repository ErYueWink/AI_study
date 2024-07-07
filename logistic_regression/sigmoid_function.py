import numpy as np
import math
import matplotlib.pyplot as plt

"""
    逻辑回归属于分类算法，那为什么逻辑回归不叫逻辑分类？因为逻辑回归算法是基于多元线性回归的算法。而正因为此，逻辑回归这个分类算法是线性的分类器。
"""
def sigmoid(x):
    a = []
    for item in x:
        a.append(1.0/(1.0+math.exp(-item)))
    return a

x = np.arange(-10,10,0.1)
y = sigmoid(x)

plt.plot(x,y,'b-')
plt.show()