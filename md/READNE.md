#### 三种梯度下降区别和优缺点

在讲三种梯度下降区别之前，我们先来总结一下梯度下降法的步骤：

1. 瞎蒙，Random随机θ，随机一组数值W0…Wn
2. 求梯度，为什么是梯度？因为梯度代表曲线某点上的切线的斜率，沿着切线往下下降就相当于沿着坡度最陡峭的方向下降
3. if g<0, theta往大调，if g>0, theta往小调
4. 判断是否收敛convergence，如果收敛跳出迭代，如果没有达到收敛，回第2步继续

四步骤对应计算方式：

a. np.random.rand()或者np.random.randn()

b. ![img](https://www.itbaizhan.com/wiki/imgs/wps634.png)

c. ![img](https://www.itbaizhan.com/wiki/imgs/wps635.png)

d. 判断收敛这里使用g=0其实并不合理，因为当损失函数是非凸函数的话g=0有可能是极大值对吗！所以其实我们判断loss的下降收益更合理，当随着迭代loss减小的幅度即收益不再变化就可以认为停止在最低点，收敛！

**区别：**其实三种梯度下降的区别仅在于第2步求梯度所用到的X数据集的样本数量不同！它们每次学习(更新模型参数)使用的样本个数，每次更新使用不同的样本会导致每次学习的准确性和学习时间不同。

![img](https://www.itbaizhan.com/wiki/imgs/wps636.jpg)