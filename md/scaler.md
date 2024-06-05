#### 标准归一化

通常标准归一化中包含了均值归一化和方差归一化。经过处理的数据符合标准正态分布，即均值为0，标准差为1，其转化函数为：

![img](https://www.itbaizhan.com/wiki/imgs/wps687.png)

其中μ为所有样本数据的均值，σ为所有样本数据的标准差。

![img](https://www.itbaizhan.com/wiki/imgs/wps688.jpg)

相对于最大值最小值归一化来说，因为标准归一化是除以的是标准差，而标准差的计算会考虑到所有样本数据，所以受到离群值的影响会小一些，这就是除以方差的好处！但是如果是使用标准归一化不一定会把数据缩放到0到1之间了。

那为什么要减去均值呢？其实做均值归一化还有一个特殊的好处，我们来看下面梯度下降法的式子，我们会发现α是正数，不管A是正还是负，比如多元线性回归的话A就是![img](https://www.itbaizhan.com/wiki/imgs/wps689.png)，对于所有的维度X，比如这里的![img](https://www.itbaizhan.com/wiki/imgs/wps690.png)和![img](https://www.itbaizhan.com/wiki/imgs/wps691.png)来说α乘上A都是一样的符号，那么每次迭代的时候![img](https://www.itbaizhan.com/wiki/imgs/wps692.png)和![img](https://www.itbaizhan.com/wiki/imgs/wps693.png)的更新幅度符号也必然是一样的，这样就会像下图有右图所以，要想从![img](https://www.itbaizhan.com/wiki/imgs/wps694.png)更新到![img](https://www.itbaizhan.com/wiki/imgs/wps695.png)就必然要么W1和W2同时变大再同时变小，或者就W1和W2同时变小再同时变大。不能如图上所示蓝色的最优解路径，即W1变小的时候W2变大。

![image-20230706174218753](https://www.itbaizhan.com/wiki/imgs/image-20230706174218753.png)

那我们如何才能做到让W1变小的时候W2变大呢？归其根本还是大部分数据集X矩阵中的数据均为正数，同学们可以想想比如人的信息，年龄、收入等不都是正数值嘛，公式中![img](https://www.itbaizhan.com/wiki/imgs/wps698.png)和![img](https://www.itbaizhan.com/wiki/imgs/wps699.png)如果都是正数的话就会出问题，所以如果我们可以让![img](https://www.itbaizhan.com/wiki/imgs/wps700.png)和![img](https://www.itbaizhan.com/wiki/imgs/wps701.png)它们符号不同，比如有正有负，其实就可以在做梯度下降的时候有更多的可能性去让更新尽可能沿着最优解路径去做。