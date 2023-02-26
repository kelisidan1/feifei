---
title: python数学建模
date: 2021-06-26 11:31:57
tags:
---



# 插值与拟合

- 拟合

- - 一般是对于离散点
    - 用函数代替列表函数使得误差在某种意义下最小



- 插值

- - 一般是对于离散点
    - 用一个函数来近似代替列表函数，并要求**函数通过列表函数中给定的数据点**



- 逼近

- - 一般是对于连续函数
    - 为复杂函数寻找近似替代函数，其误差在某种度量下最小



**作用：**构造一个简单函数作为要考察数据或复杂函数的近似。反应对象整体的变化态势。

> 插值

python的解决方法：

+ interp1d函数
+ interp2d函数
+ interpn,interpnd多维函数



调用格式：interp1d(x,y,kind=‘linear’)

> 拟合

python的解决方法：

+ ployfit函数





> 区别

**数字多时用拟合**

**数字少时用插值**

## 插值

![image-20210907162106052](https://gitee.com/coderOasis/blog-img/raw/master/img/20210907162106.png)

code【demo01】:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
x = np.arange(0,25,2) # [ 0  2  4  6  8 10 12 14 16 18 20 22 24]
y = np.array([12,9,9,10,18,24,28,27,25,20,18,15,13])
xnew = np.linspace(0,24,500) #插值点 0~24平均分成500份
f1 = interp1d(x,y);y1 = f1(xnew)    # interp1d(x,y)返回了一个插值函数
f2 = interp1d(x,y,'cubic');y2=f2(xnew) # interp1d(x,y)返回了一个插值函数 cubic指的是3阶样条插值
plt.rc('font',size=16);plt.rc('font',family='SimHei')
plt.subplot(131),plt.plot(xnew,y1),plt.xlabel("(A)分段线性插值") # subplot(131) 1行3列，本图画在第一个位置上
plt.subplot(132),plt.plot(xnew,y2),plt.xlabel("(B)三次样条插值") # subplot(132) 1行2列，本图画在第二个位置上
plt.savefig("figure7_4.png",dpi=500);plt.show()
```

result:

![image-20210907170132598](https://gitee.com/coderOasis/blog-img/raw/master/img/20210907170132.png)





## 拟合



![image-20210908161908801](https://gitee.com/coderOasis/blog-img/raw/master/img/20210908161908.png)

这个

code【demo02】:

```python
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.pyplot import plot,show,rc
y = lambda x,a,b,c:a*x**2+b*x+c
x0 = np.arange(0, 1.1, 0.1)
y0= np.array([-0.447, 1.987, 3.28, 6.16, 7.08, 7.34, 7.66, 9.56, 9.48, 9.30, 11.2])
popt,pcov = curve_fit(y,x0,y0)  # popt是拟合的参数，pcov是参数的协方差矩阵
print("拟合的参数值为：",popt)
print("预测值分别为：",y(np.array([0.25,0.35]),*popt))
```

result:

```
拟合的参数值为： [-9.8045453  20.11972648 -0.02827248]
预测值分别为： [4.38887506 5.81257499]
```



上面的这个例子是通过二次函数来进行拟合的，同样的，我们可以使用其他的拟合函数来进行拟合。

像**z = ae<sup>bx</sup>+cy<sup>2</sup>**，也是可以通过一组数据来进行拟合操作的。







