import matplotlib.pyplot as plt
from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
import numpy as np

def ava(x,y,z):
    b=sum([x,y,z])
    a=b/3
    A=[a]
    C=np.array(A)
    return C

def runplt(size=None):
    plt.figure(figsize=size)
    plt.title('Magnesium standard curve')
    plt.xlabel('Mg Addition / ppm')
    plt.ylabel('Absorbance')
    plt.axis([0, 0.5, 0, 0.5])
    plt.grid(True)
    return plt

# plt = runplt()
X = [[0.1],[0.2],[0.3],[0.4]]
y = [ava(0.0660,0.0642,0.0626), ava(0.1249,0.1235,0.1256), ava(0.1926,0.1918,0.1918), ava(0.2449,0.2387,0.2307)]
# plt.plot(X, y, 'k.')
# plt.show()

model = linear_model.LinearRegression()
model.fit(X, y)
print("回归得到直线截距为：")
print(model.intercept_)  #截距
print("一次项系数为：")
print(model.coef_)  #线性模型的系数
print("相关系数为")
print(model.score(X,y))
model.fit(y, X)
a = model.predict([ava(0.2426,0.2457,0.2441)])
a = np.around(a, 4)
print("预测得镁溶液的含量为(/ppm)：")
print(a)
b = ava(0.2426,0.2457,0.2441)
# print("Magnesium content：{:.2f}".format(model.predict([[ava(0.2426,0.2457,0.2441)]])[0][0]))
plt = runplt()
plt.plot(X, y, 'k.')
plt.axhline(b)
plt.plot(a, b, 'om')
X2 = [[0], [0.2], [0.35], [0.5]]
model = linear_model.LinearRegression()
model.fit(X,y)
y2 = model.predict(X2)
plt.plot(X2, y2, 'g-')
X = np.array(X).squeeze()
y = np.array(y).squeeze()
X = np.around(X, 4)
y = np.around(y, 4)
for X, y in zip(X, y):
    plt.text(X, y, (X,y),ha='right', va='bottom', fontsize=10)
plt.text(0.4009, 0.2441, (0.4009,0.2441),ha='left', va='bottom', fontsize=10)
plt.savefig("predict.pdf")
plt.show()
