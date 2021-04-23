import matplotlib.pyplot as plt
from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
import numpy as np

def runplt(size=None):
    plt.figure(figsize=size)
    plt.title('Zinc standard curve')
    plt.xlabel('Zinc solution concentration / ppm')
    plt.ylabel('Absorbance')
    plt.axis([0, 6.0, 0, 1.0])
    plt.grid(True)
    return plt

X = [[1.0],[2.0],[3.0],[4.0],[5.0]]
y = [[0.1932],[0.3604],[0.4937],[0.6029],[0.6942]]

model = linear_model.LinearRegression()
model.fit(X, y)
print("回归得到直线截距为：")
print(model.intercept_)  #截距
print("一次项系数为：")
print(model.coef_)  #线性模型的系数
print("相关系数方为")
print(model.score(X,y))
model.fit(y, X)
a = model.predict([[0.5166]])
a = np.around(a, 4)
print("预测得镁溶液的含量为(/ppm)：")
print(a)
b = [0.5166]
# print("Magnesium content：{:.2f}".format(model.predict([[ava(0.2426,0.2457,0.2441)]])[0][0]))
plt = runplt()
plt.plot(X, y, 'k.')
plt.axhline(b)
plt.plot(a, b, 'om')
X2 = [[0], [2], [5], [6]]
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
plt.text(3.378, 0.45, (3.378,0.5166),ha='left', va='bottom', fontsize=10)
plt.text(3.7, 0.9, '$A=0.12445c+0.095530$',ha='left', va='bottom', fontsize=10)
plt.text(3.7, 0.85, '$R^{2}=0.9857692829508851$',ha='left', va='bottom', fontsize=10)
plt.savefig("Zn.pdf")
plt.show()
