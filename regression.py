import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 导入模型相关的库
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import data_processing as dp

# sigmoid函数和初始化数据
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def init_data():
    data = pd.read_csv("data.csv")
    median = data['SalePrice'].median()
    classLabels = []
    for i in range(data.shape[0]):
        if data['SalePrice'][i] >= median:   # 高房价
            classLabels.append('高房价')
        else:
            classLabels.append('低房价')
    dataMatIn = {
        'GrLivArea': data['GrLivArea'],
        'YearBuilt': data['YearBuilt']
    }
    # dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)  #特征数据集，添加1是构造常数项x0
    return dataMatIn, classLabels

def grad_descent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  #(m,n)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    weights = np.ones((n, 1))  #初始化回归系数（n, 1)
    alpha = 0.001 #步长
    maxCycle = 500  #最大循环次数

    for i in range(maxCycle):
        h = sigmoid(dataMatrix * weights)  #sigmoid 函数
        weights = weights + alpha * dataMatrix.transpose() * (labelMat - h)  #梯度
    return weights

def plotBestFIt(weights):
    dataMatIn, classLabels = init_data()
    n = np.shape(dataMatIn)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if classLabels[i] == 1:
            xcord1.append(dataMatIn[i][1])
            ycord1.append(dataMatIn[i][2])
        else:
            xcord2.append(dataMatIn[i][1])
            ycord2.append(dataMatIn[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1,s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3, 3, 0.1)
    y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]  #matix
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':
    dataMatIn, classLabels = init_data()
    r = grad_descent(dataMatIn, classLabels)
    print(r)
    plotBestFIt(r)



