import time
import os
import numpy as np
import random
import matplotlib.pyplot as plt

data = []


def getData():
    global data
    meanVector = []
    f = open(os.getcwd() + "\\" + "secom.txt")
    for perLine in f:
        data.append(perLine.strip().split(' '))
    for i in range(len(data[0])):
        tempSum = 0
        NaNPosVector = []
        loop = 0
        tempMean = 0
        for j in range(len(data)):
            if data[j][i] == 'NaN':
                NaNPosVector.append((j, i))
                continue
            else:
                tempSum += float(data[j][i])
                loop += 1
        if loop != 0:
            tempMean = tempSum / loop
        for perTuple in NaNPosVector:
            data[perTuple[0]][perTuple[1]] = tempMean
    random.shuffle(data)
    data = np.mat(data).astype(np.float)
    data = data - np.mean(data, 0)


def PCA():
    getData()
    covMat = np.cov(data, rowvar=False)
    feaValue, feaVector = np.linalg.eig(covMat)
    print(feaVector)
    order = np.argsort(feaValue)
    order = order[::-1][:2]
    redFeaVector = np.array(feaVector[:, order]).reshape((len(feaVector), 2))
    newData = data * redFeaVector
    plt.scatter(newData[:, 0].tolist(), newData[:, 1].tolist())
    plt.show()
    pass


if __name__ == '__main__':
    PCA()
"""
PCA原理简介：
设data为m*n
经过对协方差矩阵的求解得出feaValue, feaVector
以feaValue对feaVector从大到小排序
选择前面的K个，注意redData=data*feaVector【:,0:k】
然后降维后的数据为m*k

优点
1）仅仅需要以方差衡量信息量，不受数据集以外的因素影响。
2）各主成分之间正交，可消除原始数据成分间的相互影响的因素。
3）计算方法简单，主要运算是特征值分解，易于实现


缺点
1）主成分各个特征维度的含义具有一定的模糊性，不如原始样本特征的解释性强。
2）方差小的非主成分也可能含有对样本差异的重要信息，因降维丢弃可能对后续数据处理有影响。
"""
