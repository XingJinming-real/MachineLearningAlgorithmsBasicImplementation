import time
import os
import numpy as np
import matplotlib.pyplot as plt

dataTrain = []


def getData():
    global dataTrain
    fTrain = open(os.getcwd() + "\\" + "abalone.txt")
    for perLine in fTrain:
        dataTrain.append(perLine.strip().split('\t'))
    dataTrain = np.array(dataTrain).astype(np.float)
    dataTrainX = dataTrain[:, :-1].astype(np.float)
    dataTrainY = dataTrain[:, -1].astype(np.float)
    return dataTrainX, dataTrainY


def linearRegression(dataTrainX, dataTrainY):
    dataTrainX = np.mat(dataTrainX)
    dataTrainY = np.mat(dataTrainY)
    xTx = dataTrainX.T * dataTrainX
    if np.linalg.det(xTx) == 0:
        print("不可逆")
        return None
    else:
        wHat = xTx.I * (dataTrainX.T * dataTrainY.T)
    return wHat


def test(dataTrainX, dataTrainY, wHat):
    wHat = np.mat(wHat)
    yHat = dataTrainX * wHat
    yHat = np.mat(yHat)
    errorSum = (np.power((yHat - dataTrainY), 2)).sum()
    return errorSum


def LocallyWeightedLinearRegressionPer(testPoint, dataTrainX, dataTrainY, sita):
    dataTrainX = np.mat(dataTrainX)
    dataTrainY = np.mat(dataTrainY)
    weightedMat = np.eye(len(dataTrainX))
    for i in range(len(dataTrainX)):
        diff = (testPoint - dataTrainX[i]) * (testPoint - dataTrainX[i]).T
        weightedMat[i, i] = np.exp(diff / (-2 * sita ** 2))
    xTx = dataTrainX.T * (weightedMat * dataTrainX)
    if np.linalg.det(xTx) == 0:
        print("不可逆")
        return None
    else:
        wHat = xTx.I * (dataTrainX.T * (weightedMat * dataTrainY.T))
    return wHat


def LocallyWeightedLinearRegression(dataTestX, dataTrainX, dataTrainY, sita):
    yHat = np.zeros((1, len(dataTestX))).ravel()
    for perTestId in range(len(dataTestX)):
        yHat[perTestId] = np.mat(dataTestX[perTestId]) * LocallyWeightedLinearRegressionPer(dataTestX[perTestId],
                                                                                            dataTrainX,
                                                                                            dataTrainY, sita)
    return yHat


def testForAbalone(dataTrainX, dataTrainY, sita):
    errorSum = 0
    testSample = list(range(100, 200))
    for perSampleId in testSample:
        try:
            wHat = LocallyWeightedLinearRegressionPer(dataTrainX[perSampleId], dataTrainX, dataTrainY, sita)
            yHat = np.mat(dataTrainX[perSampleId, :]) * np.mat(wHat)
            errorSum += np.power((yHat - dataTrainY[perSampleId]), 2)
        except:
            continue
    return errorSum


def ridgeRegression(dataTrainX, dataTrainY, lam=0.5):
    temp = np.eye(len(dataTrainX[0]))
    dataTrainX = np.mat(dataTrainX)
    dataTrainY = np.mat(dataTrainY)
    xTx = dataTrainX.T * dataTrainX + temp * lam
    if np.linalg.det(xTx) == 0:
        print("不可逆")
        return None
    else:
        wHat = xTx.I * (dataTrainX.T * dataTrainY.T)
    errorSum = np.power(dataTrainX * wHat - dataTrainY, 2).sum()
    return errorSum, wHat


def forwardGreedy(errorSum, dataTrainX, dataTrainY, epsilon, iterNum, wHat):
    bestError = np.inf
    bestW = wHat.copy()
    w = wHat.copy()
    for perIter in range(iterNum):
        for i in range(len(w)):
            w = bestW.copy()
            for operator in [-1, 1]:
                w[i] = w[i] + operator * epsilon
                curError = test(dataTrainX, dataTrainY, w)
                if curError < bestError:
                    bestError = curError
                    bestW = w.copy()
                w[i] = w[i] - operator * epsilon
    return bestW, bestError


def main():
    dataTrainX, dataTrainY = getData()
    wHat = linearRegression(dataTrainX, dataTrainY)
    yHat = LocallyWeightedLinearRegression(dataTrainX, dataTrainX, dataTrainY, 0.02)
    if wHat is not None:
        R2, yHat = test(dataTrainX, dataTrainY, wHat)
    else:
        R2 = yHat = None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sortedDataX = np.argsort(dataTrainX[:, 1])
    yHat = yHat[sortedDataX]
    xPlot = dataTrainX[sortedDataX]
    ax.plot(xPlot[:, 1], yHat)
    ax.scatter(dataTrainX[:, 1], dataTrainY, color='r')
    plt.show()
    print(testForAbalone(dataTrainX, dataTrainY, 1))
    bestW, bestError = forwardGreedy(dataTrainX, dataTrainY, .1, 3, ridgeRegression(dataTrainX, dataTrainY), wHat)
    print(bestError)
    dataTrainXMean = dataTrainX.mean(0)
    dataTrainXVar = dataTrainX.var(0)
    dataTrainX = (dataTrainX - dataTrainXMean) / np.sqrt(dataTrainXVar)
    dataTrainX = np.hstack((np.ones((len(dataTrainX), 1)), dataTrainX))
    errorSum, wHat = ridgeRegression(dataTrainX[100:199, :], dataTrainY[100:199])
    wHat = np.ones((len(dataTrainX[0]), 1)) * -0.5
    print('简单线性回归为{}'.format(test(dataTrainX[100:199, :], dataTrainY[100:199], linearRegression(dataTrainX[100:199, :],
                                                                                                dataTrainY[100:199]))))
    bestW, bestError = forwardGreedy(errorSum, dataTrainX[100:199, :], dataTrainY[100:199], 0.007, 2000, wHat)
    print('贪婪后：{}'.format(bestError))
    errorSum, wHat = ridgeRegression(dataTrainX[100:199, :], dataTrainY[100:199])
    print('岭回归:', test(dataTrainX[100:199, :], dataTrainY[100:199], wHat))
    errorSum = np.power(LocallyWeightedLinearRegression(dataTrainX[100:199, :], dataTrainX[100:199, :],
                                                        dataTrainY[100:199], 10) - dataTrainY[100:199], 2).sum()
    print("局部加权线性回归{}".format(errorSum))


main()
