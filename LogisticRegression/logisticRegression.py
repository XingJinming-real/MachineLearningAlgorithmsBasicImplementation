import time
import os
import numpy as np
import random

dataTrain = []
dataTrainLabel = []
dataTest = []
dataTestLabel = []
m = 0
n = 0
iterNum = 200
alpha = 0.01


def getData(address):
    global dataTrainLabel, dataTrain, m, n, dataTest, dataTestLabel
    fTrain = open(os.getcwd() + "\\HorseColicTraining.txt")
    fTest = open(os.getcwd() + "\\HorseColicTest.txt")
    for perLine in fTrain:
        dataTrain.append(perLine.strip('\n').split('\t'))
    for perLine in fTest:
        dataTest.append(perLine.strip('\n').split('\t'))
    dataTrainLabel = np.mat(dataTrain)[:, -1].astype(np.float)
    dataTrain = np.mat(dataTrain)[:, :-1]
    dataTestLabel = np.mat(dataTest)[:, -1].astype(np.float)
    dataTest = np.mat(dataTest)[:, :-1].astype(np.float)
    m, n = np.shape(dataTrain)


def cleanData():
    pass


def sigmod(x):
    result = []
    for perX in x:
        result.append(1 if 1 / (1 + np.exp(-perX)) > 0.5 else 0)
    return np.mat(result).astype(np.float).T


def gradientAscendAll():
    global dataTrain, dataTrainLabel, m, n
    weights = np.mat(np.ones((n, 1)))
    dataTrain = dataTrain.astype(np.float)
    for perIter in range(iterNum):
        h = sigmod(np.dot(dataTrain, weights))
        error = dataTrainLabel - h
        weights = weights + alpha * dataTrain.transpose() * error
    return weights


def gradientAscendRandom():
    global alpha, dataTrain
    weights = np.ones((n, 1)).astype(np.float)
    dataTrain = dataTrain.astype(np.float)
    for perIter in range(iterNum):
        candidates = list(range(len(dataTrain)))
        i = 0
        while len(candidates) != 0:
            i += 1
            alpha = (4 / (perIter + i)) + 0.01
            randIndex = random.choice(candidates)
            candidates.remove(randIndex)
            h = sigmod([dataTrain[randIndex] * weights])
            error = (dataTrainLabel[randIndex] - h)
            weights = weights + alpha * dataTrain[randIndex].T * error
    return weights


def test(weights):
    result = sigmod(dataTest * weights)
    return np.count_nonzero(result - dataTestLabel) / len(dataTest)


def main():
    getData(None)
    cleanData()
    # beginTime = time.perf_counter()
    # weights = gradientAscendAll()
    # endTime = time.perf_counter()
    # print("梯度上升算法错误率为{},用时{}s".format(test(weights), endTime - beginTime))
    tempSum = 0
    beginTime = time.perf_counter()
    for i in range(10):
        weights = gradientAscendRandom()
        tempSum += test(weights)
    endTime = time.perf_counter()
    print("随机梯度上升算法平均错误率为{},用时{}s".format(tempSum / 10, endTime - beginTime))
    pass


if __name__ == "__main__":
    main()
