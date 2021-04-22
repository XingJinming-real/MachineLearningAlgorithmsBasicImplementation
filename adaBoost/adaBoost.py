import time
import os
import numpy as np
dataTrain = []
dataTest = []


def getData():
    global dataTest, dataTrain
    fTrain = open(os.getcwd() + "\\" + "horseColicTraining2.txt")
    for perLine in fTrain:
        dataTrain.append(perLine.strip().split('\t'))
    fTest = open(os.getcwd() + "\\" + "horseColicTest2.txt")
    for perLine in fTest:
        dataTest.append(perLine.strip().split('\t'))
    dataTrain = np.array(dataTrain)
    dataTest = np.array(dataTest)
    dataTrainAttr = dataTrain[:, :-1].astype(np.float)
    dataTrainLabel = dataTrain[:, -1].astype(np.float)
    dataTestAttr = dataTest[:, :-1].astype(np.float)
    dataTestLabel = dataTest[:, -1].astype(np.float)
    # dataTrainAttr = np.array([[1, 2], [2, 1.1], [1.3, 1], [1, 1], [2, 1]]).astype(np.float)
    # dataTrainLabel = np.array([1, 1, -1, -1, 1]).astype(np.float)
    # dataTestAttr = None
    # dataTestLabel = None
    return dataTrainAttr, dataTestAttr, dataTrainLabel, dataTestLabel


def getResult(dataTrainAttr, attrId, perValue, operator):
    result = np.ones((1, len(dataTrainAttr[0]))).ravel()
    if operator == 'lower':
        result[dataTrainAttr[attrId] <= perValue] = -1
    else:
        result[dataTrainAttr[attrId] > perValue] = -1
    return result


def getPerResult(dataTrainAttr, attrId, perValue, operator):
    if (operator == 'lower' and dataTrainAttr[attrId] <= perValue) or (
            operator == 'upper' and dataTrainAttr[attrId] > perValue):
        return -1
    else:
        return 1


def getBestPerDS(dataTrainAttr, dataTrainLabel, D):
    m, n = dataTrainAttr.shape
    minError = 999
    bestDS = {}
    for perAttrId in range(n):
        perAttrMax = dataTrainAttr[:, perAttrId].max()
        perAttrMin = dataTrainAttr[:, perAttrId].min()
        interval = 10
        stepSize = (perAttrMax - perAttrMin) / interval
        for j in range(-1, int(interval) + 1):
            for operator in ['lower', 'upper']:
                perValue = perAttrMin + float(j) * stepSize
                result = getResult(dataTrainAttr.T, perAttrId, perValue, operator)
                error = np.ones((1, len(dataTrainAttr))).ravel()
                errorList = error.copy()
                error[result == dataTrainLabel] = 0
                errorList[result == dataTrainLabel] = -1
                weightedError = sum(np.multiply(D, error))
                if weightedError < minError:
                    minError = weightedError
                    bestDS['currentClass'] = np.array(result)
                    bestDS['errorValue'] = minError
                    bestDS['operator'] = operator
                    bestDS['perValue'] = perValue
                    bestDS['perAttrId'] = perAttrId
                    bestDS['errorList'] = np.array(errorList)
    return bestDS


def adaBoost(dataTrainAttr, dataTrainLabel, numIter=100):
    classifierSet = []
    D = np.ones((1, len(dataTrainAttr))).ravel() / len(dataTrainAttr)
    aggEst = np.zeros((1, len(dataTrainAttr))).ravel()
    for perIter in range(numIter):
        currentClassifier = getBestPerDS(dataTrainAttr, dataTrainLabel, D)
        classifierSet.append(currentClassifier)
        errorRation = currentClassifier['errorValue']
        alpha = float(0.5 * np.log((1 - errorRation) / errorRation))
        currentClassifier['alpha'] = alpha
        D = np.multiply(D, np.exp(alpha * currentClassifier['errorList']))
        D = D / D.sum()
        aggEst += alpha * currentClassifier['currentClass']
        if np.all(np.sign(aggEst) == dataTrainLabel):
            print(classifierSet)
            return classifierSet
        else:
            pass
            # print('当前正确率{}'.format(np.count_nonzero(np.sign(aggEst) == dataTrainLabel) / len(dataTrainAttr)))
    return classifierSet


def test(dataTestAttr, dataTestLabel, classifierSet):
    errorNum = 0
    for perTestId in range(len(dataTestAttr)):
        perSum = 0
        for perClassifier in classifierSet:
            perSum += perClassifier['alpha'] * getPerResult(dataTestAttr[perTestId], perClassifier['perAttrId'],
                                                            perClassifier['perValue'], perClassifier['operator'])
        if np.sign(perSum) != dataTestLabel[perTestId]:
            errorNum += 1
    return errorNum / len(dataTestAttr)


def main(numIter):
    dataTrainAttr, dataTestAttr, dataTrainLabel, dataTestLabel = getData()
    classifierSet = adaBoost(dataTrainAttr, dataTrainLabel, numIter)
    errorRation = test(dataTestAttr, dataTestLabel, classifierSet)
    print("测试集上错误率{}".format(errorRation))
    pass


beginTime = time.perf_counter()
main(numIter=50)
endTime = time.perf_counter()
print("用时{:*^20}s".format(endTime - beginTime))
