import os
import time
import numpy as np
import matplotlib.pyplot as plt

dataTrainAttr = []
dataTrainLabel = []


def loadFile():
    os.chdir('C:\\Users\\admin\\Desktop\\trainingDigits')
    listDir = os.listdir()
    for perFile in listDir:
        f = open(perFile, 'r')
        tempList = []
        for i in f:
            perLine = i.strip()
            for perWord in perLine:
                tempList.append(int(perWord))
        dataTrainAttr.append(tempList)
        dataTrainLabel.append(perFile[0])


def img2vec(fileName):
    f = open(fileName)
    tempList = []
    for i in f:
        i = i.strip()
        tempList.extend(i)
    tempList = list(map(lambda x: eval(x), tempList))
    return tempList


def kNN(vec, k):
    dataTrainAttrMat = np.array(dataTrainAttr)
    testVec = np.array(vec)
    mayNum = []
    dist = np.sqrt(abs(dataTrainAttrMat ** 2 - testVec ** 2))
    dist = np.sum(dist, 1)
    candidateNum = dist.argsort()
    for i in range(k):
        mayNum.append(eval(dataTrainLabel[candidateNum[i]]))
    mayNum = np.array(mayNum)
    return np.argmax(np.bincount(mayNum))


def predict(k):
    os.chdir('C:\\Users\\admin\\Desktop\\testDigitsNew')
    listDir = os.listdir()
    totalNum = len(listDir)
    errorNum = 0
    for perFileName in listDir:
        returnVec = img2vec(perFileName)
        predictNum = kNN(returnVec, k)
        if predictNum != int(perFileName[0]):
            errorNum += 1
    print("当k=={}时，正确率为{}".format(k, 1 - errorNum / totalNum))
    return 1 - errorNum / totalNum


def main():
    loadFile()
    accuracy = []
    for k in range(1, 10):
        accuracy.append(predict(k))
    plt.plot(range(1, 10), accuracy)
    plt.show()


main()
