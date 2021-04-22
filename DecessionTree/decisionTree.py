import numpy as np
import os
import time

data = []
trainNum = 0


def getData():
    global data, trainNum
    f = open('lenses.txt')
    for perLine in f:
        data.append((perLine.strip('\n')).split('\t'))
        trainNum += 1


def getOneSetCnt(oneListId):
    tempMap = {}
    tempSum = 0
    numOfSample = len(oneListId)
    for perId in oneListId:
        if tempMap.get(data[perId][-1]) is None:
            tempMap[data[perId][-1]] = 1
        else:
            tempMap[data[perId][-1]] += 1
    for perNum in tempMap.values():
        tempSum += (perNum / numOfSample) * np.log(numOfSample / perNum)
    return tempSum


def getInformationGain(attrToList):
    tempDiv = len(attrToList.keys())
    tempSum = 0
    for perKey in attrToList.keys():
        tempSum += getOneSetCnt(attrToList[perKey])
    return tempSum / tempDiv


class Node:
    def __init__(self):
        self.currentAllSamples = []
        self.parentNode = None
        self.sons = []
        self.chosenAttrId = []
        self.remainAttrId = []
        self.currentCnt = 999

    def getAttrValue(self, perAttrId):
        global data
        sub_attr = set()
        dataCopy = np.array(data)
        attrColumn = (dataCopy[:, perAttrId]).ravel()
        attrToList = {}
        for perData in attrColumn:
            sub_attr.add(perData)
        sub_attr = list(sub_attr)
        for perSub_attr in sub_attr:
            attrToList[perSub_attr] = []
        for perSample in self.currentAllSamples:
            attrToList[data[perSample][perAttrId]].append(perSample)
        value = getInformationGain(attrToList)
        return self.currentCnt - value, attrToList

    def findOneBestAttr(self):
        maxGain = 0
        maxSeparation = {}
        bestAttr = 1
        for perAttrId in self.remainAttrId:
            curValue, currentSeparation = self.getAttrValue(perAttrId)
            if curValue > maxGain:
                maxSeparation = currentSeparation
                maxGain = curValue
                bestAttr = perAttrId
        return bestAttr, maxSeparation


def createTree(curNode):
    curNode.currentCnt = getOneSetCnt(curNode.currentAllSamples)
    if curNode.currentCnt == 0:
        return
    bestAttrId, maxSeparation = curNode.findOneBestAttr()
    for perKind in maxSeparation.values():
        tempNode = Node()
        tempNode.chosenAttrId.append(bestAttrId)
        tempNode.currentAllSamples = perKind
        tempNode.parentNode = curNode
        tempNode.remainAttrId = curNode.remainAttrId.copy()
        tempNode.remainAttrId.remove(bestAttrId)
        curNode.sons.append(tempNode)
    for perSon in curNode.sons:
        createTree(perSon)


def predict(sample):
    pass


def main():
    getData()
    root = Node()
    root.currentAllSamples = [i for i in range(trainNum)]
    root.currentCnt = getOneSetCnt(root.currentAllSamples)
    root.remainAttrId = list(range(len(data[0]) - 1))
    createTree(root)


main()
