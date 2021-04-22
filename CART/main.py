import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

dataTrain = []
dataTrainX = []
dataTrainY = []


def getData():
    global dataTrain, dataTrainX, dataTrainY
    fTrain = open(os.getcwd() + "\\" + "bikeSpeedVsIq_train.txt")
    for perLine in fTrain:
        dataTrain.append(perLine.strip().split('\t'))
    dataTrain = np.array(dataTrain).astype(np.float)
    dataTrainX = dataTrain[:, :-1].astype(np.float)
    dataTrainY = dataTrain[:, -1].astype(np.float)


def split(curDataTrain, perFeatureId, perValue):
    lSet = curDataTrain[np.nonzero(curDataTrain[:, perFeatureId] <= perValue)[0], :]
    rSet = curDataTrain[np.nonzero(curDataTrain[:, perFeatureId] > perValue)[0], :]
    return lSet, rSet


def getError(perSet, featureId):
    return np.var(perSet[:, featureId]) * len(perSet)


def findTheBestFeature(curDateTrain, remainFeature):
    minError = np.inf
    bestFeatureId = None
    splitValue = 0
    for perFeatureId in remainFeature:
        for perValue in curDateTrain[:, perFeatureId]:
            lSet, rSet = split(curDateTrain, perFeatureId, perValue)
            curErrorL = getError(lSet, perFeatureId)
            curErrorR = getError(rSet, perFeatureId)
            if curErrorR + curErrorL < minError:
                minError = curErrorR + curErrorL
                bestFeatureId = perFeatureId
                splitValue = perValue
    return bestFeatureId, splitValue, minError


def createTree(curDataTrain, remainFeature):
    root = {'remainFeature': remainFeature.copy(), 'curDataTrain': curDataTrain}
    if len(remainFeature) == 0:
        return None
    bestFeatureSplit, splitValue, minError = findTheBestFeature(curDataTrain, remainFeature)
    root['splitValue'] = splitValue
    root['splitFeatureId'] = bestFeatureSplit
    lSet, rSet = split(curDataTrain, bestFeatureSplit, splitValue)
    remainFeature.remove(bestFeatureSplit)
    root['left'] = createTree(lSet, remainFeature.copy())
    root['right'] = createTree(rSet, remainFeature.copy())
    return root


def prone(root):
    if type(root['left']) == dict and type(root['right'] == dict):
        prone(root['left'])
        prone(root['right'])
    if type(root['left']) != dict and type(root['right'] != dict):
        lSet, rSet = split(root['curDataTrain'], root['splitFeatureId'], root['splitValue'])
        splitError = getError(lSet, root['splitFeatureId']) + getError(rSet, root['splitFeatureId'])
        mergeError = getError(root['curDataTrain'],root['splitFeatureId'])



def main():
    getData()
    remainFeatureId = list(range(len(dataTrain[0])))
    root = createTree(dataTrain, remainFeatureId)
    pass


if __name__ == "__main__":
    main()
