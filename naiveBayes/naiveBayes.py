import re
import os
import time
import numpy as np
import random

hamData = []
spamData = []
hamDataP = []
spamDataP = []
hamTestId = []
spamTestId = []
pHam = 0
pSpam = 0
totalTrainNum = 0


def getAlineVector(perLine):
    if perLine == '\n':
        return None
    perLine = perLine.strip('\n')
    perLine = perLine.strip('"')
    perLine = perLine.split(' ')
    i = 0
    for perWord in perLine:
        for comma in [',', '.', '?', '!', ':', ';']:
            perWord = perWord.strip(comma).lower()
        perLine[i] = perWord
        i += 1
    return perLine


def getData():
    global hamData, spamData, hamDataP, spamDataP, totalTrainNum, pHam, pSpam,hamTestId,spamTestId
    hamDataP = np.ones((1, 2000)).ravel()
    spamDataP = np.ones((1, 2000)).ravel()
    hamDirList = os.listdir(os.getcwd() + "\\ham")
    spamDirList = os.listdir(os.getcwd() + "\\spam")
    hamTestId = list(set([random.choice(hamDirList) for i in range(5)]))
    spamTestId = list(set(random.choice(spamDirList) for i in range(5)))
    for per in hamTestId:
        hamDirList.remove(per)
    for per in spamTestId:
        spamDirList.remove(per)
    totalTrainNum = len(hamDirList) + len(spamDirList)
    pHam = len(hamDirList) / totalTrainNum
    pSpam = len(spamDirList) / totalTrainNum
    for List in ['ham', 'spam']:
        if List == 'ham':
            dirList = hamDirList
        else:
            dirList = spamDirList
        for per in dirList:
            f = open(os.getcwd() + "\\" + List + "\\" + per)
            for perLine in f:
                perLine = getAlineVector(perLine)
                if perLine is None:
                    continue
                for perWord in perLine:
                    if List == "ham" and len(perWord) > 2:
                        if perWord not in hamData:
                            hamData.append(perWord)
                        hamDataP[hamData.index(perWord)] += 1
                    elif List == 'spam' and len(perWord) > 2:
                        if perWord not in spamData:
                            spamData.append(perWord)
                        spamDataP[spamData.index(perWord)] += 1
    hamDataP /= len(hamData)
    hamDataP = hamDataP[:len(hamData)]
    spamDataP /= len(spamData)
    spamDataP = spamDataP[:len(hamData)]


def test():
    errNum = 0
    totalTestNum = len(hamTestId) + len(spamTestId)
    for perData, name in [(hamTestId, 'ham'), (spamTestId, 'spam')]:
        for perTest in perData:
            f = open(os.getcwd() + '\\' + name + '\\' + perTest)
            if name == 'ham':
                perTestVector = np.zeros((1,len(hamDataP))).ravel()
            else:
                perTestVector = np.zeros((1,len(spamDataP))).ravel()
            for perLine in f:
                perLine = getAlineVector(perLine)
                if perLine is None:
                    continue
                for perWord in perLine:
                    if name == 'ham':
                        try:
                            perTestVector[hamData.index(perWord)] += 1
                        except ValueError:
                            continue
                    else:
                        try:
                            perTestVector[spamData.index(perWord)] += 1
                        except ValueError:
                            continue
            pTestHam = perTestVector * hamDataP
            pTestSpam = perTestVector * spamDataP
            for i in range(len(pTestHam)):
                if pTestHam[i] == 0:
                    pTestHam[i] = 1
            for i in range(len(pTestSpam)):
                if pTestSpam[i] == 0:
                    pTestSpam[i] = 1
            pTestHam = np.log(pTestHam)
            pTestSpam = np.log(pTestSpam)
            if name == 'ham' and sum(pTestHam) + np.log(pHam) < sum(pTestSpam) + np.log(pSpam):
                errNum += 1
            if name == 'spam' and sum(pTestHam) + np.log(pHam) > sum(pTestSpam) + np.log(pSpam):
                errNum += 1
    print('错误率{}'.format(errNum / totalTestNum))


if __name__ == "__main__":
    getData()
    test()
