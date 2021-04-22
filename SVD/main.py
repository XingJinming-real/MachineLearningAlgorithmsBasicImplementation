"""暂时程序错误"""


import numpy as np
import cv2
import os


def main():
    img = cv2.imread("D:\\1.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('afs', img)
    cv2.waitKey(0)
    u, s, v = np.linalg.svd(img)
    s = np.diag(s)
    newImg = (u[:, :50].dot(s[:50, :50])).dot(v[:50, :])
    cv2.imshow('fff', newImg)
    cv2.waitKey(0)


main()
