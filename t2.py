# >>>>>>
# Descripttion:
# version: 1.0
# Author: Zx
# Email: ureinsecure@outlook.com
# Date: 2023-04-24 09:10:04
# LastEditors: Zx
# LastEditTime: 2023-04-24 09:36:34
# FilePath: /homework_2023/t2.py
# <<<<<<
# encoding:utf-8
import cv2
from matplotlib import pyplot as plt
# import numpy as np
# 计算直方图
img = cv2.imread('data/1.jpg', 0)


equ = cv2.equalizeHist(img)
# res = np.hstack((img, equ))
# stacking images side-by-side
cv2.imshow('img', equ)
# cv2.imwrite('equ.jpg', equ)

hist = cv2.calcHist([equ], [0], None, [256], [0, 255])
# 画出直方图
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("number of Pixels")
plt.plot(hist)
plt.xlim([0, 256])

blur = cv2.blur(img, (3, 3))
cv2.imshow('img', blur)
# cv2.imwrite('blur.jpg', blur)


x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)   # 转回uint8
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


cv2.imshow('img', dst)
cv2.imwrite('t2/dst.jpg', dst)


plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
