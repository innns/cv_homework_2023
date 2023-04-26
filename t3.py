# >>>>>>
# Descripttion:
# version: 1.0
# Author: Zx
# Email: ureinsecure@outlook.com
# Date: 2023-04-24 09:37:40
# LastEditors: Zx
# LastEditTime: 2023-04-24 09:44:29
# FilePath: /homework_2023/t3.py
# <<<<<<
import cv2

bin = cv2.imread('t1/bin.jpg')

er = cv2.erode(bin, (5, 5))
cv2.imshow("er", er)
cv2.imwrite("er.jpg", er)

di = cv2.dilate(bin, (5, 5))
cv2.imshow("di", di)
cv2.imwrite("di.jpg", di)

op = cv2.morphologyEx(bin, cv2.MORPH_OPEN, (5, 5))
cv2.imshow("op", op)
cv2.imwrite("op.jpg", op)


cl = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, (5, 5))
cv2.imshow("cl", cl)
cv2.imwrite("cl.jpg", cl)
