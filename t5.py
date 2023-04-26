# >>>>>>
# Descripttion:
# version: 1.0
# Author: Zx
# Email: ureinsecure@outlook.com
# Date: 2023-04-24 09:51:00
# LastEditors: Zx
# LastEditTime: 2023-04-24 10:06:56
# FilePath: /homework_2023/t5.py
# <<<<<<
import cv2
import numpy as np

img = cv2.imread("data/5-carNumber/5.jpg")
# H：190~245, S：0.35~1, V：0.3~1；
low_car = np.array([85, 88, 76])
up_car = np.array([122, 255, 255])
cv2.imshow("img", img)
cv2.waitKey()
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
bin_rect = cv2.inRange(hsv_img, low_car, up_car)

cv2.imshow("bin", bin_rect)

cv2.waitKey()
cv2.waitKey()
