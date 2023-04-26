# >>>>>>
# Descripttion:
# version: 1.0
# Author: Zx
# Email: ureinsecure@outlook.com
# Date: 2023-04-24 09:45:41
# LastEditors: Zx
# LastEditTime: 2023-04-24 09:50:38
# FilePath: /homework_2023/t4.py
# <<<<<<

import cv2

img = cv2.imread("data/3.jpg", 0)

blur = cv2.GaussianBlur(img, (7, 7), 1, 1)


# 调用 Laplacian 算法的 OpenCV 库函数进行图像轮廓提取
result = cv2.Laplacian(blur, cv2.CV_16S, ksize=1)
LOG = cv2.convertScaleAbs(result)  # 得到 LOG 算法处理结果

cv2.imwrite("log.jpg", LOG)
# Canny 算子进行边缘提取
Canny = cv2.Canny(blur, 50, 150)

cv2.imwrite("canny.jpg", Canny)


x = cv2.Scharr(blur, cv2.CV_16S, 1, 0)  # X 方向
y = cv2.Scharr(blur, cv2.CV_16S, 0, 1)  # Y 方向
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
cv2.imwrite("Scharr.jpg", Scharr)
