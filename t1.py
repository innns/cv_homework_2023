# >>>>>>
# Descripttion:
# version: 1.0
# Author: Zx
# Email: ureinsecure@outlook.com
# Date: 2023-04-24 08:52:55
# LastEditors: Zx
# LastEditTime: 2023-04-24 09:38:34
# FilePath: /homework_2023/t1.py
# <<<<<<
import cv2 as cv  # 引入 OpenCV 库


img = cv.imread('data/1.jpg')  # 使用 imread 函数读取图像，并以 numpy 数组形式储存
print(img.shape)  # 查看图像的大小。返回的元组（touple）中的三个数依次表示高度、宽度和通道数
print(img.dtype)  # 查看图片的类型
cv.imshow('img', img)  # 使用imshow函数显示图像，第一个参数是窗口名称（可不写），第二个参数是要显示的图像的名称，一定要写
cv.imwrite('origin.jpg', img)

# cv.waitKey(0) #可以让窗口一直显示图像直到按下任意按键
# 使用 cv.cvtColor 函数转换色彩空间，参数‘cv.COLOR_BGR2GRAY’表示从 RGB 空间转换到灰度空间
img_GRAY = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', img_GRAY)
cv.imwrite('gray.jpg', img_GRAY)

# 使用 cv.threshold函数进行图像阈值处理，参数‘cv.THRESH_BINARY’代表了阈值的类型，127 为阈值
ret, thresh = cv.threshold(img_GRAY, 127, 255, cv.THRESH_BINARY)
cv.imshow('threshold', thresh)
cv.imwrite('bin.jpg', thresh)

# cv.waitKey(0)
# 使用 cv.resize 函数进行图像缩放
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
cv.imshow('resize', res)
cv.waitKey(0)
cv.imwrite('result.jpg', res)  # 保存图像
cv.waitKey(0)
