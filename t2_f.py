# >>>>>>
# Descripttion:
# version: 1.0
# Author: Zx
# Email: ureinsecure@outlook.com
# Date: 2023-04-24 09:29:37
# LastEditors: Zx
# LastEditTime: 2023-04-24 09:31:36
# FilePath: /homework_2023/t2_f.py
# <<<<<<
"""
频域平滑滤波器
(1) 理想低通滤波器
(2) Butterworth低通滤波器
(3) 高斯低通滤波器

频域锐化滤波器
(1) 理想高通滤波器
(2) Butterworth高通滤波器
(3) 高斯高通滤波器
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def filter(img, D0, N=2, type='lp', filter='butterworth'):
    '''
    频域滤波器
    Args:
        img: 灰度图片
        D0: 截止频率
        N: butterworth的阶数(默认使用二阶)
        type: lp-低通 hp-高通
        filter:butterworth、ideal、Gaussian即巴特沃斯、理想、高斯滤波器
    Returns:
        imgback：滤波后的图像
    '''
    # 离散傅里叶变换
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    # 中心化
    dtf_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    mask = np.zeros((rows, cols, 2))  # 生成rows行cols列的二维矩阵

    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)  # 计算D(u,v)
            if (filter.lower() == 'butterworth'):  # 巴特沃斯滤波器
                if (type == 'lp'):
                    mask[i, j] = 1 / (1 + (D / D0) ** (2 * N))
                elif (type == 'hp'):
                    mask[i, j] = 1 / (1 + (D0 / D) ** (2 * N))
                else:
                    assert ('type error')
            elif (filter.lower() == 'ideal'):  # 理想滤波器
                if (type == 'lp'):
                    if (D <= D0):
                        mask[i, j] = 1
                elif (type == 'hp'):
                    if (D > D0):
                        mask[i, j] = 1
                else:
                    assert ('type error')
            elif (filter.lower() == 'gaussian'):  # 高斯滤波器
                if (type == 'lp'):
                    mask[i, j] = np.exp(-(D * D) / (2 * D0 * D0))
                elif (type == 'hp'):
                    mask[i, j] = (1 - np.exp(-(D * D) / (2 * D0 * D0)))
                else:
                    assert ('type error')

    fshift = dtf_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 计算像素梯度的绝对值
    img_back = np.abs(img_back)
    # img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
    return img_back


img = cv.imread('data/2.jpg', 0)

# 低通滤波器
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Input')
# plt.xticks([]), plt.yticks([])
# img_back1 = filter(img, 30, type='lp', filter='ideal')  # 截止频率30
# plt.subplot(222), plt.imshow(img_back1, cmap='gray'), plt.title('Output_ideal')
# plt.xticks([]), plt.yticks([])
# img_back2 = filter(img, 30, type='lp', filter='butterworth')
# plt.subplot(223), plt.imshow(
#     img_back2, cmap='gray'), plt.title('Output_butterworth')
# plt.xticks([]), plt.yticks([])
# img_back3 = filter(img, 30, type='lp', filter='gaussian')
# plt.subplot(224), plt.imshow(
#     img_back3, cmap='gray'), plt.title('Output_gaussian')
# plt.xticks([]), plt.yticks([])
# plt.show()


# 高通滤波器
# img = cv.imread('D:/Study/digital image processing/tu.png', 0)
plt.subplot(221), plt.imshow(img, cmap='gray'), plt.title('Input')
plt.xticks([]), plt.yticks([])
img_back1 = filter(img, 30, type='hp', filter='ideal') # 截止频率30
plt.subplot(222), plt.imshow(img_back1, cmap='gray'), plt.title('Output_ideal')
plt.xticks([]), plt.yticks([])
img_back2 = filter(img, 30, type='hp', filter='butterworth') 
plt.subplot(223), plt.imshow(img_back2, cmap='gray'), plt.title('Output_butterworth')
plt.xticks([]), plt.yticks([])
img_back3 = filter(img, 30, type='hp', filter='gaussian') 
plt.subplot(224), plt.imshow(img_back3, cmap='gray'), plt.title('Output_gaussian')
plt.xticks([]), plt.yticks([])
plt.show()
