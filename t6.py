'''
Descripttion: 
version: 1.0
Author: Zx
Email: ureinsecure@outlook.com
Date: 2023-05-19 15:07:09
LastEditors: Zx
LastEditTime: 2023-05-19 15:27:42
FilePath: /cv_homework_2023/t6.py
'''
# >>>>>>
# Descripttion:
# version: 1.0
# Author: Zx
# Email: ureinsecure@outlook.com
# Date: 2022-06-08 19:57:31
# LastEditors: Zx
# LastEditTime: 2022-06-08 22:45:28
# FilePath: /cxl/main.py
# <<<<<<
import cv2
import numpy as np
import math

PATH = "data"  # 文件路径
THRES = 220  # 二值化阈值

# HSV 区间参考值 https://blog.csdn.net/leo_888/article/details/88284251
# 红色HSV区间
RED_HSV_LOW = np.array([0, 150, 100])
RED_HSV_UP = np.array([15, 255, 255])
RED_HSV_LOW_ = np.array([220, 150, 100])
RED_HSV_UP_ = np.array([255, 255, 255])

PINK_HSV_LOW = np.array([0, 100, 200])
PINK_HSV_UP = np.array([10, 150, 255])

# 芒果、香蕉HSV区间
YELLOW_HSV_LOW = np.array([20, 160, 43])
YELLOW_HSV_UP = np.array([40, 255, 255])

# 梨HSV区间
PEAR_HSV_LOW = np.array([20, 50, 120])
PEAR_HSV_UP = np.array([40, 160, 255])

GREEN_HSV_LOW = np.array([30, 20, 100])
GREEN_HSV_UP = np.array([90, 255, 255])


def process(temp_img):  # 图像处理
    # >>>>>>>>>>> 预处理 >>>>>>>>>>>
    # 去噪
    gauss_img = cv2.GaussianBlur(temp_img, (5, 5), 0)  # 高斯滤波
    cv2.imwrite("t6/gauss_img.jpg", gauss_img)  # 保存高斯滤波图
    # 灰度化
    gray_img = cv2.cvtColor(gauss_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("t6/gray_img.jpg", gray_img)  # 保存灰度图
    # 二值化
    # https://blog.csdn.net/m0_38068229/article/details/111769009
    bin_img = cv2.threshold(gray_img, THRES, 255, cv2.THRESH_BINARY_INV)[
        1]  # cv2.THRESH_BINARY_INV 取反向，便于处理
    cv2.imwrite("t6/bin_img.jpg", bin_img)  # 保存二值化图

    # >>>>>>>>>>> 区域分割 >>>>>>>>>>>
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dc_img = cv2.drawContours(temp_img, contours, -1, (255, 255, 0), 3)
    cv2.imwrite("t6/contours_img.jpg", dc_img)  # 保存边缘提取图

    # >>>>>>>>>>> 图像特征提取、特征参数的计算 >>>>>>>>>>>
    # 颜色特征 HSV
    hsv_img = cv2.cvtColor(gauss_img, cv2.COLOR_BGR2HSV)

    red_img = cv2.inRange(hsv_img, RED_HSV_LOW, RED_HSV_UP)
    red_img_ = cv2.inRange(hsv_img, RED_HSV_LOW_, RED_HSV_UP_)
    red_img = red_img + red_img_
    red_img = cv2.morphologyEx(
        red_img, cv2.MORPH_ERODE, np.ones((5, 5)))  # 腐蚀
    red_img = cv2.morphologyEx(
        red_img, cv2.MORPH_DILATE, np.ones((15, 15)))  # 膨胀
    # cv2.imwrite("t6/red_img.jpg", red_img)  # 保存红色图

    pink_img = cv2.inRange(hsv_img, PINK_HSV_LOW, PINK_HSV_UP)
    pink_img = cv2.morphologyEx(
        pink_img, cv2.MORPH_ERODE, np.ones((5, 5)))  # 腐蚀
    pink_img = cv2.morphologyEx(
        pink_img, cv2.MORPH_DILATE, np.ones((15, 15)))  # 膨胀
    # cv2.imwrite("t6/pink_img.jpg", pink_img)  # 保存粉红色图

    yellow_img = cv2.inRange(hsv_img, YELLOW_HSV_LOW, YELLOW_HSV_UP)
    # 形态学处理
    yellow_img = cv2.morphologyEx(
        yellow_img, cv2.MORPH_ERODE, np.ones((5, 5)))  # 腐蚀
    yellow_img = cv2.morphologyEx(
        yellow_img, cv2.MORPH_DILATE, np.ones((9, 9)))  # 膨胀
    # cv2.imwrite("t6/yellow_img.jpg", yellow_img)  # 保存黄色图

    pear_img = cv2.inRange(hsv_img, PEAR_HSV_LOW, PEAR_HSV_UP)
    # 形态学处理
    pear_img = cv2.morphologyEx(
        pear_img, cv2.MORPH_ERODE, np.ones((5, 5)))  # 腐蚀
    pear_img = cv2.morphologyEx(
        pear_img, cv2.MORPH_DILATE, np.ones((17, 17)))  # 膨胀
    # cv2.imwrite("t6/pear_img.jpg", pear_img)  # 保存梨图

    green_img = cv2.inRange(hsv_img, GREEN_HSV_LOW, GREEN_HSV_UP)
    # 形态学处理
    green_img = cv2.morphologyEx(
        green_img, cv2.MORPH_ERODE, np.ones((3, 3)))  # 腐蚀
    green_img = cv2.morphologyEx(
        green_img, cv2.MORPH_DILATE, np.ones((17, 17)))  # 膨胀

    # cv2.imwrite("t6/green_img.jpg", green_img)  # 保存黄色图

    # 划分ROI用
    mask = np.zeros(temp_img.shape[:2], np.uint8)  # 原图大小 mask图像
    black = np.zeros(temp_img.shape[:2], np.uint8)  # 原图大小 黑图像

    # 输出图像
    output_img = temp_img.copy()

    # 主循环
    for c in contours:
        # 求面积
        tmp_area = cv2.contourArea(c)
        # 求周长
        tmp_length = cv2.arcLength(c, True)
        # 求弧度
        tmp_rad = (4 * math.pi * tmp_area / tmp_length**2)
        # 求几何矩并计算区域中心（在图片上写字的时候用）
        tmp_M = cv2.moments(c)
        tmp_cen_X = int(tmp_M["m10"] / tmp_M["m00"])
        tmp_cen_Y = int(tmp_M["m01"] / tmp_M["m00"])
        mask = black.copy()  # 重置mask

        while (tmp_area > 30):  # 忽略过小的区域
            print("\nthis fruit:\narea:", tmp_area,
                  "\nlength", tmp_length, "\nrad:", tmp_rad)
            cv2.drawContours(
                mask, [c], -1, (255, 255, 255), cv2.FILLED)  # 将轮廓区域填充成白色
            # imshow("mask", mask)
            # 判断颜色：
            ROI_RED = cv2.bitwise_and(red_img, mask)
            if (len(ROI_RED[ROI_RED == 255]) / tmp_area > 0.4):
                # print("RED")
                # # 苹果桃子草莓
                if (tmp_rad < 0.75):
                    print("THIS IS STRAWBERRY")
                    cv2.putText(output_img, "STRAWBERRY", (tmp_cen_X - 80, tmp_cen_Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 100, 10), 3)
                    break
                else:
                    if (tmp_area > 52000):
                        print("THIS IS APPLE")
                        cv2.putText(output_img, "APPLE", (tmp_cen_X, tmp_cen_Y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 100, 10), 3)
                        break
                    else:
                        print("THIS IS PEACH")
                        cv2.putText(output_img, "PEACH", (tmp_cen_X, tmp_cen_Y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 100, 10), 3)
                        break

            ROI_PINK = cv2.bitwise_and(pink_img, mask)
            if (len(ROI_PINK[ROI_PINK == 255]) / tmp_area > 0.5):
                # print("PINK")
                # 荔枝
                print("THIS IS LICHEE")
                cv2.putText(output_img, "LICHEE", (tmp_cen_X, tmp_cen_Y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 100, 10), 3)
                break

            ROI_YELLOW = cv2.bitwise_and(yellow_img, mask)
            if (len(ROI_YELLOW[ROI_YELLOW == 255]) / tmp_area > 0.5):
                # print("YELLOW")
                # 芒果、香蕉
                if tmp_rad < 0.5:  # 根据弧度判断
                    print("THIS IS BANANA")
                    cv2.putText(output_img, "BANANA", (tmp_cen_X, tmp_cen_Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 100, 10), 3)
                else:
                    print("THIS IS MANGO")
                    cv2.putText(output_img, "MANGO", (tmp_cen_X, tmp_cen_Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 100, 10), 3)
                break

            ROI_PEAR = cv2.bitwise_and(pear_img, mask)
            if (len(ROI_PEAR[ROI_PEAR == 255]) / tmp_area > 0.5):
                print("THIS IS PEAR")
                # 梨子
                cv2.putText(output_img, "PEAR", (tmp_cen_X, tmp_cen_Y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 100, 10), 3)
                break

            ROI_GREEN = cv2.bitwise_and(green_img, mask)
            if (len(ROI_GREEN[ROI_GREEN == 255]) / tmp_area > 0.5):
                print("THIS IS PINEAPLLE")
                cv2.putText(output_img, "PINEAPPLE", (tmp_cen_X, tmp_cen_Y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 100, 10), 3)
                # 菠萝
                break
            print("UNKNOWN")
            break
        # cv2.waitKey()
    return output_img


if __name__ == "__main__":
    img = cv2.imread("data/6_fruit.jpg")  # 读取图像
    # cv2.imshow("img", img)
    output_img = process(img)
    cv2.imshow("output_img", output_img)
    cv2.imwrite("t6/output.jpg", output_img)
    cv2.waitKey()
