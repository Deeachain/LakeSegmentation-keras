import os
import cv2 as cv
import numpy as np

#全局阈值
def threshold_demo(path,image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    # print("threshold value %s"%ret)
    # cv.namedWindow("binary0", cv.WINDOW_NORMAL)
    # cv.imshow("binary0", binary)
    print(path[:-4]+'.png')
    cv.imwrite(path[:-4]+'.png',binary)

# src = cv.imread('/media/ding/学习/研一下学期/Lake Segmentation/dataset/label/train/20161126-1.png')
# cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
# cv.imshow('input_image', src)

transfer_path = '/media/ding/Study/First_next_term/Lake Segmentation/dataset/label/train1/'
for (path, dirname, files) in os.walk(transfer_path):
    x_paths = []
    for filename in files:
        x_paths.append(os.path.join(path, filename))
    for path in x_paths:
        print(path)
        src = cv.imread(path)
        #################################
        threshold_demo(path,src)
        ##############移除文件#############
        # if path[-6:] == '-1.png':
        #     os.remove(path)
        #################################

# cv.waitKey(0)
# cv.destroyAllWindows()
