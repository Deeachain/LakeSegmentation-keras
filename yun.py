import cv2 as cv
from PIL import Image
import numpy as np
# I = Image.open('/media/ding/Study/First_next_term/Lake Segmentation/data/多云和冰/label/1.png')
# L = I.convert('L')
# img = np.asarray(L)
# print(np.unique(img))
# L.save('/media/ding/Study/First_next_term/Lake Segmentation/data/多云和冰/label/11.png')
def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    # print("threshold value %s"%ret)
    # cv.namedWindow("binary0", cv.WINDOW_NORMAL)
    # cv.imshow("binary0", binary)
    cv.imwrite('/media/ding/Study/First_next_term/Lake Segmentation/data/多云和冰/label/92.png',binary)
# image = cv.imread('/media/ding/Study/First_next_term/Lake Segmentation/data/多云和冰/label/9.png')
# threshold_demo(image)
I = Image.open('/media/ding/Study/First_next_term/Lake Segmentation/dataset/label/train1/yun7_0_3276.png')
img = np.asarray(I)
print(np.unique(img))
