import numpy as np
import cv2
import os

# 给标签图上色

def color_annotation(label_path, output_path):
    '''
    给class图上色
    '''
    img = cv2.imread(label_path,cv2.CAP_MODE_GRAY)

    color = np.ones([img.shape[0], img.shape[1], 3])
    print(img.shape[0],img.shape[1])
    print(color.shape)
    # color[img==0] = [0, 0, 0]    #road，白色，0
    # color[img==1] = [0, 1, 0]     #person，绿色，1
    # color[img==2] = [0, 0, 2]     #car，蓝色，2
    # color[img==255] = [255, 255, 255]     #background，黄色，3


    # color[img==0] = [128, 64, 128]    #road，紫色，0
    # color[img==1] = [60, 20, 220]   #person，红色，1
    # color[img==2] = [142, 0, 0]     #car，蓝色，2
    # color[img==255] = [0, 0, 0]     #background，黑色，3
    color[img == 0] = [255, 255, 255]  #
    color[img == 255] = [0, 0, 0]  #
################
#(B,G,R)通道顺序是反的
################
    cv2.imwrite(output_path,color)
transfer_path = '/media/ding/学习/研一下学期/Lake Segmentation/image/xuyiming/membrane/train/label'
for (path, dirname, files) in os.walk(transfer_path):
    x_paths = []
    for filename in files:
        x_paths.append(os.path.join(path, filename))
    for path in x_paths:
        print(path)
        color_annotation(path,path)