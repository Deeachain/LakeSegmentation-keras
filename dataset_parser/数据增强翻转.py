import os
import PIL.Image as img
import numpy as np
import matplotlib.pyplot as plt

def get_files(path='/media/ding/学习/研一下学期/Lake Segmentation/image/512-1/labeled/origin_label/'):
    image = []
    for root,dirs,fs in os.walk(path):   # os.walk获取所有的目录
        for f in fs:
            filename = os.path.join(root,f)
            print(filename)         #打印出所有文件（包括子目录）
            print(filename[:59]+'top'+filename[-13:-4])
            print('#########')
            im = img.open(filename)
            ng = im.transpose(img.FLIP_TOP_BOTTOM) #上下对换
            ng.save(filename[:59]+ 'top' + filename[-13:-4] + 'top.tif')
        for f in fs:
            filename = os.path.join(root, f)
            print(filename)  # 打印出所有文件（包括子目录）
            print('#########')
            im = img.open(filename)
            ng = im.transpose(img.FLIP_LEFT_RIGHT)  # 左右对换
            ng.save(filename[:59]+ 'left' + filename[-13:-4] + 'left.tif')
        for f in fs:
            filename = os.path.join(root, f)
            print(filename)  # 打印出所有文件（包括子目录）
            print('#########')
            im = img.open(filename)
            ng = im.transpose(img.ROTATE_90)  # 旋转 90 度角
            ng.save(filename[:59]+ '90' + filename[-13:-4] + '-90.tif')
        for f in fs:
            filename = os.path.join(root, f)
            print(filename)  # 打印出所有文件（包括子目录）
            print('#########')
            im = img.open(filename)
            ng = im.transpose(img.ROTATE_180)  # 旋转 180 度角
            ng.save(filename[:59]+ '180' + filename[-13:-4] + '-180.tif')
        for f in fs:
            filename = os.path.join(root, f)
            print(filename)  # 打印出所有文件（包括子目录）
            print('#########')
            im = img.open(filename)
            ng = im.transpose(img.ROTATE_270)  # 旋转 270 度角
            ng.save(filename[:59]+ '270' + filename[-13:-4]  + '-270.tif')

            if filename.endswith('.tif'):  # 判断是否是"rule"结尾
                image.append(filename)
        print(len(image))
    return image
get_files()
