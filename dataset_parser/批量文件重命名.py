import os
import cv2 as cv
for (path, dirname, files) in os.walk('/media/ding/学习/研一下学期/Lake Segmentation/image/512-1/labeled/origin_label/'):
    x_paths = []
    for filename in files:
        x_paths.append(os.path.join(path, filename))
    for path in x_paths:
        print(path)
        print(path[:80]+'.png')
        a = cv.imread(path)
        save = cv.imwrite(path[:80]+'.png',a)