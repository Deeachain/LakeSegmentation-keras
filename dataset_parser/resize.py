import cv2
import os
# img = cv2.imread('/media/ding/学习/研一下学期/Lake Segmentation/dataset/origin/test/20140602.tif')
# img_resize = cv2.resize(img,(512,512),interpolation=cv2.INTER_NEAREST)
# img = cv2.imwrite('/media/ding/学习/研一下学期/湖泊分割论文项目/Fifth/img/test-512.tif',img_resize)
transfer_path = '/media/ding/学习/研一下学期/Lake Segmentation/image/1024/origin/w'
# transfer_path = '/media/ding/学习/研一下学期/Lake Segmentation/image/1024/labeled'
for (path, dirname, files) in os.walk(transfer_path):
    paths = []
    for filename in files:
        paths.append(os.path.join(path, filename))
    for path in paths:
        print(path)
        print(path[:45]+'512-1'+path[49:])
        origin_path = path
        new_path = path[:45]+'512-1'+path[49:]
        img = cv2.imread(path)
        img_resize = cv2.resize(img, (512,512), interpolation=cv2.INTER_NEAREST)
        img = cv2.imwrite(new_path, img_resize)