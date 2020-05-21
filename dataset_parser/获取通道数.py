# from skimage import io
# import numpy as np
# # path_c = r"/media/ding/学习/青海湖/20131008_json/label.png"
# path_c = r"/media/ding/学习/研一下学期/Lake Segmentation/label/train/20171222.png"
# color = io.imread(path_c)
# # # np.set_printoptions(threshold='nan')
# print(color, color.shape)
#
import cv2

# img = cv2.imread('/media/ding/学习/研一下学期/PSPNET-UNET-FCN_keras/img/unet_epoch_1.png')
# img = cv2.imread('/media/ding/学习/研一下学期/Lake Segmentation/dataset/label/train/x5_0_4869.png')
img = cv2.imread('1.jpg')
print(img.shape)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



