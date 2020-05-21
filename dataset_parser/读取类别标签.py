import numpy as np
from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = 100000000000
# img = Image.open('/media/ding/学习/研一下学期/PSPNET-UNET-FCN_keras/ceshi/biaozhu/1.png')
# img = Image.open('/media/ding/学习/研一下学期/PSPNET-UNET-FCN_keras/img/unet_epoch_1.png')
img = Image.open('/media/ding/Study/tianchi/nongye_ai/data/train_1_label/img_512_43008_.png')
# img = cv2.imread('/media/ding/学习/研一下学期/Lake Segmentation/dataset/label/train/20161126.png')
# img = Image.open('/media/ding/学习/Segmentation/Datasets/CITYSCAPES/gtFine_trainvaltest/gtFine/val/frankfurt/frankfurt_000000_014480_gtFine_labelTrainIds.png')
img = np.array(img)

print(np.unique(img))

##对应label编号
# [ 0 38]

