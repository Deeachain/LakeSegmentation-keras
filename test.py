from __future__ import print_function

import cv2
import numpy as np

from model.unet import unet
from model.unet_fpa15 import unet_fpa
from model.unet_att16 import unet_att
from model.unet_se17 import unet_se
from model.unet_se_gau19 import unet_se_gau
from model.unet_se_gau_fpa21 import unet_se_gau_fpa
from model.unet_atts18 import unet_atts
from model.unet_gau5 import unet_gau
from model.unet_fpagau11 import unet_fpagau
from model.fcn import fcn_8s
from model.pspnet import pspnet50

def result_map_to_img(res_map):
    img = np.zeros((512, 512, 1), dtype=np.uint8)
    res_map = np.squeeze(res_map)
    argmax_idx = np.argmax(res_map, axis=2)
    # For np.where calculation.
    lake = (argmax_idx == 1)
    img[:, :, 0] = np.where(lake, 255, 0)
    return img

model_name = 'fcn'
img_path = './img/test-512.tif'

# Use only 2 classes.
labels = ['lake','_background_']


# Choose model to train
if model_name == "fcn":
    model = fcn_8s(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet":
    model = unet(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet_fpa":
    model = unet_fpa(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet_gau":
    model = unet_gau(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet_fpagau":
    model = unet_fpagau(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet_att":
    model = unet_att(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet_se":
    model = unet_se(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet_se_gau":
    model = unet_se_gau(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet_se_gau_fpa":
    model = unet_se_gau_fpa(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "unet_atts":
    model = unet_atts(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
elif model_name == "pspnet":
    model = pspnet50(input_shape=(512, 512, 3), num_classes=len(labels), lr_init=1e-3, lr_decay=5e-4)
try:
    model.load_weights('/media/ding/Study/First_next_term/lake segmentation project/Fifth/img/experiments/11/0.0001/unet_fpagau_model_weight.h5')
except:
    print("You must train model and get weight before test.")

x_img = cv2.imread(img_path)
cv2.imshow('x_img', x_img)
x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
x_img = x_img / 127.5 - 1
x_img = np.expand_dims(x_img, 0)

pred = model.predict(x_img)
res = result_map_to_img(pred[0])
cv2.imwrite('./img/test.png',res)
# cv2.imshow('res', res)
# cv2.waitKey(0)
