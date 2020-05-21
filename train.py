from __future__ import print_function
import time
import os
import matplotlib.pyplot as plt

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
 # 只显示 Error

from keras.backend.tensorflow_backend import set_session    #必须先import这个包，且必须在import keras之前
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard,CSVLogger
from callbacks import TrainCheck
from utils.metrics import Evaluator
from model.fcn import fcn_8s
from model.pspnet import pspnet50
# from model.deeplabv3plus import deeplabv3_plus
from dataset_parser.generator import data_generator


# Current python dir path
dir_path = os.path.dirname(os.path.realpath('__file__'))

start = time.time()
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
from model.unet import unet
from model.unet29 import unet29
from model.unet29_nobn import unet29_nobn
from model.unet_gpa import unet_gpa
from model.unet_attention import unet_attention
model_name = 'fcn'
TRAIN_BATCH = 3
VAL_BATCH = 3
lr_init = 0.0001
lr_decay = 0.0001
# vgg_path = '/media/ding/Study/Model weights/VGG16_VGG19_and ResNet50/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# vgg_path = None
vgg_path = 'E:/Model weights/VGG16_VGG19_and ResNet50/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# Use only 1 classes.
labels = ['lake','_background_']

# Choose model to train
if model_name == "fcn":
    model = fcn_8s(input_shape=(512, 512, 3), num_classes=len(labels),
                   lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet":
    model = unet(input_shape=(512, 512, 3), num_classes=len(labels),
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet29":
    model = unet29(input_shape=(512, 512, 3), num_classes=len(labels),
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet29_nobn":
    model = unet29(input_shape=(512, 512, 3), num_classes=len(labels),
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet_gpa":
    model = unet_gpa(input_shape=(512, 512, 3), num_classes=len(labels),
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)
elif model_name == "unet_attention":
    model = unet_attention(input_shape=(512, 512, 3), num_classes=len(labels),
                 lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)


filepath ='log/' +model_name + '_model_weight.h5'
if os.path.exists(filepath):
    model.load_weights(filepath)
    #若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")

# Define callbacks

checkpoint = ModelCheckpoint(filepath='./log/'+model_name + '_model_weight.h5',
                             monitor='val_mean_iou',
                             save_best_only=True,
                             save_weights_only=True,
                             period = 1)
train_check = TrainCheck(output_path='./img', model_name=model_name)
early_stopping = EarlyStopping(monitor='mean_iou', patience=800)
csv_logger = CSVLogger('./log/'+ model_name + 'logs.csv')

# training
# history = model.fit_generator(data_generator('/media/ding/Study/First_next_term/Lake Segmentation/lake-yun.h5', TRAIN_BATCH, 'train'),
history = model.fit_generator(data_generator('E:/First_next_term/Lake Segmentation/lake-yun.h5', TRAIN_BATCH, 'train'),
                              steps_per_epoch= 2401// TRAIN_BATCH,
                              # validation_data=data_generator('/media/ding/Study/First_next_term/Lake Segmentation/lake-yun.h5', VAL_BATCH, 'val'),
                              validation_data=data_generator('E:/First_next_term/Lake Segmentation/lake-yun.h5', VAL_BATCH, 'val'),
                              validation_steps=160 // VAL_BATCH,
                              callbacks=[checkpoint, train_check,TensorBoard(log_dir='./log'),csv_logger],
                              epochs= 100,
                              verbose=1)
end = time.time()
print('Run time is %.2f h' %((end-start)/3600))
plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="g", label="val")
plt.legend(loc="best")
plt.savefig('./log/'+model_name + '_loss_.png')

plt.gcf().clear()
plt.title("acc")
plt.plot(history.history["mean_iou"], color="b", label="miou")

plt.legend(loc="best")
plt.savefig('./log/'+model_name + '_acc_.png')
