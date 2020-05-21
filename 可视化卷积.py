from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation,Add,GlobalAveragePooling2D ,Reshape,multiply,Dense,Permute
from keras.layers import BatchNormalization
from keras.optimizers import Adam,SGD
from keras import backend as K
from keras.utils.vis_utils import plot_model

import numpy as np
import tensorflow as tf


def visual(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    img_input = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    output = Activation('relu')(x)
    model = Model(img_input, output)
    # model.summary()
    model.compile(optimizer=SGD(lr=lr_init, decay=lr_decay, momentum=0.9))
    return model

