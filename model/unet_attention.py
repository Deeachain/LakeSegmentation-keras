from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation,Add,GlobalAveragePooling2D ,Reshape,multiply,Dense,Permute
from keras.layers import BatchNormalization
from keras.optimizers import Adam,SGD
from keras import backend as K
from keras.utils.vis_utils import plot_model

import numpy as np
import tensorflow as tf

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
##############
# def dice_coef(y_true, y_pred):
#     return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
###############
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)
###############
def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block

    Args:
        input: input tensor
        filters: number of output filters

    Returns: a keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
def FPN(x):
    ##global pooling branch
    x = Conv2D(128, (1, 1), padding='same', name='tongdao/4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x_gpb = GlobalAveragePooling2D()(x)
    # shape = (1,1,128)
    # x_gpb = Reshape(shape)(x_gpb)
    # x_gpb = UpSampling2D(size=(32, 32), data_format=None , interpolation='bilinear')(x_gpb)
    # x_gpb = BatchNormalization()(x_gpb)

    #master branch
    # x_master = Conv2D(128, (1, 1), padding='same', name='master_branch')(x)
    # x_master = BatchNormalization()(x_master)
    # x_master = Activation('relu')(x_master)

    #last branch
    x_1 = Conv2D(128, (7, 7), padding='same', strides=(1, 1), name='down1')(x)
    x_1 = BatchNormalization()(x_1)
    x_1 = Activation('relu')(x_1)
    x_11 = Conv2D(128, (7, 7), padding='same', strides=(1, 1), name='down11')(x_1)
    x_11 = BatchNormalization()(x_11)
    x_11 = Activation('relu')(x_11)

    x_2 = Conv2D(128, (5, 5), padding='same',strides=(1, 1),  name='down2')(x_1)
    x_2 = BatchNormalization()(x_2)
    x_2 = Activation('relu')(x_2)
    x_22 = Conv2D(128, (5, 5), padding='same',strides=(1, 1),  name='down22')(x_2)
    x_22 = BatchNormalization()(x_22)
    x_22 = Activation('relu')(x_22)

    x_3 = Conv2D(128, (3, 3), padding='same',strides=(1, 1),  name='down3')(x_2)
    x_3 = BatchNormalization()(x_3)
    x_3 = Activation('relu')(x_3)
    x_33 = Conv2D(128, (3, 3), padding='same',strides=(1, 1),  name='down33')(x_3)
    x_33 = BatchNormalization()(x_33)
    x_33 = Activation('relu')(x_33)

    x = Add()([x_11, x_22,x_33])
    # x1 = Conv2D(128, (3, 3), padding='same')(x1)
    # x1 = BatchNormalization()(x1)
    # x1 = Activation('relu')(x1)
    #
    # x_master = multiply([x1,x_master])
    # x = Add()([x_master, x_gpb])
    return x
def GAU_1(down4,down5):
    y = GlobalAveragePooling2D()(down5)         #高级特征
    shape = (1,1,512)
    y = Reshape(shape)(y)
    y = Conv2D(512,(1,1),padding='same',name='gau1')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)

    mul = multiply([down4,y])

    # last = Add()([mul, down5])
    # last = concatenate([mul,down5],axis=3)
    return  mul
def GAU_2(down3,gau1):
    y = GlobalAveragePooling2D()(gau1)         # 高级特征
    shape = (1,1,256)
    y = Reshape(shape)(y)
    y = Conv2D(256,(1,1),padding='same',name='gau2')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)

    mul = multiply([down3,y])

    # last = Add()([mul,gau1])
    # last = concatenate([mul,gau1],axis=3)
    return  mul
def GAU_3(down2,gau2):
    y = GlobalAveragePooling2D()(gau2)         # 高级特征
    shape = (1,1,128)
    y = Reshape(shape)(y)
    y = Conv2D(128,(1,1),padding='same',name='gau3')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)


    mul = multiply([down2,y])

    # last = Add()([mul, gau2])
    # last = concatenate([mul, gau2],axis=3)
    return  mul
def GAU_4(down1,gau3):
    y = GlobalAveragePooling2D()(gau3)         #高级特征
    shape = (1,1,64)
    y = Reshape(shape)(y)
    y = Conv2D(64,(1,1),padding='same',name='gau4')(y)
    y = BatchNormalization()(y)
    y = Activation('sigmoid')(y)


    mul = multiply([down1,y])

    # last = Add()([mul, gau3])
    # last = concatenate([mul, gau3],axis=3)
    return  mul

def unet_attention(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    img_input = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    block_1_out = Activation('relu')(x)

    x = MaxPooling2D()(block_1_out)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    block_2_out = Activation('relu')(x)

    x = MaxPooling2D()(block_2_out)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    block_3_out = Activation('relu')(x)

    x = MaxPooling2D()(block_3_out)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    block_4_out = Activation('relu')(x)

    x = MaxPooling2D()(block_4_out)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for_pretrained_weight = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, for_pretrained_weight)
        vgg16.load_weights(vgg_weight_path,by_name=True)

    # UP 1
    ######
    # x = FPN(x)
    ######
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = GAU_1(block_4_out,x)

    x = concatenate([x, block_4_out])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x =GAU_2(block_3_out,x)

    x = concatenate([x, block_3_out])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = GAU_3(block_2_out,x)
    x = concatenate([x, block_2_out])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = GAU_4(block_1_out,x)

    x = concatenate([x, block_1_out])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(2, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    ''''''
    # ATTENTION PART STARTS HERE
    attention_probs = Dense(2, activation='softmax', name='attention_vec')(x)
    output = multiply([x, attention_probs])
    # ATTENTION PART FINISHES HERE
    ''''''
    model = Model(img_input, output)
    # model.summary()
    model.compile(optimizer=SGD(lr=lr_init, decay=lr_decay, momentum=0.9),
    # model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  # loss='categorical_crossentropy',
                  loss='binary_crossentropy',
                  metrics=['acc',mean_iou])
    return model

'''binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率
categorical_accuracy:对多分类问题,计算再所有预测值上的平均正确率
sparse_categorical_accuracy:与categorical_accuracy相同,在对稀疏的目标值预测时有用
top_k_categorical_accracy: 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
sparse_top_k_categorical_accuracy：与top_k_categorical_accracy作用相同，但适用于稀疏情况'''

if __name__ == "__main__":
    model = unet_attention(input_shape=(512, 512, 3), num_classes=2,
                 lr_init=0.001, lr_decay=0.001,
                 vgg_weight_path='E:/Model weights/VGG16_VGG19_and ResNet50/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
                 # vgg_weight_path = '/media/ding/Study/Model weights/VGG16_VGG19_and ResNet50/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # model.eval()
    model.summary()
    plot_model(model, to_file='unet_attentionmodel.png', show_shapes=True)