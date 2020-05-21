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

def unet29(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
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
    x = GAU_1(block_4_out,x)

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
    x =GAU_2(block_3_out,x)

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

    x = GAU_3(block_2_out,x)
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
    x = GAU_4(block_1_out,x)

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

    # last conv
    # output = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)
    # output = squeeze_excite_block(output,16)


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
    model = unet29(input_shape=(512, 512, 3), num_classes=2,
                 lr_init=0.001, lr_decay=0.001,
                 vgg_weight_path='E:/Model weights/VGG16_VGG19_and ResNet50/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
                 # vgg_weight_path = '/media/ding/Study/Model weights/VGG16_VGG19_and ResNet50/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    # model.eval()
    model.summary()
    plot_model(model, to_file='unet29_model.png', show_shapes=True)