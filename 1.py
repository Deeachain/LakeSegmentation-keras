import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

# 使用的图片 !wget https://pixabay.com/get/54e1d24b4f55a414f1dc8460825668204022dfe05554764e75297cd3/animals-2178578_640.jpg
# 待处理图像的路径, 自定义
img_path_orig = 'timg'
img_path = '{}.jpeg'.format(img_path_orig)
# 将图片数据处理成预训练模型可以处理的格式
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = VGG16(weights='/media/ding/Study/Model weights/VGG16_VGG19_and ResNet50/vgg16_weights_tf_dim_ordering_tf_kernels.h5')  # 加载VGG16模型, 用imagenet的权重
model.summary()
# 获取预测结果
preds = model.predict(x)
print('preds', preds.shape)
print(decode_predictions(preds, top=3)[0])  # 查看前三种预测类别的概率

# 利用 Grad-CAM算法 生成热力图. 然而看不懂为什么是这么实现的.
african_elephant_output = model.output[:, 386]

last_conv_layer = model.get_layer('block5_conv1')  # 最后一个卷积层

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0] # 返回的是一个列表
print(grads) #(?, 14, 14, 512)
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])
print('pooled_grads_value shape ', pooled_grads_value.shape)  #(512,)
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# 读取原图
img = cv2.imread(img_path)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# 转化为RGB
heatmap = np.uint8(255 * heatmap)

# 应用热力图到原图上
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img

cv2.imwrite('./{}_result.jpg'.format(img_path_orig), superimposed_img)
