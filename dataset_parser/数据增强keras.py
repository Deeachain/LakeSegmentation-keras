from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
seed = 1024

datagen = ImageDataGenerator(
    rotation_range=1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    cval = 0)
'''
rotation_range 是一个度数（0-180）的值，是一个随机旋转图片的范围
width_shift并且height_shift是在垂直或水平方向上随机平移图片的范围（作为总宽度或高度的一部分）
rescale是一个值，我们将在任何其他处理之前将数据相乘。我们的原始图像包含0-255中的RGB系数，但是这些值对于我们的模型来说太高了（给定典型的学习速率），所以我们将目标值设置在0和1之间，而不是用1/255进行缩放。因子。
shear_range用于随机应用剪切变换
zoom_range 用于随机缩放图片内部
horizontal_flip 用于水平地随机翻转一半图像 - 当没有水平不对称假设时（例如真实世界的图片）是相关的。
fill_mode 是用于填充新创建的像素的策略，可以在旋转或宽度/高度偏移后出现。
'''
# train_tmp_path = '/media/ding/Study/First_next_term/Lake Segmentation/data/多云和冰/origin/1.png'
# mask_tmp_path = '/media/ding/Study/First_next_term/Lake Segmentation/data/多云和冰/origin/1'
train_tmp_path = '/media/ding/Study/First_next_term/Lake Segmentation/data/多云和冰/label/22.png'
mask_tmp_path = '/media/ding/Study/First_next_term/Lake Segmentation/data/多云和冰/label/22'
img = load_img(train_tmp_path)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,  seed = seed  ,# save_to_dir 文件夹   prefix图片名字   format格式
                          save_to_dir=mask_tmp_path, save_prefix='yun2', save_format='png'):

    i += 1
    if i > 100:  # 如果不break会无限循环
        break  # otherwise the generator would loop indefinitely
