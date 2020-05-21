import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

datagen = ImageDataGenerator(
           rotation_range=1,       # 一个度数（0-180）的值，是一个随机旋转图片的范围
           width_shift_range=0.2,  # 在垂直或水平方向上随机平移图片的范围（作为总宽度或高度的一部分）
           height_shift_range=0.2,
           shear_range=0.2,        # 随机应用剪切变换
           zoom_range=0.2,         # 随机缩放图片内部
           horizontal_flip=True,   # 水平地随机翻转一半图像 - 当没有水平不对称假设时（例如真实世界的图片）是相关的。
           fill_mode='constant')    # 填充新创建的像素的策略，可以在旋转或宽度/高度偏移后出现。

seed = 1024
Expansion = 4 # 扩张因子
transfer_path ='/media/ding/Study/First_next_term/Lake Segmentation/image/512-1/origin/origin_image/'
# transfer_label_path = '/media/ding/Study/First_next_term/Lake Segmentation/image/512-1/labeled/origin_label/'
for root, dirs, files in os.walk(transfer_path): # 获取文件名字
    pass

for name in files:
    # # ***************************对原图进行数据增强***************************#
    train_ori_path = transfer_path +  name
    train_img = load_img(train_ori_path)  # this is a PIL image
    x = img_to_array(train_img)  # this is a Numpy array with shape (3, 256, 256)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 256, 256)

    i = 0
    for batch in datagen.flow(x, batch_size=1, seed=seed,
                              save_to_dir='/media/ding/Study/First_next_term/Lake Segmentation/image/zengqiang/origin/',
                              save_prefix='x'+name.split(".")[0], save_format='png'):
        i = i + 1
        if i > Expansion - 1:
            break

    #***************************对标签进行数据增强***************************#
    # label_ori_path = transfer_path  + name
    # label_img = load_img(label_ori_path)  # this is a PIL image
    # x = img_to_array(label_img)           # this is a Numpy array with shape (3, 256, 256)
    # x = x.reshape((1,) + x.shape)         # this is a Numpy array with shape (1, 3, 256, 256)
    #
    # i = 0
    # for batch in datagen.flow(x, batch_size=1,  seed = seed,
    #                           save_to_dir='/media/ding/Study/First_next_term/Lake Segmentation/image/zengqiang/label/',
    #                           save_prefix=name.split(".")[0], save_format='png'):
    #     i = i+1
    #     if i > Expansion-1:
    #         break

