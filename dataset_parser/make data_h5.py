from __future__ import print_function

import h5py
import os
import numpy as np
import cv2
#
dir_path = '/media/ding/Study/First_next_term/Lake Segmentation'
origin_folder_path = '/media/ding/Study/First_next_term/Lake Segmentation/dataset/origin'
label_folder_path = '/media/ding/Study/First_next_term/Lake Segmentation/dataset/label'
# dir_path = '/media/ding/Study/tianchi/nongye_ai/data'
# origin_folder_path = '/media/ding/Study/tianchi/nongye_ai/data/origin'
# label_folder_path = '/media/ding/Study/tianchi/nongye_ai/data/label'
# Use only 2 classes.
# labels = ['_background_', 'lake']


# Reads paths from cityscape data.
def get_data(mode):
    if mode == 'train' or mode == 'val' or mode == 'test':
        x_paths = []
        y_paths = []
        tmp_origin_folder_path = os.path.join(origin_folder_path, mode)
        tmp_label_folder_path = os.path.join(label_folder_path, mode)

        # os.walk helps to find all files in directory.
        for (path, dirname, files) in os.walk(tmp_origin_folder_path):
            for filename in files:
                x_paths.append(os.path.join(path, filename))

        # Find ground_truth file paths with x_paths.
        idx = len(tmp_origin_folder_path) #取出源图片路径，与下面的标签文件对应
        for x_path in x_paths:
            y_paths.append(tmp_label_folder_path + x_path[idx:-4] + '.png')

        return x_paths, y_paths
    else:
        print("Please call get_data function with arg 'train', 'val', 'test'.")


# Make h5 group and write data
def write_data(h5py_file, mode, x_paths, y_paths):
    num_data = len(x_paths)       #图片文件的数量
    print(num_data)


    # h5py special data type for image.
    uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8'))

    # Make group and data set.
    group = h5py_file.create_group(mode)
    x_dset = group.create_dataset('x', shape=(num_data, ), dtype=uint8_dt)
    y_dset = group.create_dataset('y', shape=(num_data, ), dtype=uint8_dt)

    for i in range(num_data):
        # Read image and resize
        x_img = cv2.imread(x_paths[i])
        # cv2.imshow('image', x_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x_img = cv2.resize(x_img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)   #图片扩大  x 和 y 的缩放因子
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        # print(x_img)

        y_img = cv2.imread(y_paths[i])
        # print(y_paths[i])
        # cv2.imshow('image', y_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        y_img = cv2.resize(y_img, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)  #图片缩小,这时候是3通通道的
        y_img = y_img[:, :, 0]  #选取第一个通道了
        # print(y_img.shape)
        # cv2.imshow('image', y_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        x_dset[i] = x_img.flatten()
        y_dset[i] = y_img.flatten()
        # print(y_dset[0])


# Make h5 file.
def make_h5py():
    x_train_paths, y_train_paths = get_data('train')
    print(x_train_paths)
    print(y_train_paths)
    x_val_paths, y_val_paths = get_data('val')
    x_test_paths, y_test_paths = get_data('test')


    # Make h5py file with write option.
    h5py_file = h5py.File(os.path.join(dir_path, 'lake-yun.h5'), 'w')

    # Write data
    print('Parsing train datas...')
    write_data(h5py_file, 'train', x_train_paths, y_train_paths)
    print('Finish.')
    print('#######################################################')

    print('Parsing val datas...')
    write_data(h5py_file, 'val', x_val_paths, y_val_paths)
    print('Finish.')
    print('#######################################################')

    print('Parsing test datas...')
    write_data(h5py_file, 'test', x_test_paths, y_test_paths)
    print('#######################################################')
    print('Finish.')


make_h5py()
