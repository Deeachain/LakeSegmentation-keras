# coding:utf-8
import os

path = '/media/ding/Study/First_next_term/Lake Segmentation/data/多云和冰/origin/'  # path是你存放json的路径
json_file = os.listdir(path)

for file in json_file:
    os.system("python E:\Anocado\Anocado3\envs\labelme\Scripts\labelme_json_to_dataset.py %s" % (path + file))  # 使用自己的labelme路径
