from __future__ import print_function
from keras.callbacks import Callback

import cv2
import numpy as np
import os
import pandas as pd
from keras import backend as K

class TrainCheck(Callback):
    def __init__(self, output_path, model_name):
        self.epoch = 0
        self.output_path = output_path
        self.model_name = model_name

    def result_map_to_img(self, res_map):
        img = np.zeros((512, 512, 1), dtype=np.uint8)
        res_map = np.squeeze(res_map)
        # print(res_map.shape)  #(256, 256, 2)两个类别：湖泊和背景
        argmax_idx = np.argmax(res_map, axis=2)
        # For np.where calculation.
        # print(argmax_idx.shape)      #256*256
        lake = (argmax_idx == 1)
        # _background_ = np.logical_not(lake)
        # img[:, :, 0] = np.where(_background_,0, 0)
        img[:, :, 0] = np.where(lake, 255, 0)
        return img

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch + 1
        # self.visualize('img/test-512.tif')
        self.visualize('img/2.png')



    def visualize(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        img = img / 127.5 - 1

        pred = self.model.predict(img)
        # print('pred shape is ', pred.shape) #pred shape is  (1, 512, 512, 2)
        res_img = self.result_map_to_img(pred[0])
        cv2.imwrite(os.path.join(self.output_path,  self.model_name + str(self.epoch) + '.png'), res_img)
        '''heatmap 1'''
        predict=pred[0,:,:,1]
        pmin=np.min(predict)
        pmax=np.max(predict)
        predict=((predict-pmin)/(pmax-pmin+0.000001))*225
        predict=predict.astype(np.uint8)
        predict=cv2.applyColorMap(predict,cv2.COLORMAP_JET)
        img = cv2.imread(path)
        predict = predict * 0.6 + img
        cv2.imwrite(os.path.join(self.output_path,  self.model_name  + str(self.epoch) + 'heatmap1.png'), predict)

        # '''heatmap2'''
        # from keras.preprocessing import image
        # from keras.applications.vgg16 import preprocess_input
        # img = image.load_img(path, target_size=(512, 512))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        #
        # # get category loss
        # class_output = self.model.output[:, :, :, 1]
        # # layer output
        # convolution_output = self.model.get_layer('conv2d_9')
        # # get gradients
        # grads = K.gradients(class_output, convolution_output.output)[0]
        # # print(grads)  # (?, 32, 32, 512)
        # pooled_grads = K.mean(grads, axis=-1)
        # # print(grads)  # (?, 32, 32, 512)
        # iterate = K.function([self.model.input], [pooled_grads, convolution_output.output[0]])
        #
        # pooled_grads_value, conv_layer_output_value = iterate([x])
        # # print('pooled_grads_value shape ', pooled_grads_value.shape)
        # for i in range(1):
        #     conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        #
        # heatmap = np.mean(conv_layer_output_value, axis=-1)
        # # print('heat map shape', heatmap.shape)
        # heatmap = np.maximum(heatmap, -1)
        # heatmap /= np.max(heatmap)
        # # print('heat map shape', heatmap.shape)
        # # create heat map
        # img = cv2.imread(path)
        # heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
        # # print('heat map shape', heatmap.shape)
        # # 转化为RGB
        # heatmap = np.uint8(255 * heatmap)
        #
        # # 应用热力图到原图上
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #
        # superimposed_img = heatmap * 0.4 + img
        # cv2.imwrite(os.path.join(self.output_path, self.model_name + str(self.epoch) + 'heatmap2.png'), superimposed_img)







