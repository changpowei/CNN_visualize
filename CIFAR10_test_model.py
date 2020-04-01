#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:20:58 2020

@author: po-wei
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
import numpy as np
import matplotlib.pyplot as plt


class Unpooling2D(Layer):
    def __init__(self, poolsize=(2, 2), ignore_border=True):
        super(Unpooling2D,self).__init__()
        #self.input = tf.Tensor()
        self.poolsize = poolsize
        self.ignore_border = ignore_border

    def get_output(self, feature_map):
        #X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = feature_map.repeat(s1, axis=1).repeat(s2, axis=2)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
                "poolsize":self.poolsize,
                "ignore_border":self.ignore_border}

def get_layers_info(model):
    for i in range(len(model.layers)):
        layer = model.layers[i]
        # check if the layer is not conv or pooling, then break.
        if layer.output.shape.__len__() != 4:
            break
        
        else:
            try:
                filters, biases = layer.get_weights()
            except ValueError:
                filters, biases = None, None
            
        # summarize output shape
        layers_weights.append(filters)
        layers_biases.append(biases)
        layers_number.append(i)
        layers_config.append(layer.get_config())
        layers_name.append(layer.name)
        layers_output_shape.append(layer.output.shape)

def get_data():
    
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

    # 輸出0-9轉換爲ont-hot形式
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    return (train_images, train_labels), (test_images, test_labels)

def normal(deconvlution_first_layer):
    deconvlution_first_layer = deconvlution_first_layer.reshape(tuple(feature_model.inputs[0].get_shape()[1:]))
    deconvlution_org_img = (deconvlution_first_layer - np.min(deconvlution_first_layer))/np.ptp(deconvlution_first_layer)
    return deconvlution_org_img


def get_feature_map(test_img, model):
    while True:
        try:
            feature_map_layer = int(input("Feature map from which layer(From %d to %d) : " % (0, len(layers_number) - 1)))
            if feature_map_layer < 0 or feature_map_layer > len(layers_number) - 1:
                raise 'Number is out of boundary!'
        except ValueError:
            print("Not an integer! Try again.")
            continue
        except:
            print('Number is out of boundary!')
        else:
            feature_model = Model(inputs = model.inputs, outputs = model.layers[feature_map_layer].output)
            feature_model.summary()
            feature_map = feature_model.predict(np.expand_dims(test_img, axis=0))
            return feature_map_layer, feature_map, feature_model
            break 

def decon(feature_map_layer, feature_map, decov_method = 'deconvnet'):
    for i in range(feature_map_layer, -1, -1):
        if i == 0:
            feature_map = tf.nn.relu(feature_map)       
            return normal(rev_action['deconv_to_input_layer'](feature_map, i))
        elif 'conv' in layers_name[i]:
            feature_map = tf.nn.relu(feature_map)
            feature_map = rev_action['deconv'](feature_map, i)
        elif 'max_pooling' in layers_name[i]:
            feature_map = rev_action['rev_maxpool'](feature_map, i)



class_names = {0 : 'airplane',
               1 : 'automobile',
               2 : 'bird',
               3 : 'cat',
               4 : 'deer',
               5 : 'dog',
               6 : 'frog',
               7 : 'horse',
               8 : 'ship',
               9 : 'truck'}

rev_action = {
    'deconv_to_input_layer' : lambda input_map , i: tf.nn.conv2d_transpose(
                                                input_map - layers_biases[i],
                                                layers_weights[i], 
                                                output_shape = (1, feature_model.inputs[0].get_shape()[1],
                                                                feature_model.inputs[0].get_shape()[2], 
                                                                feature_model.inputs[0].get_shape()[3]),
                                                strides = layers_config[i]['strides'],
                                                padding = layers_config[i]['padding'].upper(),
                                                data_format='NHWC').numpy(),
    
    'deconv' : lambda input_map , i: tf.nn.conv2d_transpose(
                                                input_map - layers_biases[i],
                                                layers_weights[i], 
                                                output_shape = (1, layers_output_shape[i-1][1],
                                                                layers_output_shape[i-1][2], 
                                                                layers_output_shape[i-1][3]),
                                                strides = layers_config[i]['strides'],
                                                padding = layers_config[i]['padding'].upper(),
                                                data_format='NHWC').numpy(),
    
    'rev_maxpool' : lambda input_map , i: Unpooling2D(poolsize=layers_config[i]['pool_size']).get_output(input_map)
    }


# Reload model
model=keras.models.load_model('./WithEarlyStopping_L2_1/full_model/model.h5')

layers_weights = []
layers_biases = []
layers_number = []
layers_config = []
layers_name = []
layers_output_shape = []

(train_images, train_labels), (test_images, test_labels) = get_data()

test_img = train_images[0]

predict_class = class_names[np.argmax(model.predict(np.expand_dims(test_img, axis=0)))]

get_layers_info(model)

feature_map_layer, feature_map, feature_model = get_feature_map(test_img, model)

decov_img = decon(feature_map_layer, feature_map)

plt.imshow(decov_img)



















