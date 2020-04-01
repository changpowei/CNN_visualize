#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:27:00 2020

@author: po-wei
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--layer_name", default = 'block5_conv1', type = str, help = "Kernel from which layer?")
parser.add_argument("-e", "--epochs", default = 150, type = int, help = "How many epoches?")
parser.add_argument("-s", "--step_size", default = 1., type = float, help = "Like learning rate!")
parser.add_argument("-i", "--filter_index", default = 0, type = int, help = "which Kernel from the layer?")
args = parser.parse_args()

layer_name = args.layer_name
epochs = args.epochs
step_size = args.step_size
filter_index = args.filter_index


def normal(img):
    if np.ndim(img) == 4:
        img = np.squeeze(img, axis=0)
    img_norm = (img - np.min(img)) * 255 /np.ptp(img)
    return img_norm.astype('uint8')

def hisEqulColor(img):
    print("channel numbers %d:"%(np.ndim(img)))
    if np.ndim(img) >= 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        channels[0] = cv2.equalizeHist(channels[0])
        ycrcb = cv2.merge(channels)
        img_hist = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)

    else:
        img_hist = cv2.equalizeHist(img)
    return img_hist

# Create a connection between the input and the target layer
model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])

# Initiate random noise
input_img_data = np.random.random((1, 224, 224, 3))
input_img_data = (input_img_data - 0.5) * 20 + 128.

# Cast random noise from np.float64 to tf.float32 Variable
input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

# Iterate gradient ascents
for i in range(epochs):
    with tf.GradientTape() as tape:
        outputs = submodel(input_img_data)
        loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
    grads = tape.gradient(loss_value, input_img_data)
    normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    input_img_data.assign_add(normalized_grads * step_size)
    print(i)
    
    

img_norm = normal(input_img_data.numpy())
img_hist = hisEqulColor(img_norm)


cv2.imshow("img_norm", img_norm)
cv2.imshow("img_hist", img_hist)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('kernel_norm.jpg', img_norm)









