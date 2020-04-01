#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:33:49 2020

@author: po-wei
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import datasets, layers, models, Model, regularizers
import matplotlib.pyplot as plt
import os
import json

from EarlyStoppingAtMinLoss import EarlyStoppingAtMinLoss
from LearningRateScheduler import LearningRateScheduler, lr_schedule

# 訓練參數
learning_rate = 0.001
epochs = 25
batch_size = 32
n_classes = 10



def loss_plot(history):
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.ylim([min(min(history.history['accuracy']), min(history.history['val_accuracy'])), 1])
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('epoch')
    plt.show()
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.ylim([0, max(max(history.history['loss']), max(history.history['val_loss']))])
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 輸出0-9轉換爲ont-hot形式
train_labels = tf.keras.utils.to_categorical(train_labels, n_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, n_classes)


train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation= 'softmax'))

model.summary()


#save the model architecture as json format
json_config = json.dumps(model.to_json())
with open('./WithEarlyStopping/architecture/model.json', 'w') as fp:
    fp.write(json_config)

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "./WithEarlyStopping/checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# 編譯模型
'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
'''
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#callback函數
early_stop = EarlyStoppingAtMinLoss(patience = 5)
learning_rate_schedule = LearningRateScheduler(lr_schedule) #把函式lr_schedule丟入class中

# Create a callback that saves the model's weights every single epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=1)

#打印模型# verbose=1顯示進度條
history = model.fit(train_ds, epochs = epochs, verbose=1,validation_data = test_ds, callbacks = [early_stop, learning_rate_schedule, cp_callback])


























