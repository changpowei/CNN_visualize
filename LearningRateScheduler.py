#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:26:01 2020

@author: po-wei
"""
import tensorflow as tf
import numpy as np


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (5, 0.0005), (10, 0.00025), (15, 0.0001)
    ]

#將函式 lr_schedule丟入class中，透過callback設定不同epoch的learning rate
class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.
    
    Arguments:
        schedule: a function that takes an epoch index
        (integer, indexed from 0) and current learning rate
        as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        #schedule為函式lr_schedule(epoch, lr)
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('\nEpoch %05d: Learning rate is %6.5f.' % (epoch, scheduled_lr))


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    #LR_SCHEDULE為全域變數，一個tuple所組成的list
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr