#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:22:39 2020

@author: po-wei
"""

import tensorflow as tf
import numpy as np

#若val_loss持續3個epochs沒有下降，則early stoping
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.
    
    Arguments:
        patience: Number of epochs to wait after min has been hit. After this
        number of no improvement, training stops.
    """
    
    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        
        self.patience = patience
        
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if np.less(logs.get('val_loss'), self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            print('\nRestoring model weights from the end of the best epoch.\n')
            self.best_weights = self.model.get_weights()
            self.model.set_weights(self.best_weights)
            
            # save the full model which have a smallest loss
            self.model.save('./WithEarlyStopping/full_model/model.h5')
            
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('\nEpoch %05d: early stopping' % (self.stopped_epoch + 1))