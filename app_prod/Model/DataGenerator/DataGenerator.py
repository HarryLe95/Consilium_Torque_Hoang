import tensorflow 
import pandas as pd 
import random
from random import randrange 
import numpy as np 
import keras 
from keras.callbacks import LearningRateScheduler, History
from keras import backend as K
from typing import Tuple 


class data_generator(keras.utils.Sequence):
    def __init__(self, X_all, Y_all, well_id_all, num_classes, batch_size, window_size = None, is_validation=False):
        self.len = 128
        self.X_all = X_all
        self.X_all_roll_avg = pd.DataFrame(X_all).rolling(120).mean().values
        self.Y_all = Y_all
        self.well_id_all = well_id_all
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.window_size = None
        if window_size is not None:
            self.window_size = window_size
        self.is_validation = is_validation

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        X = []
        Y = []
        
        if self.window_size is None:
            window_size = 2 ** (random.randrange(9,13,1))
        else:
            window_size = self.window_size
            
        if self.is_validation:
            augmentation_method = "raw_data"
        
        else:
            augmentation_method = np.random.choice(["raw_data", "rolling_average", "vertical_offset", "compression", "vertical_compression", "vertical_expansion"],
                                                p=[5/13, 4/13, 1/13, 1/13, 1/13, 1/13])

        if augmentation_method == "raw_data":
            # use raw data
            feature_data = self.X_all
        elif augmentation_method == "vertical_offset":
            # from inspection, motor torque medians looked to be within 30 of each other
            motor_torque_offset = random.choice(np.arange(50))
            operation = random.choice(["add", "subtract"])
            if operation == "add":
                feature_data = self.X_all + motor_torque_offset
                feature_data[feature_data == motor_torque_offset] = 0
            else:
                feature_data = self.X_all - motor_torque_offset
                feature_data[feature_data < 0 ] = 0
        elif augmentation_method == "compression":
            # remove every second element in the (1, N) numpy array
            feature_data = np.delete(self.X_all, slice(self.X_all.shape[0], 1, 2))
            feature_data = np.reshape(feature_data, (np.max(feature_data.shape), 1))
        elif augmentation_method == "vertical_compression":
            # remove every second element in the (1, N) numpy array
            feature_data = self.X_all / random.choice([1.2, 1.4, 1.6, 1.8, 2])
        
        elif augmentation_method == "vertical_expansion":
            # remove every second element in the (1, N) numpy array
            feature_data = self.X_all * random.choice([1.2, 1.4, 1.6, 1.8, 2])
        else:
            feature_data = self.X_all_roll_avg
            
        X, Y = self._get_augmented_data(feature_data, window_size, X, Y)

        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        Y = tensorflow.keras.utils.to_categorical(Y, self.num_classes) 
        return X, Y
    
    def _get_augmented_data(self, features_df: pd.DataFrame, window_size: int, X: list, Y: list) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(int(self.batch_size/2)):            
                # get sample with at least one positive class label
                while True:
                    StartInd = randrange(features_df.shape[0]-window_size)
                    if self.well_id_all[StartInd] == self.well_id_all[StartInd+window_size]: # ensure sample is from one well only
    #                     print(StartInd, window_size,self.X_all[StartInd:StartInd+window_size])
                        if not np.isnan(features_df[StartInd:StartInd+window_size]).any(): # ensure no NaN values in sample
                            if np.sum(self.Y_all[StartInd:StartInd+window_size]) > 1: # ensure sample contains at least one postive class label
                                X.append(features_df[StartInd:StartInd+window_size])
                                Y.append(self.Y_all[StartInd:StartInd+window_size])
                                break

                # get sample with any class label
                while True:
                    StartInd = randrange(features_df.shape[0]-window_size)
                    if self.well_id_all[StartInd] == self.well_id_all[StartInd+window_size]: # ensure sample is from one well only
                        if not np.isnan(features_df[StartInd:StartInd+window_size]).any(): # ensure no NaN values in sample
                            X.append(features_df[StartInd:StartInd+window_size])
                            Y.append(self.Y_all[StartInd:StartInd+window_size])
                            break
        return X, Y
# might need to fill NA's/mask. 