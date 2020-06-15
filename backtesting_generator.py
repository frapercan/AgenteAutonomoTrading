
import keras
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


class BacktestingDataGenerator(keras.utils.Sequence):
    def __init__(self):
        self.batch_size = 1
        self.X = None
        self.Y = None
        self.CLOSE = None
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        data = np.load(os.path.join("./validation_data/")+"valset.npz")
        self.X = data['X']
        self.Y = data['Y']
        self.CLOSE = data['CLOSE']



    def __getitem__(self, index):

        return self.X[index*self.batch_size:(index+1)*self.batch_size], self.Y[index*self.batch_size:(index+1)*self.batch_size],self.CLOSE[index*self.batch_size:(index+1)*self.batch_size]