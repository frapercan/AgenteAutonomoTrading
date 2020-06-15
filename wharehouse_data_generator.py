import numpy as np
import os
import keras
import glob
import matplotlib.pyplot as plt
import copy
import random
import time


class WharehouseGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, training_directory):
        self.batch_size = batch_size
        self.filenames = os.listdir(os.path.join(training_directory))
        self.filenames_indices = [i for i in range(len(self.filenames))]
        self.num_files = len(self.filenames)
        self.length = 0
        self.indices = []
        self.local_indice = -1
        self.file_index = 0
        for i in range(len(self.filenames)):
            data = np.load(os.path.join("training_data", self.filenames[i]))
            X = data['X']
            self.length = self.length + X.shape[0]
            self.indices.extend(np.ones(X.shape[0]) * i)
            if i == 0:  ## initialization of the loop
                self.load(self.filenames[self.filenames_indices[i]])

    def __len__(self):
        return int(np.floor(self.length/self.batch_size))
    def on_epoch_end(self):
        self.file_index = 0
        self.local_indice = -1
    def __getitem__(self, index):
        current_file = int(self.indices[index * self.batch_size])
        if self.file_index != current_file:
            self.file_index = current_file
            #print(self.filenames[self.filenames_indices[self.file_index]])
            self.load(self.filenames[self.filenames_indices[self.file_index]])
            self.local_indice = -1
        self.local_indice = self.local_indice + 1
        X = self.X[self.local_indice * self.batch_size:(self.local_indice + 1) * self.batch_size]
        Y = self.Y[self.local_indice * self.batch_size:(self.local_indice + 1) * self.batch_size]
        return X , Y
    def load(self, filename):
        data = np.load(os.path.join("training_data", filename))
        XY = list(zip(data['X'], data['Y']))
        random.shuffle(XY)
        X, Y = zip(*XY)
        self.X = np.array(X)
        self.Y = np.array(Y)
