import keras
import numpy as np
import os
import glob


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.filenames = os.listdir(os.path.join("training_data"))
        self.X = None
        self.Y = None
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        print(np.random.choice(self.filenames))
        data = np.load(os.path.join("training_data",np.random.choice(self.filenames)))
        self.X = data['X']
        self.Y = data['Y']



    def __getitem__(self, index):
        return self.X[index*self.batch_size:(index+1)*self.batch_size],self.Y[index*self.batch_size:(index+1)*self.batch_size]
