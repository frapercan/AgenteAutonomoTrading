from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dropout, LSTM, Dense
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import numpy as np
from sklearn.linear_model import LinearRegression
import time

import numpy as np



class ModelHandler:
    def __init__(self, horizon, forecast, batch_size,epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        model = Sequential()
        model.add(LSTM(100, input_shape=(horizon, 5), return_sequences=False))
        model.add(Dropout(0.4))
        model.add(Dense(forecast))
        self.model = model

    def compile(self,learning_rate):
        self.model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=learning_rate))
    def load(self):
        self.model.load_weights("best_model.h5")

    def train(self,generator,validation_generator):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=20, min_lr=0.00001, verbose=1)

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        history = self.model.fit(generator, validation_data=validation_generator, epochs=self.epochs,
                       shuffle=True, verbose=2, callbacks=[early_stopping, checkpoint, reduce_lr])

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    def plot(self,generator):
        Y_array = []
        prediction_array = []
        for X, Y in generator:
            prediction_array.append(self.model.predict(X)[0])
            Y_array.append(Y[0])
            plt.plot(Y_array, 'blue')
            plt.plot(prediction_array, 'r')

            plt.show()
            time.sleep(1)
            print((np.sign(prediction_array) == np.sign(Y_array)).sum()/len(prediction_array) * 100)








