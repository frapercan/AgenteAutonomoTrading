from keras import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dropout, LSTM, Dense, Bidirectional, Flatten, TimeDistributed
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
import time
from keras.layers import BatchNormalization
from clr_callback import CyclicLR
from keras.layers import Input
from keras.metrics import accuracy
from keras_self_attention import SeqSelfAttention
import numpy as np
from sklearn.metrics import accuracy_score
import time
from tcn import TCN, tcn_full_summary



class ModelHandler:
    def __init__(self, horizon, forecast, batch_size,epochs, learning_rate):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        model = Sequential()
        model.add(LSTM(200, input_shape=(horizon, 5), return_sequences=False))
        model.add(Dense(1, activation="tanh"))
        self.model = model
        self.model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=learning_rate))
        model.summary()




    def load(self,name):
        self.model.load_weights(name)

    def train(self,generator,validation_generator):


        clr = CyclicLR(
        	mode="triangular",
        	base_lr=1e-5,
        	max_lr=1e-3,
        	step_size=403*4)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=3, min_lr=0.00001, verbose=1)

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

        checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        history = self.model.fit(generator, validation_data=validation_generator, epochs=self.epochs,
                                 shuffle=True, verbose=2, callbacks=[checkpoint,clr])

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        N = np.arange(0, len(clr.history["lr"]))
        plt.figure()
        plt.plot(N, clr.history["lr"])
        plt.title("Cyclical Learning Rate (CLR)")
        plt.xlabel("Training Iterations")
        plt.ylabel("Learning Rate")
        plt.show()

    def plot(self,generator):
        num_plots = 10
        Y_array = []
        prediction_array = []
        for X, Y in generator:
            prediction = self.model.predict(X)
            plt.plot(prediction,c="r")
            plt.plot(Y)
            pred = [1 if p  > 0 else 0 for p in prediction]
            plt.show()
            plt.plot(pred)
            Y_prima = [1 if y  > 0 else 0 for y in Y]
            plt.plot(Y_prima)
            plt.show()
            print(accuracy_score(pred,Y_prima))
            num_plots = num_plots -1
            if num_plots == 0:
                break






    def evaluate(self,generator):
        #mse = self.model.evaluate(generator)
        predictions = []
        real = []
        for X,Y in generator:
            predictions.extend(self.model.predict(X))
            real.extend(Y)
        predictions = [1 if p  > 0 else 0 for p in predictions]
        real = [1 if r  > 0 else 0 for r in real]
        print(accuracy_score(predictions, real))

















    def backtesting(self,generator):
        Y_array = []
        prediction_array = []
        for X, Y in generator:
            prediction_array.append(self.model.predict(X)[0])
            Y_array.append(Y[0])



    def save(self,name):
        self.model.save_weights(name)

