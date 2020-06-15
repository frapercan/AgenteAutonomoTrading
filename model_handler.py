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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
from tcn import TCN, tcn_full_summary
from keras.initializers import RandomNormal



class ModelHandler:
    def __init__(self, horizon, forecast, batch_size,epochs, learning_rate):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.forecast = forecast
        initializer = RandomNormal(mean=0., stddev=1.)
        model = Sequential()
        model.add(LSTM(60, kernel_initializer = initializer,input_shape=(horizon, 5), return_sequences=True,dropout=0.1,recurrent_dropout=0.1))
        model.add(LSTM(60, kernel_initializer = initializer,dropout=0.1,recurrent_dropout=0.1, return_sequences=False))
        model.add(Dense(32, kernel_initializer=initializer, activation="tanh"))
        model.add(Dense(1, kernel_initializer=initializer, activation="tanh"))
        self.model = model
        self.model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=learning_rate))
        model.summary()




    def load(self,name):
        self.model.load_weights(name)

    def train(self,generator,validation_generator):


        clr = CyclicLR(
        	mode="triangular",
        	base_lr=1e-5,
        	max_lr=1e-2,
        	step_size=400)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=3, min_lr=0.00001, verbose=1)

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=999)

        checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        history = self.model.fit(generator, validation_data=validation_generator, epochs=self.epochs,
                                 shuffle=True, verbose=1, callbacks=[checkpoint,early_stopping,clr])

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
        num_plots = 20
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
        predictions = []
        real = []
        for X,Y in generator:
            predictions.extend(self.model.predict(X))
            real.extend(Y)
        predictions = [1 if p  > 0 else 0 for p in predictions]
        real = [1 if r  > 0 else 0 for r in real]
        print("accuracy: ",accuracy_score(predictions, real))
        print("f1: ",f1_score(predictions, real))
        print("precision: ", precision_score(predictions, real))
        print("recall_score: ", recall_score(predictions, real) )

















    def backtesting(self,generator,initial_money):
        cash = initial_money
        invest = 0
        position = False
        i = 0
        close_prices = []
        hold_amount = 0
        values = []
        stay_out = []
        for X, Y, CLOSE in generator:
            if not i:
                hold_amount = cash/CLOSE

            if i % self.forecast:
                prediction = 1 if self.model.predict(X) > -0.3 else 0

                if position != prediction:
                    if prediction:
                        invest = cash/CLOSE #- cash/CLOSE*0.01
                        cash = 0
                        position = True
                    else:
                        cash = invest * CLOSE #- invest * CLOSE * 0.01
                        invest = 0
                        position = False
                close_prices.append(hold_amount*CLOSE)
                values.append(cash if cash else invest*CLOSE)
                stay_out.append(initial_money)

            i = i+1
            print(i)
        plt.plot(values, label="ML strategy")
        plt.plot(close_prices, label='hold')
        plt.plot(stay_out, label='stay out')
        plt.legend()
        plt.show()




    def save(self,name):
        self.model.save_weights(name)

