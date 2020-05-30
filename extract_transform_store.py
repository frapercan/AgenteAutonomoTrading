import pendulum
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt




def dataset_sliding_window_supervised(dataset, horizon, forecast, output):
    num_features = dataset.shape[1]
    num_samples = dataset.shape[0] - horizon - forecast
    X = np.zeros((num_samples, horizon, dataset.shape[1]))
    Y = np.zeros((num_samples))
    for i in range(num_samples):
        subset = np.array(dataset[i:i + horizon, :num_features])
        X[i, :, :] = subset
        log_change = np.sum(np.array(dataset[i + horizon:i + horizon + forecast, output]))
        if log_change < 0:
            Y[i] = -1
        else:
            Y[i] = 1

    return X, Y


class ETS:
    def __init__(self, client, pair, data_scale, test_set, chunks_number, data_chunk_length, horizon,
                 forecast, output_feature, num_cores):
        self.client = client
        self.pair = pair
        self.data_scale = data_scale
        self.test_set = test_set
        self.chunks_number = chunks_number
        self.data_chunk_length = data_chunk_length
        self.horizon = horizon
        self.forecast = forecast
        self.output_feature = output_feature
        self.num_cores = num_cores
        self.features_range = {}

    def generate_features_from_iter(self, iteration):
        client = self.client.generate_client()
        train_test_pivot = self.test_set

        start = str(train_test_pivot + self.data_chunk_length * (iteration + 1))
        stop = str(train_test_pivot + self.data_chunk_length * iteration)
        print('aprox slice start: ', pendulum.now().subtract(days=int(start)))

        query = \
            'from(bucket: "{pair}") \
          |> range(start: -{start}d, stop: -{stop}d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
          |> filter(fn: (r) => r["_field"] == "close_return" or r["_field"] == "high_return" or r["_field"] == "low_return" or r["_field"] == "open_return" or r["_field"] == "volume_return")\
          |> yield(name: "raw") \
                '.format(pair=self.pair, start=start, stop=stop, data_scale=self.data_scale)
        data_response = client.query_api().query_data_frame(query)
        values = np.array([ table._value for table in data_response])
        columns = [ table._field[0] for table in data_response]
        scale = MinMaxScaler((-1,1))
        feature_range = np.vstack(([ self.features_range['max'+column] for column in columns],[ -self.features_range['max'+column] for column in columns]))
        scale.fit(feature_range)
        values = scale.transform(values.transpose())

        volume_feature = columns.index("volume_return")
        values[:,volume_feature] = values[:,volume_feature]/10
        plt.plot(values)
        output_feature = columns.index(self.output_feature)
        X,Y = dataset_sliding_window_supervised(values,self.horizon,self.forecast,output_feature)
        plt.show()
        print("entra")
        np.savez_compressed(os.path.join("./training_data/")+str(iteration), X=X, Y=Y)
        print("guardado")


    def generate_validation_set(self):
        client = self.client.generate_client()
        query = \
            'from(bucket: "{pair}") \
          |> range(start: -{test_set}d, stop: 0d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
          |> filter(fn: (r) => r["_field"] == "close_return" or r["_field"] == "high_return" or r["_field"] == "low_return" or r["_field"] == "open_return" or r["_field"] == "volume_return")\
          |> yield(name: "raw") \
        '.format(pair=self.pair, test_set=self.test_set)
        data_response = client.query_api().query_data_frame(query)
        start = data_response[0]._time.values[0]
        stop = data_response[0]._time.values[-1]
        values = np.array([ table._value for table in data_response])

        columns = [ table._field[0] for table in data_response]

        scale = MinMaxScaler((-1,1))
        feature_range = np.vstack(([ self.features_range['max'+column] for column in columns],[ -self.features_range['max'+column] for column in columns]))
        scale.fit(feature_range)
        values = scale.transform(values.transpose())
        output_feature = columns.index(self.output_feature)
        volume_feature = columns.index("volume_return")
        values[:,volume_feature] = values[:,volume_feature]/10
        plt.plot(values)
        plt.show()
        X,Y = dataset_sliding_window_supervised(values,self.horizon,self.forecast,output_feature)
        np.savez_compressed(os.path.join("./validation_data/")+"valset", X=X, Y=Y)





    def data_to_local_chunks_features(self):
        for chunk_number in range(int(self.chunks_number / self.num_cores + 1)):
            p = Pool(self.num_cores)
            p.map(self.generate_features_from_iter, [i for i in range(chunk_number * self.num_cores,
                                                                      chunk_number * self.num_cores + self.num_cores) if self.chunks_number>i ])

    def get_minimum_values(self):
        print('Obteniendo los valores inferiores del conjunto de entrenamiento.')
        client = self.client.generate_client()
        query_min = \
            'from(bucket: "{pair}") \
          |> range(start: -2000d, stop: -{test_set}d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
        |> filter(fn: (r) => r["_field"] == "close" or r["_field"] == "close_return" or r["_field"] == "high" or r["_field"] == "high_return" or r["_field"] == "low" or r["_field"] == "low_return" or r["_field"] == "open" or r["_field"] == "open_return" or r["_field"] == "volume" or r["_field"] == "volume_return")\
          |> min() \
                    '.format(pair=self.pair, test_set=self.test_set, data_scale=self.data_scale)
        table = client.query_api().query_data_frame(query_min)
        columns = table._field.values
        values = table._value
        for column,value in zip(columns,values):
            self.features_range['min'+column] = value

        client.close()
        print("Valores inferiores almacenados")

    def set_data_limits(self):
        print('Obteniendo los valores superiores del conjunto de entrenamiento.')
        client = self.client.generate_client()
        query_max = \
            'from(bucket: "{pair}") \
            |> range(start: -2000d, stop: -{test_set}d)\
            |> filter(fn: (r) => r._measurement == "{pair}") \
              |> filter(fn: (r) => r["_field"] == "close" or r["_field"] == "close_return" or r["_field"] == "high" or r["_field"] == "high_return" or r["_field"] == "low" or r["_field"] == "low_return" or r["_field"] == "open" or r["_field"] == "open_return" or r["_field"] == "volume" or r["_field"] == "volume_return")\
            |> max()\
            '.format(pair=self.pair, test_set=self.test_set, data_scale=self.data_scale)
        table = client.query_api().query_data_frame(query_max)
        columns = table._field.values
        values = table._value
        for column,value in zip(columns,values):
            self.features_range['max' + column] = value
            self.features_range['min' + column] = -value
        client.close()
        print(self.features_range)
        print("Límites del histórico de datos almacenados")






