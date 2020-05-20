import pendulum
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, client, pair, data_scale, test_set, chunks_number, data_chunk_length, horizon,
                 forecast, num_cores):
        self.client = client
        self.pair = pair
        self.data_scale = data_scale
        self.test_set = test_set
        self.chunks_number = chunks_number
        self.data_chunk_length = data_chunk_length
        self.horizon = horizon
        self.forecast = forecast
        self.num_cores = num_cores
        self.features_range = {}
        self.scalers = None

    def generate_features_from_iter(self, iteration):
        client = self.client.generate_client()
        train_test_pivot = self.test_set

        start = str(train_test_pivot + self.data_chunk_length * (iteration + 1))
        stop = str(train_test_pivot + self.data_chunk_length * iteration)
        print('aprox slice start: ', pendulum.now().subtract(days=int(start)))

        query = \
            'import "math"\
            from(bucket: "{pair}") \
          |> range(start: -{start}d, stop: -{stop}d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
          |> filter(fn: (r) => r._field == "close" or r._field == "high" or r._field == "low" or r._field == "open") \
          |> yield(name: "raw") \
          from(bucket: "{pair}") \
          |> range(start: -{start}d, stop: -{stop}d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
          |> filter(fn: (r) => r._field == "close" or r._field == "high" or r._field == "low" or r._field == "open") \
          |> derivative(unit: {data_scale}, nonNegative: false, columns: ["_value"], timeColumn: "_time") \
          |> yield(name: "diff")  \
          from(bucket: "{pair}") \
          |> range(start: -{start}d, stop: -{stop}d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
          |> filter(fn: (r) => r._field == "volume") \
          |> map(fn: (r) => ({{ r with _value:  math.log1p(x: r._value) }})) \
          |> yield(name: "log") \
                    '.format(pair=self.pair, start=start, stop=stop, data_scale=self.data_scale)
        data_response = client.query_api().query_data_frame(query)
        values = {}
        features = []
        for table in data_response:
            for result, value, field in zip(table.result, table._value, table._field):
                scaled_value = 2*((value - self.features_range['min'+result+field])/(self.features_range['max'+result+field]-self.features_range['min'+result+field]))-1
                if result+field not in values.keys():
                    values[result+field] = list()
                values[result + field].append(scaled_value)
        for key in values.keys():
            if "diff" not in key:
                values[key] = np.array(values[key])[1:]
            else:
                values[key] = np.array(values[key])
        for key in values.keys():
            if not len(features):
                print('entra')
                features = values[key]
            else:
                features = np.vstack([features, values[key]])
        print(features.transpose().shape)






    def data_to_local_chunks_features(self):
        for chunk_number in range(int(self.chunks_number / self.num_cores + 1)):
            p = Pool(self.num_cores)
            p.map(self.generate_features_from_iter, [i for i in range(chunk_number * self.num_cores,
                                                                      chunk_number * self.num_cores + self.num_cores)])

    def get_minimum_values(self):
        print('Obteniendo los valores inferiores del conjunto de entrenamiento.')
        client = self.client.generate_client()
        query_min = \
            'import "math"\
            from(bucket: "{pair}") \
          |> range(start: -30d, stop: -{test_set}d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
          |> filter(fn: (r) => r._field == "close" or r._field == "high" or r._field == "low" or r._field == "open") \
          |> min() \
          |> yield(name: "raw") \
           \
          from(bucket: "{pair}") \
          |> range(start: -30d, stop: -{test_set}d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
          |> filter(fn: (r) => r._field == "close" or r._field == "high" or r._field == "low" or r._field == "open") \
          |> derivative(unit: {data_scale}, nonNegative: false, columns: ["_value"], timeColumn: "_time") \
          |> min() \
          |> yield(name: "diff")  \
          from(bucket: "{pair}") \
          |> range(start: -30d, stop: -{test_set}d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
          |> filter(fn: (r) => r._field == "volume") \
          |> map(fn: (r) => ({{ r with _value:  math.log1p(x: r._value) }})) \
          |> min() \
          |> yield(name: "log") \
                    '.format(pair=self.pair, test_set=self.test_set, data_scale=self.data_scale)
        for table in client.query_api().query_data_frame(query_min):
            for result, value, field in zip(table.result, table._value, table._field):
                self.features_range['min'+result+field] =  value

        client.close()
        print("Valores inferiores almacenados")

    def get_maximum_values(self):
        print('Obteniendo los valores superiores del conjunto de entrenamiento.')
        client = self.client.generate_client()
        query_max = \
            'import "math"\
            from(bucket: "{pair}") \
          |> range(start: -30d, stop: -{test_set}d)\
          |> filter(fn: (r) => r._measurement == "{pair}") \
          |> filter(fn: (r) => r._field == "close_return" or r._field == "high_return" or r._field == "low_return"' \
            ' or r._field == "open_return" or r._field == "volume_return) \
          |> max() \
                    '.format(pair=self.pair, test_set=self.test_set, data_scale=self.data_scale)
        for table in client.query_api().query_data_frame(query_max):
            for result, value, field in zip(table.result, table._value, table._field):
                self.features_range['max'+result+field] =  value
        client.close()
        print("Valores superiores almacenados")






