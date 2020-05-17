from influxdb_client import InfluxDBClient, Point
from sklearn.preprocessing import MinMaxScaler
import pendulum
import numpy as np
import pandas as pd
import threading, queue

params = {
    "url": "http://localhost:9999",
    "org": "US",
    "token": "zDxSwoznrqSipuTs2VJfrOJl-e2twnCKsraGd-K_4YKUY0c_EB9fb341_kCi0lQqz9dUx_yWYGKUqiZaI2cGOA==",
    "http_api_url": 'https://api-pub.bitfinex.com/v2/',
    "pair": 'tBTCUSD',
    "test_set":"5", #dias
    "chunks_number":50,
    "data_chunk_length":"90", #dias
    "horizon":"1", #dia
    "forecast":"1"} #hora

client = InfluxDBClient(
    url=params['url'],
    token=params['token'],
    org=params['org']
)

limit = 500000

validation_data_query = 'from(bucket: "{}") ' \
        '|> range(start: -{}d) ' \
        '|> filter(fn: (r) => r._measurement == "{}")'.format(params['pair'],params['test_set'],params['pair']) #count
val_set = client.query_api().query_data_frame(validation_data_query)
data_test = pd.DataFrame(np.vstack([table._value for table in val_set]).transpose(), columns = [table._field[0] for  table in val_set])

train_test_pivot = params['test_set']

query_min = 'from(bucket: "{}") \
  |> range(start: 0, stop: -{}d) \
  |> filter(fn: (r) => r._measurement == "{}") \
  |> min()'.format(params['pair'],params['test_set'],params['pair'])
min_values = client.query_api().query_data_frame(query_min)._value.values

query_max = 'from(bucket: "{}") \
  |> range(start: 0, stop: -{}d) \
  |> filter(fn: (r) => r._measurement == "{}") \
  |> max()'.format(params['pair'],params['test_set'],params['pair'])


max_values = client.query_api().query_data_frame(query_max)._value.values

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.min_ = min_values
scaler.max_ = max_values

print(scaler.max_)
for i in range(params['chunks_number']):
    start = train_test_pivot +'d' +'-'+ str(int(params['data_chunk_length']) * (i+1)) + 'd' +"-" + params['horizon']+"h"
    stop = train_test_pivot +'d' +'-'+ str(int(params['data_chunk_length']) * (i)) + 'd' +"-" + params['horizon']+"h"
    print("start: ", start, "stop: ",stop)
    training_data_query = 'from(bucket: "{}") ' \
                            '|> range(start: -{}, stop: -{}) ' \
                            '|> filter(fn: (r) => r._measurement == "{}")'.format(params['pair'],
                                                                                  start,
                                                                                  stop,

                                                                                  params['pair'])
    trainning_set = client.query_api().query_data_frame(training_data_query)
    print(trainning_set[0]._time)
    trainning_set = pd.DataFrame(np.vstack([table._value for table in trainning_set]).transpose(),
                             columns=[table._field[0] for table in trainning_set])
    trainning_set['close_pct_change'] = trainning_set.close.pct_change()
    trainning_set['open_pct_change'] = trainning_set.open.pct_change()
    trainning_set['high_pct_change'] = trainning_set.high.pct_change()
    trainning_set['low_pct_change'] = trainning_set.low.pct_change()
    trainning_set['volume_log'] = np.log(1+trainning_set.volume)

    q.put(trainning_set)


    print(trainning_set.open_pct_change.max(),'hola')
