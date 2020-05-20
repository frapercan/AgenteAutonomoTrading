from influxdb_client import InfluxDBClient
from influx_client import influxdbClient
from stock_pair_sync_influxdb import DataSync
from data_preprocessing import DataPreprocessor
import logging
import pendulum
import time

logging.basicConfig(filename='log', level=logging.DEBUG)

params = {
    ## Config Params
    "url": "http://localhost:9999",
    "org": "US",
    "token": "DGgBPTM4HxiaHQMGPOcrEuKvnNp3sq-E1EaJ3gWwPml2HOTk29a0q6t4gy2RBANjtaHuec5Pm953yoXrI3Fs6Q==",
    "http_api_url": 'https://api-pub.bitfinex.com/v2/',
    "pair": 'tBTCUSD',
    "data_scale": '1m',
    "test_set": 5,  # dias
    "chunks_number": 1,
    "data_chunk_length": 5,  # dias
    "horizon": 1,  # dia
    "forecast": 1,
    ## Control of the system
    "sync": True,
    "generate_training_data": True,
    "num_cores": 1}

if __name__ == '__main__':
    try:
        client = influxdbClient(params['url'], params['token'], params['org'])
        if params['sync']:
            DataSync(client, params['pair'], params['data_scale'], params['http_api_url']).run()
        if params['generate_training_data']:
            dp = DataPreprocessor(client,
                                  params['pair'],
                                  params['data_scale'],
                                  params['test_set'],
                                  params['chunks_number'],
                                  params['data_chunk_length'],
                                  params['horizon'],
                                  params['forecast'],
                                  params['num_cores'])
            dp.get_maximum_values()
            dp.get_minimum_values()
            dp.data_to_local_chunks_features()
    except Exception as e:
        print(e)
        logging.warning(msg=str(e) + ':' + pendulum.now().to_datetime_string())
