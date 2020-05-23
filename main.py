from influxdb_client import InfluxDBClient
from influx_client import influxdbClient
from stock_pair_sync_influxdb import DataSync
from extract_transform_store import ETS
from train_data_generator import DataGenerator
from validation_data_generator import ValidationDataGenerator
from model_handler import ModelHandler
import logging
import pendulum
import time
import matplotlib.pyplot as plt


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
    "chunks_number": 10,
    "data_chunk_length": 90,  # dias
    "horizon": 60,  # dia
    "forecast": 1,
    "output_feature": "close_return",
    "batch_size": 216,
    "epochs": 1,
    "learning_rate":0.001,

    ## Control of the system
    "sync": False,
    "generate_training_data": False,
    "train": True,
    "num_cores": 10}

if __name__ == '__main__':
    try:
        client = influxdbClient(params['url'], params['token'], params['org'])

        if params['sync']:
            DataSync(client, params['pair'], params['data_scale'], params['http_api_url']).run()

        if params['generate_training_data']:
            extract_transform_store = ETS(client,
                                  params['pair'],
                                  params['data_scale'],
                                  params['test_set'],
                                  params['chunks_number'],
                                  params['data_chunk_length'],
                                  params['horizon'],
                                  params['forecast'],
                                  params['output_feature'],
                                  params['num_cores'])
            extract_transform_store.set_data_limits()
            extract_transform_store.data_to_local_chunks_features()
            extract_transform_store.generate_validation_set()

        if params['train']:
            dg = DataGenerator(params['batch_size'])
            vdg = ValidationDataGenerator(1)
            mh = ModelHandler(params['horizon'], params['forecast'], params['batch_size'], params['epochs'])
            mh.compile(params['learning_rate'])
            mh.load()
            #mh.train(dg,vdg)
            mh.plot(vdg)




    except Exception as e:
        print(e)
        logging.warning(msg=str(e) + ':' + pendulum.now().to_datetime_string())
