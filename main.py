from influxdb_client import InfluxDBClient
from influx_client import influxdbClient
from stock_pair_sync_influxdb import DataSync
from extract_transform_store import ETS
from validation_data_generator import ValidationDataGenerator
from backtesting_generator import BacktestingDataGenerator
from model_handler import ModelHandler
import logging
import pendulum
import time
import matplotlib.pyplot as plt
import numpy as np
from learningratefinder import LearningRateFinder

from wharehouse_data_generator import WharehouseGenerator

logging.basicConfig(filename='log', level=logging.DEBUG)

params = {#2013-08-16T21:46:46.094304+02:00
    ## General Params
    "url": "http://localhost:9999",
    "org": "US",
    "token": "DGgBPTM4HxiaHQMGPOcrEuKvnNp3sq-E1EaJ3gWwPml2HOTk29a0q6t4gy2RBANjtaHuec5Pm953yoXrI3Fs6Q==",
    "http_api_url": 'https://api-pub.bitfinex.com/v2/',

    ## Data collecting params
    "pair": 'tBTCUSD',  #
    "data_scale": '1m',  #
    ## Data generator params
    "test_set": 30,  # dias
    "chunks_number": 84,
    "data_chunk_length": 30,  # dias
    "horizon": 60,  # minutos
    "forecast": 5,  # minutos
    "output_feature": "close_return",
    "num_cores": 10,
    "training_directory": "training_data",
    ### Train params
    "batch_size": 60,
    "epochs": 50,
    "learning_rate":0.001
    ,
    ## Control of the system
    "sync": False,
    "generate_data": False,
    "train": False,
    "backtest": True
}

if __name__ == '__main__':
    try:
        client = influxdbClient(params['url'], params['token'], params['org'])

        if params['sync']:
            DataSync(client, params['pair'], params['data_scale'], params['http_api_url']).run()

        if params['generate_data']:
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
            #extract_transform_store.data_to_local_chunks_features()
            extract_transform_store.generate_validation_set()


        if params['train']:

            dg = WharehouseGenerator(params['batch_size'],params['training_directory'])
            vdg = ValidationDataGenerator(params['batch_size'])
            mh = ModelHandler(params['horizon'], params['forecast'], params['batch_size'], params['epochs'],
                              params['learning_rate'])

            #lrf = LearningRateFinder(mh.model)
            #lrf.find(
            #    dg,
            #    1e-9, 10,
            #    stepsPerEpoch=np.ceil((len(dg) / float(params['batch_size']))),
            #    batchSize=params['batch_size'])
            #lrf.plot_loss()

            #mh.load("tggPlasmado.h5")
            mh.train(dg,vdg)
            mh.evaluate(vdg)
            mh.plot(vdg)
            mh.save("modelo.h5")
        if params['backtest']:
            dg = WharehouseGenerator(params['batch_size'],params['training_directory'])
            bdg = BacktestingDataGenerator()
            vdg = ValidationDataGenerator(params['batch_size'])
            mh = ModelHandler(params['horizon'], params['forecast'], params['batch_size'], params['epochs'],
                              params['learning_rate'])
            mh.load('pesosTFG.h5')
            i= 0
            for X,Y,CLOSE in bdg:
                prediction = mh.model.predict(X)[0][0]
                prediction = 1 if prediction > 0 else 0
                Y = 1 if Y > 0 else 0
                if prediction == Y:
                    plt.scatter(i,prediction,c = 'g')
                else:
                    plt.scatter(i, Y, c='r')
                i = i+1
                if i == 60:
                    break
            plt.show()





    except Exception as e:
        print(e)
        logging.warning(msg=str(e) + ':' + pendulum.now().to_datetime_string())
