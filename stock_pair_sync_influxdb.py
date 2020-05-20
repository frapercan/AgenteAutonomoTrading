import json
import requests
import pendulum
import ntplib  # Network Time protocol
from influxdb_client import Point
import time
import logging
import numpy as np
import math

logging.basicConfig(filename='log', level=logging.DEBUG)

ERROR_CODE_SUBSCRIPTION_FAILED = 10300
ERROR_CODE_RATE_LIMIT = 11010
INFO_CODE_RECONNECT = 20051
ERROR_CODE_START_MAINTENANCE = 20006

NTPClient = ntplib.NTPClient()


class DataSync:
    def __init__(self, client, pair, data_scale, http_api_url):
        self.client = client.generate_client()
        self.pair = pair
        self.data_scale = data_scale
        self.http_api_url = http_api_url

    def run(self):
        last_ts_query = 'from(bucket: "{}") |> range(start: -9999d) |> last()'.format(self.pair)
        last_ts = self.client.query_api().query_data_frame(last_ts_query)
        if '_time' not in last_ts.columns:
            pivot_stamp = 1067681680000  # 2003 - More than enough
        else:
            pivot_stamp = last_ts._time[0].timestamp() * 1000
        while 1:
            print(pendulum.from_timestamp(pivot_stamp / 1000), 'updating')
            # Build url
            url = self.url_generator(pivot_stamp)
            # Request API
            json_response = requests.get(url)
            response = json.loads(json_response.text)
            time.sleep(3)
            if 'error' in response:
                # Check rate limit
                if response[1] == ERROR_CODE_RATE_LIMIT:
                    print('Error: reached the limit number of requests. Wait 120 seconds...')
                    time.sleep(120)
                    continue
                # Check platform status
                elif response[1] == ERROR_CODE_START_MAINTENANCE:
                    print('Error: platform is in maintenance. Forced to stop all requests.')
                    break
            else:
                # Get last timestamp of request (in second, so divided by 1000)
                last_dt = int(response[::-1][0][0]) // 1000
                last_dt = pendulum.from_timestamp(last_dt)

                # Put it as new start datetime (in millisecond, so multiplied by 1000)
                if pivot_stamp == last_dt.int_timestamp * 1000:
                    logging.info('Correctly sync at: {}'.format(pendulum.from_timestamp(pivot_stamp / 1000)))
                    time.sleep(60)
                pivot_stamp = last_dt.int_timestamp * 1000
                self.client.write_api().write(record=self.serialize_points(response), bucket=self.pair)

    def serialize_points(self, response):
        lag_response = True
        points = []
        for tick in list(response):
            if lag_response:
                open_price = float(tick[1])
                low_price = float(tick[2])
                high_price = float(tick[3])
                close_price = float(tick[4])
                volume = float(tick[5])
                lag_response = False
            else:
                open_return = np.log(tick[1] / open_price)
                low_return = np.log(tick[2] / low_price)
                high_return = np.log(tick[3] / high_price)
                close_return = np.log(tick[4] / close_price)
                volume_return = np.log(tick[5] / volume)
                open_price = float(tick[1])
                low_price = float(tick[2])
                high_price = float(tick[3])
                close_price = float(tick[4])
                volume = float(tick[5])
                timestamp = pendulum.from_timestamp(int(tick[0]) // 1000).in_tz('UTC').to_atom_string()
                point = Point(self.pair).field('close', close_price).field('open', open_price).field('high', high_price) \
                    .field('low', low_price)\
                    .field('volume', volume)\
                    .time(timestamp)\
                    .field('open_return', open_return)\
                    .field('low_return', low_return)\
                    .field('high_return', high_return)\
                    .field('close_return', close_return)\
                    .field('volume_return', volume_return)
                points.append(point)
        # Return True is operation is successful
        return points

    def url_generator(self, pivot_stamp):
        return self.http_api_url + 'candles/trade:{data_scale}:{pair}/' \
                                   'hist?limit={limit}&start={pivot_stamp}&sort=1' \
            .format(data_scale=self.data_scale, pair=self.pair, limit=10000, pivot_stamp=pivot_stamp)
