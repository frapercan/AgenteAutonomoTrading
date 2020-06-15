from influxdb_client import InfluxDBClient


class influxdbClient:
    def __init__(self, url, token, org):
        self.url = url
        self.token = token
        self.org = org

    def generate_client(self):
        return InfluxDBClient(url=self.url, token=self.token, org=self.org)
