import requests
url='https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-03.parquet'
filename='../data/yellow_tripdata_2019-03.csv'
urlData = requests.get(url).content
with open(filename ,mode='wb') as f:
  f.write(urlData)
url='https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-02.parquet'
filename='../data/yellow_tripdata_2019-02.csv'
urlData = requests.get(url).content
with open(filename ,mode='wb') as f:
  f.write(urlData)
url='https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.parquet'
filename='../data/yellow_tripdata_2019-01.csv'
urlData = requests.get(url).content
with open(filename ,mode='wb') as f:
  f.write(urlData)
