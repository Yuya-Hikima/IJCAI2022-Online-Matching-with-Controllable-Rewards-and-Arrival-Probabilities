import requests
import tarfile
import pandas as pd

#download the data
url='http://www.ischool.utexas.edu/~ml/data/trec-rf10-crowd.tgz'
filename='../data/trec-rf10-crowd.tgz'
urlData = requests.get(url).content
with open(filename ,mode='wb') as f:
  f.write(urlData)

#decompress the data
with tarfile.open(name="../data/trec-rf10-crowd.tgz", mode="r") as tar:
    tar.extract("trec-rf10-crowd/trec-rf10-data.txt",path='../data')
    tar.extract("trec-rf10-crowd/trec-rf10-readme.txt",path='../data')

#write the data as csv
read_text_file = pd.read_csv ("../data/trec-rf10-crowd/trec-rf10-data.txt", sep='\s+')
read_text_file.to_csv ("../work/trec-rf10-data.csv", index=None)
