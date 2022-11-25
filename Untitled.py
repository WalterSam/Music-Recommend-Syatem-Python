#!/usr/bin/env python
# coding: utf-8
"""
Created on Thu Nov 24 17:47:42 2022

@author: Sam~walter
"""
import warnings
warnings.filterwarnings('ignore')
import seaborn as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()

data = pd.read_csv("spotify.csv")
data.head()
data.info()
data.isnull().sum()

data.dropna(inplace = True)
data.isnull().sum().plot.bar()
plt.show()

data['name'].nunique(), data.shape

data = data.sort_values(by=['popularity'], ascending=False)
data.drop_duplicates(subset=['name'], keep='first', inplace=True)

plt.figure(figsize = (10, 5))
sb.countplot(data['year'])
plt.axis('off')
plt.show()

df = data.drop(columns=['id', 'name', 'artists', 'release_date', 'year'])
df.corr()

from sklearn.preprocessing import MinMaxScaler
datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
normarization = data.select_dtypes(include=datatypes)
for col in normarization.columns:
    MinMaxScaler(col)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
features = kmeans.fit_predict(normarization)
data['features'] = features
MinMaxScaler(data['features'])

class Spotify_Recommendation():
    def __init__(self, dataset):
        self.dataset = dataset
    def recommend(self, songs, amount=1):
        distance = []
        song = self.dataset[(self.dataset.name.str.lower() == songs.lower())].head(1).values[0]
        rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]
        for songs in tqdm(rec.values):
            d = 0
            for col in np.arange(len(rec.columns)):
                if not col in [1, 6, 12, 14, 18]:
                    d = d + np.absolute(float(song[col]) - float(songs[col]))
            distance.append(d)
        rec['distance'] = distance
        rec = rec.sort_values('distance')
        columns = ['artists', 'name']
        return rec[columns][:amount]

recommendations = Spotify_Recommendation(data)
recommendations.recommend("Lovers Rock", 10)





