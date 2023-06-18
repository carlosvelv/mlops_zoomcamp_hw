#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import numpy as np
import os
import sys



if len(sys.argv) >= 3:
    year = int(sys.argv[1])
    month = str(sys.argv[2])
    print("Year:", year)
    print("Month:", month)
else:
    print("Please provide both year and month as command-line arguments.")

output_file = f'./outputs/output_file_{year}_{month}.parquet'


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



df = read_data(f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet")



dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)










df['ride_id'] = f'{year:04d}/{int(month):02d}_' + df.index.astype('str')





df_result = df[['ride_id']].copy()
df_result['result'] = y_pred
df_result.head()



df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


print(f'Mean: {np.mean(y_pred)}')