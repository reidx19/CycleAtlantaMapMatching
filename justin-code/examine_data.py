# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:39:57 2023

@author: tpassmore6
"""

#imports
import pandas as pd
import geopandas as gpd
import fiona
import glob
from shapely.geometry import shape, Point, LineString, MultiPoint, box
import datetime
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

export_fp = Path.home() / 'Downloads/cleaned_trips/justin'

sample_trips = pd.read_csv(export_fp/'sample_mm_trips.csv')

sample_trips['geometry'] = gpd.points_from_xy(sample_trips['lon'],sample_trips['lat'])
sample_trips = gpd.GeoDataFrame(sample_trips,geometry='geometry',crs='epsg:4326')
sample_trips.to_file(export_fp/'sample_trips.gpkg',layer='sample_trips')

#cluster
#DBSCAN clustering seems to work pretty well for clustering into similar areas
#https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/how-density-based-clustering-works.htm
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

#import clustered ones
clustered_trips = gpd.read_file(export_fp/'sample_trips.gpkg',layer='Clusters')

# initialize empty dataframe and dictionary for adding cleaned data
coords = clustered_trips.copy()
coords_dict = {}

# drop unneeded ones
coords.drop(columns=['Unnamed: 0'],inplace=True)
coords.rename(columns={'trip_id':'tripid'},inplace=True)

# convert speed and accuracy to imperial units (mph and ft)
coords['speed'] = coords['speed'] * 2.2369362920544025
# coords['hAccuracy'] = coords['hAccuracy'] * 3.28084 

# change dates to datetime
coords['datetime'] = pd.to_datetime(coords['collect_time'])

# sort everything
coords.sort_values(['tripid','datetime'],inplace=True)

# count and print number of trips
print(f"{coords['tripid'].nunique()} initial trips found")

#%% Create trips_df
    
'''
These next lines aggregate the points into information about the trips and appends
them to trips df
''' 
    
#get start time
start_time = coords.groupby('tripid')['datetime'].min()
start_time.rename('start_time',inplace=True)

#get end time
end_time = coords.groupby('tripid')['datetime'].max()
end_time.rename('end_time',inplace=True)

#get duration
duration = end_time - start_time
duration.rename('duration',inplace=True)

#get starting location
start_lon = coords.groupby('tripid')['datetime'].idxmin().map(coords['lon'])
start_lat = coords.groupby('tripid')['datetime'].idxmin().map(coords['lat'])
start_lon.rename('start_lon',inplace=True)
start_lat.rename('start_lat',inplace=True)

#get ending location
end_lon = coords.groupby('tripid')['datetime'].idxmax().map(coords['lon'])
end_lat = coords.groupby('tripid')['datetime'].idxmax().map(coords['lat'])
end_lon.rename('end_lon',inplace=True)
end_lat.rename('end_lat',inplace=True)

#get number of points
num_of_points = coords['tripid'].value_counts()
num_of_points.rename('initial_tot_points',inplace=True)

#get average haccuracy
avg_accuracy = coords.groupby('tripid')['accuracy'].mean()
avg_accuracy.rename('avg_accuracy',inplace=True)

#turn into df
trips_df = pd.concat([start_time,end_time,duration,start_lon,start_lat,end_lon,end_lat,num_of_points,avg_accuracy],axis=1)
trips_df.reset_index(inplace=True)
trips_df.rename(columns={'index':'tripid'},inplace=True)

#make status column
trips_df['status'] = 'retain'

# ### hAccuracy ###
# '''
# In this step we remove points if the haccracy value is more than 2.5 standard deviations above the mean value.
# '''
# hAccuracy_filt = coords.groupby('tripid')['accuracy'].transform(lambda x: (x - x.mean()) > (x.std() * 2.5))
# coords = coords[-hAccuracy_filt]


#%% seperate trips and store in dictionary for further processing
coords_dict.update({tripid : df.reset_index(drop=True) for tripid, df in coords.groupby('tripid')})


#%% export step

#export to final 
with (export_fp/'coords_dict.pkl').open('wb') as fh:
    pickle.dump(coords_dict,fh)

#initialize a matching column
trips_df['matching status'] = 'unmatched'
trips_df['match_ratio'] = np.nan

#export trip_df
trips_df.to_csv(export_fp/'trips.csv',index=False)
