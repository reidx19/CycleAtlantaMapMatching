# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:09:45 2021

@author: tpassmore6

This script is for cleaning the GPS traces from CycleAtlanta. It's based on the
GPS_Clean_Route_Snapping.sql script that Dr. Aditi Misra used.

Overview
- Imports and combines all of the coords.csv files
- Add column names and format data
- Find duplicate trips
- Remove all trips with points outside a specified study area (inside the i-285 perimeter for this study)
- Remove all points with a low accuracy reading (need to figure out how to incorporate this with kalman filter)
- Remove all trips less than 5 minutes long OR that have fewer than 5 minutes of points recorded
- Remove all points that are more than three hours past the starting time
- Use Kalman filter to update coordinates and fill in missing timestamps (if needed)
- Search for unrealistic speed values
- Reduce point spacing to 50ft apart
- Export for further processing

"""

#imports
import pandas as pd
import geopandas as gpd
import datetime
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

#export filepath
export_fp = Path.home() / 'Documents/BikewaySimData/Projects/gdot/gps_traces'

#filepaths for traces
coords_fps = Path.home() / 'Documents/ridership_data/CycleAtlantaClean/9-10-16 Trip Lines and Data/raw data'
coords_fps = coords_fps.glob('coord*.csv')

#coordinate reference system to project to
project_crs = "epsg:2240"

#set the minimum number of points required for a trip to be considered
#300 is based on the number of points needed for a second-by-second trace of five minutes
point_threshold = 5 * 60

#%%

'''
This section is for reading in the raw gps coordinate files from CycleAtlanta.
There are several coordinate files, and the some trips span across them so this
block loads them all into memory (32GB RAM on the computer this code was run on).
'''

# initialize empty dataframe and dictionary for adding cleaned data
coords = pd.DataFrame()

#add each coords csv to all coords dataframe (only possible if lots of RAM)
# NOTE some trips span across the different CSV files so retain last trip ids somewhere if you have to read
# in one at a time
for coords_fp in coords_fps:
    print('Reading',coords_fp.name)
    one_coords = pd.read_csv(coords_fp,header=None)
    coords = pd.concat([coords,one_coords],ignore_index=True)

# rename columns
col_names = ['tripid','datetime','lat','lon','altitude_m','speed_kph','hAccuracy_m','vAccuracy_m']
coords.columns = col_names

# replace -1 speed with NA
coords.loc[coords['speed_kph']==-1,'speed_kph'] = np.nan

# convert speed and accuracy to imperial units (mph and ft)
coords['speed_mph'] = coords['speed_kph'] * 2.2369362920544025
coords['hAccuracy_ft'] = coords['hAccuracy_m'] * 3.28084 

# drop unneeded ones
coords.drop(columns=['altitude_m','vAccuracy_m','speed_kph'],inplace=True)

# change dates to datetime
coords['datetime'] = pd.to_datetime(coords['datetime'])

# add geometry info and turn into geodataframe
coords['geometry'] = gpd.points_from_xy(coords['lon'],coords['lat'])
coords = gpd.GeoDataFrame(coords,geometry='geometry',crs='epsg:4326')
coords.to_crs(project_crs,inplace=True)

# sort everything
coords.sort_values(['tripid','datetime'],inplace=True)

# count and print number of trips
print(f"{coords['tripid'].nunique()} initial trips found")

# drop duplicate readings
coords = coords.loc[~coords[['tripid','lat','lon','datetime']].duplicated()]

#%% remove trips less than 300 points (theoretical max for 5 minute trip with at least 1 sec sampling)

'''
Doing it this way because duration alone does not say anything about the amount of data
'''

below_5 = (coords['tripid'].value_counts() < point_threshold)
coords = coords[~coords['tripid'].isin(below_5[below_5].index.tolist())]
print(below_5.sum(),f'trips had fewer than {point_threshold} points')

#%% Create trips dataframe and remove duplicates
    
'''
These next lines aggregate the points by trip id to find the start/end time/location,
duration, numbner of points, and average hAccuracy. These are recorded as initial values to compare against
the cleaned dataset.

Then find and remove duplicate trips

''' 
    
#get start time
start_time = coords.groupby('tripid')['datetime'].min()
start_time.rename('initial_start_time',inplace=True)

#get end time
end_time = coords.groupby('tripid')['datetime'].max()
end_time.rename('initial_end_time',inplace=True)

#get duration
duration = end_time - start_time
duration.rename('initial_duration',inplace=True)

#get starting location
start_lon = coords.groupby('tripid')['datetime'].idxmin().map(coords['lon'])
start_lat = coords.groupby('tripid')['datetime'].idxmin().map(coords['lat'])
start_lon.rename('initial_start_lon',inplace=True)
start_lat.rename('initial_start_lat',inplace=True)

#get ending location
end_lon = coords.groupby('tripid')['datetime'].idxmax().map(coords['lon'])
end_lat = coords.groupby('tripid')['datetime'].idxmax().map(coords['lat'])
end_lon.rename('initial_end_lon',inplace=True)
end_lat.rename('initial_end_lat',inplace=True)

#get number of points
num_of_points = coords['tripid'].value_counts()
num_of_points.rename('initial_total_points',inplace=True)

#get average haccuracy
avg_accuracy = coords.groupby('tripid')['hAccuracy_ft'].mean()
avg_accuracy.rename('initial_avg_accuracy',inplace=True)

#turn into df
trips_df = pd.concat([start_time,end_time,duration,num_of_points,avg_accuracy],axis=1)
trips_df.reset_index(inplace=True)
trips_df.rename(columns={'index':'tripid'},inplace=True)

# find and remove duplicates
duplicates = trips_df.drop(columns=['tripid']).duplicated()
print(f"{duplicates.sum()} trips are duplicates")
duplicate_tripids = trips_df[duplicates]['tripid'].tolist()
trips_df = trips_df[-duplicates]
coords = coords[-coords['tripid'].isin(duplicate_tripids)]

#%% seperate trips and store in dictionary for further processing
coords_dict = {}
coords_dict.update({tripid : df.reset_index(drop=True) for tripid, df in coords.groupby('tripid')})

#export
export_files = (coords_dict,trips_df)

with (export_fp/'raw_coords.pkl').open('wb') as fh:
    pickle.dump(export_files,fh)