# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:09:45 2021

@author: tpassmore6

This script is for cleaning the GPS traces from CycleAtlanta. It's based on the
GPS_Clean_Route_Snapping.sql script that Dr. Aditi Misra used.

Overview
- Imports and combines all of the coords.csv files (in the future we should just work from the original sql)
- Add column names and format data
- Find duplicate trips (will have same start and end time but seperate trip ids)
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

def calculate_coordinate_metrics(coords):
    '''
    Calculates acceleration, time differences, distance differences, time elapsed, etc.
    '''
    #find acceleration from change in GPS speed
    coords.loc[:,'acceleration_ft/s**2'] = coords['speed_mph'].diff()
    
    #find time and euclidean distance between successive points
    coords.loc[:,'delta_time'] = coords['datetime'].diff() #/ datetime.timedelta(seconds=1)
    coords.loc[:,'delta_distance_ft'] = coords.distance(coords.shift(1))

    #get cumulative versions
    coords.loc[:,'traversed_distance_ft'] = coords['delta_distance_ft'].cumsum()
    coords.loc[:,'time_elapsed'] = (coords['datetime'] - coords['datetime'].min()) #/ datetime.timedelta(seconds=1)
    
    #calculate speed in mph
    feet_per_mile = 5280
    seconds_per_hour = 3600
    coords.loc[:,'calculated_speed_mph'] = (coords['delta_distance_ft'] / feet_per_mile) / ((coords['delta_time'] / datetime.timedelta(seconds=1)) * seconds_per_hour)

    #attach sequence and total points for easier GIS examination
    coords['sequence'] = range(0,coords.shape[0])

    return coords

def calculate_trip_metrics(key,coords,trips_df):
    '''
    Updates the trip dataframe using the trace data
    '''
    
    # total points and duration
    trips_df.at[trips_df['tripid'] == key, 'total_points'] = coords.shape[0]
    trips_df.at[trips_df['tripid'] == key, 'duration'] = coords['datetime'].max() - coords['datetime'].min()

    # time
    #trips_df.at[trips_df['tripid'] == key, 'min_delta_time'] = coords['delta_time'].min()
    trips_df.at[trips_df['tripid'] == key, 'max_delta_time'] = coords['delta_time'].max()
    trips_df.at[trips_df['tripid'] == key, 'mean_delta_time'] = coords['delta_time'].mean()
    
    # distance
    #trips_df.at[trips_df['tripid'] == key, 'min_distance_ft'] = coords['delta_distance_ft'].min()
    trips_df.at[trips_df['tripid'] == key, 'max_distance_ft'] = coords['delta_distance_ft'].max()
    trips_df.at[trips_df['tripid'] == key, 'avg_distance_ft'] = coords['delta_distance_ft'].mean()
    trips_df.at[trips_df['tripid'] == key, 'total_distance_ft'] = coords['traversed_distance_ft'].max()
    
    #distance between first and last point
    first_point = coords.at[coords['datetime'].idxmin(),'geometry']
    last_point = coords.at[coords['datetime'].idxmax(),'geometry']
    trips_df.at[trips_df['tripid'] == key, 'first_to_last_ft'] = first_point.distance(last_point)

    # speed
    trips_df.at[trips_df['tripid'] == key, 'max_speed_mph'] = coords['speed_mph'].max()  
    trips_df.at[trips_df['tripid'] == key, 'min_speed_mph'] = coords['speed_mph'].min()
    trips_df.at[trips_df['tripid'] == key, 'avg_speed_mph'] = coords['speed_mph'].mean()
    
    return trips_df

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

#%% speed deviation, filter to study area, and trim out trip chains

remove_list = []
itp = gpd.read_file(Path.home()/'Documents/BikewaySimData/Data/Study Areas/itp.gpkg')

speed_deviation_max_mph = 47
pause_threshold_min = 10

for tripid, coords in tqdm(coords_dict.items()):
    '''
    This loop searches for unrealistic jumps between gps points
    and removes these points until there are no more jumps.

    It first looks at the first point in the data to see if there
    is a jump after it. This helps remove points that were recorded before
    the GPS had been triangulated.
    
    Then it looks at subsequent points and removes them if the threshold is
    exceeded.
    
    All the while, trips are removed if this process eliminates to many points
    '''
    #calculate metrics for the coordinates dataframe
    coords = calculate_coordinate_metrics(coords)
    # Calculate summary statistics for trips_df #
    trips_df = calculate_trip_metrics(tripid,coords,trips_df)

    #if the trip drops below the point_threshold, add it to the remove list (repeats throughout)
    if coords.shape[0] < point_threshold:
        remove_list.append(tripid)
        continue
    
    # if second value has speed above speed_deviation_max_mph remove the first point until no jump
    while coords.iloc[1]['speed_mph'] >= speed_deviation_max_mph:
        coords = coords.iloc[1:,:]
        if coords.shape[0] < 3:
            remove_list.append(tripid)
            break
        #recalculate metrics
        coords = calculate_coordinate_metrics(coords)
    
    if coords.shape[0] < point_threshold:
        remove_list.append(tripid)
        continue

    # remove point if speed_deviation_max_mph between it and previous point
    while (coords['speed_mph']>=speed_deviation_max_mph).sum() > 0:
        # only keep if below speed_deviation_max_mph mph or it's the first row and has na as the value
        coords = coords[(coords['delta_distance_ft']<speed_deviation_max_mph) | (coords['delta_distance_ft'].isna())]
        # make sure there are enough points
        if coords.shape[0] < point_threshold:
            remove_list.append(tripid)
            break
        #recalculate metrics
        coords = calculate_coordinate_metrics(coords)

    if coords.shape[0] < point_threshold:
        remove_list.append(tripid)
        continue

    '''
    Only keep trips that are entirely inside the perimeter (within Interstate 285) as
    some trips were recorded out of country or in different states.
    
    We do this now incase there were bad points that were outside study area that got removed in this step
    '''
    within = coords.clip(itp)
    if within.shape[0] < coords.shape[0]:
        remove_list.append(tripid)
        continue

    '''
    Identifies GPS trip chains and trim trips

    Some trips will have gaps between point recordings. When the gap is large, it
    indicates that the person may have forgotten to stop their recording,
    stopped at a destination along the way (trip chaining), or that there was some error.

    This algorithm checks for cases in which the person probably forgot to stop
    the recording and trims the excess points.

    If a chain was detected, then only the first leg of the trip is retained.
    There are only a small number of these.

    Step 0: Set the pause threshold
    Step 1: For each trip, count the number of points that exceed this threshold
    Step 2: For the first pause detected, trim trip if all subsequent points are
    within pause threshold biking distance (using avg speed of 8 mph)
    Step 3: If not all points are contained, then consider these points to be part of a new trip
    chain. Repeat step 1 until all legs of trip chains are identfied.
    Step 4: Remove trip chains from databases and export them for later examination.

    '''

    # distance travelled assuming 8 mph
    # mins * hr/mins * mile/hr * ft/mile = ft
    # divided by two to be conservative with street network
    travel_dist =  pause_threshold_min * (1/60) * 8 * 5280 / 2
    
    # reset index to ensure subsequent numbers
    coords.reset_index(drop=True,inplace=True)    

    #while (coords['delta_time'] > datetime.timedelta(minutes=pause_threshold_min)).any():
    if (coords['delta_time'] > datetime.timedelta(minutes=pause_threshold_min)).any():
        
        #find first pause
        first_pause = (coords['delta_time'] > datetime.timedelta(minutes=pause_threshold_min)).idxmax()
        
        #trim trip
        coords = coords.loc[0:first_pause]        
        
    coords = calculate_coordinate_metrics(coords)

#remove these trips
for key in remove_list:
    try:
        coords_dict.pop(key)
    except:
        continue
trips_df = trips_df[~trips_df['tripid'].isin(remove_list)]
print(len(remove_list),'trips removed')

#%% export coords to database

import sqlite3

# Connect to SQLite database
conn = sqlite3.connect(export_fp/'cycleatlanta.db')
cursor = conn.cursor()

# Create table
cursor.execute('''CREATE TABLE IF NOT EXISTS coordinates (
                    tripid INTEGER PRIMARY KEY,
                    datetime TEXT,
                    lat TEXT,
                    lon TEXT,
                    hAccuracy_ft,
                    speed_mph,
                    geometry,
                    

               
                'tripid', 'datetime', 'lat', 'lon', 'hAccuracy_m', 'speed_mph',
       'hAccuracy_ft', 'geometry', 'acceleration_ft/s**2', 'delta_time',
       'delta_distance_ft', 'traversed_distance_ft', 'time_elapsed',
       'calculated_speed_mph', 'sequence'
                    
                )''')


#export
export_files = (coords_dict,trips_df)

with (export_fp/'raw_coords.pkl').open('wb') as fh:
    pickle.dump(export_files,fh)