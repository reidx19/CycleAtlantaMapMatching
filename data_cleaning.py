# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:09:45 2021

@author: tpassmore6

This script is for processing and snapping the GPS traces from CycleAtlanta. It's based on the
GPS_Clean_Route_Snapping.sql script that Dr. Aditi Misra used. However, some additional steps
have been added.

Overview
- Imports and combines all of the coords.csv files
- Add column names and format data
- Find duplicate trips
- Remove all trips with points outside the NAD83 UTM18 (west Georgia) bounds
- Remove all points with a low accuracy reading (need to figure out how to incorporate this with kalman filter)
- Remove all trips less than 5 minutes long
- Remove all points that are more than three hours past the starting time
- Use Kalman filter to update coordinates and fill in missing timestamps (if needed)
- Search for unrealistic speed values
- Reduce points to 50ft apart
- Export for snapping

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


# For RDP algo:
# import rdp
# import contextily as cx
# import matplotlib.pyplot as plt

# For kalman filter:
# import numpy as np
# import numpy.ma as ma
# from pykalman import KalmanFilter

#export filepath
export_fp = Path.home() / 'Downloads/cleaned_trips'

#filepaths for traces
coords_fps = Path.home() / 'Documents/ridership_data/CycleAtlantaClean/9-10-16 Trip Lines and Data/raw data'
coords_fps = coords_fps.glob('coord*.csv')

#coordinate reference system to project to
project_crs = "epsg:2240"

#%%

'''
This section is for reading in the raw gps coordinate files from CycleAtlanta.
There are several coordinate files, and the some trips span across them so this
block loads them all into memory (32GB RAM on the computer this code was run on).

The column names and data types are defined, and then trips that only had one point,
or were less than five minutes were removed. Points (not trips) that were outside the
NAD83 UTM West Georgia extents were removed. Lastly, a duplicate check was conducted to
find trips that had different tripids but the same start time/location, end time/location,
and number of GPS points.

The gps traces are then split into different dataframes and contained in a dictionary with
the tripids as the keys. In addition, the trip data is summurized into one dataframe that
contains start/end time and the trip id for referencing back to later.

The results are printed to a log file.

'''

#write start time to log
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# initialize empty dataframe and dictionary for adding cleaned data
coords = pd.DataFrame()
coords_dict = {}

#number of initial trips
starting_trips = 0
one_point = 0
five_mins = 0
outside = 0
three_hrs = 0

#add each coords csv to all coords dataframe (only possible if lots of RAM)
# NOTE some trips span across the different CSV files
for coords_fp in coords_fps:
    print(f'Importing {coords_fp.parts[-1]}')
    one_coords = pd.read_csv(coords_fp,header=None)
    coords = pd.concat([coords,one_coords],ignore_index=True)

# rename columns
col_names = ['tripid','datetime','lat','lon','altitude','speed','hAccuracy','vAccuracy']
coords.columns = col_names
    
# drop unneeded ones
coords.drop(columns=['altitude','vAccuracy'],inplace=True)
# convert speed and accuracy to imperial units (mph and ft)
coords['speed'] = coords['speed'] * 2.2369362920544025
coords['hAccuracy'] = coords['hAccuracy'] * 3.28084 

# change dates to datetime
coords['datetime'] = pd.to_datetime(coords['datetime'])

# add geometry info
coords['geometry'] = gpd.points_from_xy(coords['lon'],coords['lat'])
# turn into geodataframe
coords = gpd.GeoDataFrame(coords,geometry='geometry',crs=4326)

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
avg_accuracy = coords.groupby('tripid')['hAccuracy'].mean()
avg_accuracy.rename('avg_accuracy',inplace=True)

#turn into df
trips_df = pd.concat([start_time,end_time,duration,start_lon,start_lat,end_lon,end_lat,num_of_points,avg_accuracy],axis=1)
trips_df.reset_index(inplace=True)
trips_df.rename(columns={'index':'tripid'},inplace=True)

#make status column
trips_df['status'] = 'retain'

#%% GPS cleaning - drop trips

### find duplicates ###
#only keep first appearance of a trip
duplicates = trips_df.drop(columns=['tripid']).duplicated()
trips_df.loc[duplicates,'status'] = 'dropped - duplicate trip'
print(f"{duplicates.sum()} trips are duplicates")
duplicate_tripids = trips_df[duplicates]['tripid'].tolist()
coords = coords[-coords['tripid'].isin(duplicate_tripids)]

### Remove trip if they have points outside NAD83 UTM 18 (west Georgia) ###
check1 = (coords.geometry.x > -85.0200) & (coords.geometry.x < -83.0000)
check2 = (coords.geometry.y > 30.6200) & (coords.geometry.y < 35.0000)
outside = coords[(check1 & check2)==False]['tripid'].drop_duplicates().tolist()
trips_df.loc[trips_df['tripid'].isin(outside),'status'] = 'dropped - outside study area'
coords = coords[check1 & check2]

#Project Data
coords.to_crs(project_crs,inplace=True)

### drop any trip that's less than five minutes ###
start = coords.groupby('tripid')['datetime'].min()
end = coords.groupby('tripid')['datetime'].max()
trip_length = end - start
five_mins = trip_length[trip_length < datetime.timedelta(minutes=5)].index.tolist()
trips_df.at[trips_df['tripid'].isin(five_mins),'status'] = 'dropped - less than 5 mins'
coords = coords[-coords['tripid'].isin(five_mins)]

#%% GPS cleaning - drop points

### remove points that occur three hours after initial point ###
#find time from start
coords['time_from_start'] = coords.groupby('tripid')['datetime'].transform(lambda x: x - x.min())
less_than_3 = coords['time_from_start'] < datetime.timedelta(hours=3)
coords = coords[less_than_3]
    
### hAccuracy ###
'''
In this step we remove points if the haccracy value is more than 2.5 standard deviations above the mean value.
'''
hAccuracy_filt = coords.groupby('tripid')['hAccuracy'].transform(lambda x: (x - x.mean()) > (x.std() * 2.5))
coords = coords[-hAccuracy_filt]


#%% seperate trips and store in dictionary for further processing
coords_dict.update({tripid : df.reset_index(drop=True) for tripid, df in coords.groupby('tripid')})

#%% pause detection

'''
8/15/23: Come back to this code later, it does not feel neccessary to get the matching process started
but it probably should be implemented
'''

# remove = []
# gaps = {}
# trips_df['split_id'] = 0

# for key, item in tqdm(coords_dict.items()):
#     #trips with a gap greater than 15 minutes
#     #naming convention is tripid_0, tripid_1, etc.
#     '''
#     If the app was paused for more than 15 minutes, we'll split that trip into segments

#     steps
#     1: find points that are recorded 15 minutes or more after previous point

#     2: split trip up and remove original trace from dict
    
#     3: update trips_df with original and new rows representing the new ones

#     '''
    
#     item['split_tripid'] = None
    
#     ### Pauses ###
#     pause = item['datetime'].diff() > datetime.timedelta(minutes=15)
    
#     if pause.sum() > 0:
        
#         #update the trips df
#         trips_df.loc[trips_df['tripid']==key,'status'] = f'retain - split into {pause.sum()-1} trips'   
        
#         #get list of the positions with a large pause
#         indices = pause[pause].index.tolist()
#         indices = [0] + indices + [item.shape[0]-1]
        
#         #apply names
#         for i in range(0,len(indices)-1):
#             item.loc[indices[i]:indices[i+1],'split_tripid'] = i
                
#         #populate new dict and add to trips_df
#         for split_tripid in item['split_tripid'].unique():
#             gaps[(key,split_tripid)] = item[item['split_tripid']==split_tripid]
#             trip_df = {
#                 'tripid':key,
#                 'split_id':split_tripid,
#                 '':,
#                 '':
#                 }
            
#         #add old trip id to remove list
#         remove.append(key)

# #remove these trips
# for key in remove:
#     coords_dict.pop(key)

# #add the split ones
# coords_dict.update(gaps)

# use for finding how many trips have long pauses
# for y in range(1,21):
#     print((coords[coords.groupby('tripid')['datetime'].diff() > datetime.timedelta(minutes=y)]['tripid'].nunique()))


#%% iterate through trip dictionary (not sure how to do with just groupby)

remove = []
print('Speed Deviation')
for key, item in tqdm(coords_dict.items()):
    
    if isinstance(key,tuple):
        split_id = key[1]
        tripid = key[0]
    else:
        tripid = key
    
    ### Deviation ###
    '''
    In this step we look at the speed between successive points to decide if any points are implausible
    (Aditi used speed >= 47 mph). These points are removed and the new speeds are recalculated. This process
    is repeated until no implausible points remain. Function will print out how many points were removed.
    
    Note that if first couple of gps points were way off, then this function will remove most points. However,
    the previous steps should have minimized this risk.
    '''
    
    #calculate distance difference and time difference from previous point
    item['distance_diff'] = item.distance(item.shift(1))
    item['time_diff'] = item['datetime'].diff()
    
    #turn to total seconds
    item['time_diff'] = item['time_diff'].apply(lambda x: x.total_seconds())
    
    #divide the difference in feet by seconds and convert to miles per hour
    item['deviation'] = item['distance_diff'] / item['time_diff'] / 5280 * 60 * 60
    
    #repeat
    while (item['deviation']>=47).sum() > 0:
        
        # only keep if below 47 mph or it's the first row and has na as the value
        item = item[(item['deviation']<47) | (item['deviation'].isna())]
        
        #calculate distance difference and time difference from previous point
        item['distance_diff'] = item.distance(item.shift(1))
        item['time_diff'] = item['datetime'].diff()
    
        #turn to total seconds
        item['time_diff'] = item['time_diff'].apply(lambda x: x.total_seconds())
        
        #divide the difference in feet by the distance in seconds (feet per second)
        item['deviation'] = item['distance_diff'] / item['time_diff'] / 5280 * 60 * 60
    
    #drop calculated columns?
    item.drop(columns=['distance_diff','time_diff','deviation'],inplace=True)
    
    ### Mask start and end location by 500ft ###
    #calculate distance difference and time difference from previous point
    item['distance_from_prev'] = item.distance(item.shift(1))
    #make first a 0
    item.iat[0,-1] = 0
    #find cumulative distance
    item['cumulative_dist'] = item['distance_from_prev'].cumsum()
    #find first 500ft
    first_500 = item['cumulative_dist'] < 500
    #find last 500ft
    last_500 = (item.iat[-1,-1] - item['cumulative_dist']) < 500
    #filter
    item = item[(first_500 | last_500) == False]

    #remove if 1 or fewer points
    if item.shape[0] < 2:
        remove.append(key)
        trips_df.at[trips_df['tripid']==tripid,'status'] = 'dropped - 1 or fewer points'
        continue
    
    #attach sequence and total points for easier GIS examination
    item['sequence'] = range(0,item.shape[0])
    item['tot_points'] = item.shape[0]
    
    #update dict
    coords_dict[key] = item


#remove trips with 1 or fewer points
for rem in remove:
    coords_dict.pop(rem)


#%% run rdp algo (removes too many points)

# =============================================================================
# simp_dict = {}
# 
# for key, item in coords_dict.items():
# 
#     tolerance_ft = 5
#         
#     #turn all trips into lines
#     line = LineString(item.sort_values('datetime')['geometry'].tolist())
#     
#     #simplify using douglass-peucker algo 
#     line = line.simplify(tolerance_ft, preserve_topology = False)
#     
#     #create dataframe
#     df = pd.DataFrame({'x':line.coords.xy[0],'y':line.coords.xy[1]})
#     
#     #create tuple for merging on
#     df['geometry_tup'] = [(xy) for xy in zip(df.x,df.y)]
#     item['geometry_tup'] = [(xy) for xy in zip(item.geometry.x,item.geometry.y)]
#     
#     #merge
#     simplified_traces = pd.merge(item,df,on='geometry_tup')
#     
#     #drop the tuple
#     simplified_traces.drop(columns=['geometry_tup'],inplace=True)
#         
#     #export
#     simp_dict[key] = simplified_traces
# =============================================================================


#%% spacing

#take cumulative distance from original
#find when value exceeds 50ft and record it

simp_dict = {}

print('Reducing number of points')
for key, item in tqdm(coords_dict.items()):
    '''
    Drop points that have a speed less than 2 mph to clean up the traces
    '''
    
    item = item[item['speed'].abs() >2]
    
    #if that removes too many values then drop trip
    if item.shape[0] == 0:
        trips_df.at[trips_df['tripid']==key,'status'] = 'dropped - speed values too low'
        continue
    
    #recalculate distances
    item['distance_from_prev'] = item.distance(item.shift(1))
    
    #get cumdist
    item['cumulative_dist'] = item['distance_from_prev'].cumsum()
    
    spacing_ft = 50
    current_spacing = spacing_ft
    
    #start with first point
    keep = [item.index.tolist()[0]]
    
    for index, value in item['cumulative_dist'].items():
        if value > current_spacing:
            keep.append(index)
            current_spacing = value + spacing_ft
    
    #count number of points dropped
    trips_df.at[trips_df['tripid']==tripid,'final_tot_points'] = item.shape[0]
    
    #add to simp dict for export
    simp_dict[key] = item.loc[keep]


#%% export step

#export to final 
with (export_fp/'coords_dict.pkl').open('wb') as fh:
    pickle.dump(simp_dict,fh)

#initialize a matching column
trips_df['matching status'] = 'unmatched'
trips_df['match_ratio'] = np.nan

#export trip_df
trips_df.to_csv(export_fp/'trips.csv',index=False)

# #%% deprecated below


# #not sure what this was for
# # for x in random_trips:
    
# #     #subset dataframes
# #     sub_coords_max = coords_max[coords_max['tripid']==x].geometry.item()
# #     sub_coords_min = coords_min[coords_min['tripid']==x].geometry.item()
   
# #     #include these coordinates
# #     include_these = sample_coords.apply(lambda row: True if (row['tripid'] == x) & ((row['geometry'].distance(sub_coords_min) <= 500) | (row['geometry'].distance(sub_coords_max) <= 500)) else False, axis = 1)
    
# #     #set values to true
# #     sample_coords.loc[include_these,'include'] = False

# #doesn't like work related for some reason?
# #user_info['trip_type'] = user_info['trip_type'].str.replace('Work-related','workrelated')
# #user_info['trip_type'] = user_info['trip_type'].str.replace('Work-Related','workrelated')
# #user_info['tripid'].isin(points['tripid'])
# #unique_users = points['tripid'].drop_duplicates()
# #user_info = user_info[-(user_info['trip_type'] == 'workrelated')]





# #%% turn gps traces into linestrings


# def turn_into_linestring(points,name):
#     #turn all trips into lines
#     lines = points.sort_values(by=['tripid','datetime']).groupby('tripid')['geometry'].apply(lambda x: LineString(x.tolist()))

#     #get start time
#     start_time = points.groupby('tripid')['datetime'].min()

#     #get end time
#     end_time = points.groupby('tripid')['datetime'].max()
    
#     #turn into gdf
#     linestrings = gpd.GeoDataFrame({'start_time':start_time,'end_time':end_time,'geometry':lines}, geometry='geometry',crs=project_crs)
    
#     #write to file
#     linestrings.to_file(rf'C:\Users\tpassmore6\Documents\GitHub\ridership_data_analysis\{name}.geojson', driver = 'GeoJSON')

#     return linestrings

# all_lines = turn_into_linestring(sample_coords, 'all')
# sample_lines = turn_into_linestring(sample_coords_cleaned, 'cleaned')


# #%% some cleaning

# #begining
# all_coords['datetime'].min()
# #end
# all_coords['datetime'].max()
# # Data goes from 01-01-2010 to 06-09-2016

# #put month in new column
# all_coords['monthyear'] = pd.DatetimeIndex(all_coords['datetime']).to_period('M')

# #%% sum stats

# #get number of trips
# all_coords['tripid'].nunique()
# #28,224

# #get number of users
# all_coords['userid'].nunique()
# #1,626

# #average number of points per trip
# points_per_trip = all_coords.groupby('tripid').size()
# points_per_trip.plot(kind="hist")
# points_per_trip.size()
# #about 2,076

# #distribution of dates
# all_coords['monthyear'].plot(kind="hist")

# #summary stats
# all_coords.describe()

# #%% read gdf
# all_trips_gdf = gpd.read_file(r'C:\Users\tpassmore6\Documents\GitHub\ridership_data_analysis\test.geojson')

# #project
# all_trips_gdf.to_crs(epsg=2240,inplace=True)

# #trip distance
# all_trips_gdf['length_mi'] = all_trips_gdf.length / 5280

# #describe trip distance
# all_trips_gdf['length_mi'].describe()

# #drop anything less than 500 ft
# all_trips_gdf = all_trips_gdf[all_trips_gdf['length_mi'] >= 0.25 ]
# all_trips_gdf = all_trips_gdf[all_trips_gdf['length_mi'] < 20 ]

# #describe trip distance
# all_trips_gdf['length_mi'].describe()

# #merge
# all_trips_gdf = pd.merge(all_trips_gdf,trip,on='tripid')

# #num users
# all_trips_gdf['userid'].nunique()
# #1398

# #num trips
# all_trips_gdf['tripid'].nunique()
# #27,620


# #%% import cycleatlanta data from aditi

# clean_notes = pd.read_csv(r'C:/Users/tpassmore6/Documents/ridership_data/CycleAtlanta/Aditi Cycle Atlanta data/CATL_data/catl-data-2014-06-08/clean_notes.csv')
# clean_trips = pd.read_csv(r'C:/Users/tpassmore6/Documents/ridership_data/CycleAtlanta/Aditi Cycle Atlanta data/CATL_data/catl-data-2014-06-08/clean_trips.csv')
# clean_user = pd.read_csv(r'C:/Users/tpassmore6/Documents/ridership_data/CycleAtlanta/Aditi Cycle Atlanta data/CATL_data/catl-data-2014-06-08/clean_user.csv')

# #%% raw data chris

# #from 10/10/2012-5/2/2013
# df = pd.read_csv('C:/Users/tpassmore6/Documents/ridership_data/fromchris/CycleAtlanta/CycleAtlanta/9-10-16 Trip Lines and Data/raw data/coord.csv', header=None)

# df.sort_values(df[1], inplace= True)
# df.head()
# df.tail()

# df[1].max()
# df[1].min()


# fiona.listlayers(r'C:/Users/tpassmore6/Downloads/OTHER.gpx')

# layer = fiona.open(r'C:/Users/tpassmore6/Downloads/OTHER.gpx', layer = 'tracks')

# geom = layer[0]

# coords = geom['geometry']['coordinates'][0]

# points = [] 
# for x in coords:
#     points.append(Point(x))
    
# #df = gpd.GeoDataFrame({'geometry':points})
# #gdf.to_file(r'C:/Users/tpassmore6/Downloads/OTHER.geojson', driver = 'GeoJSON')

# #%% trip lines chris

# trip_lines = gpd.read_file(r'C:/Users/tpassmore6/Documents/ridership_data/fromchris/CycleAtlanta/CycleAtlanta/9-10-16 Trip Lines and Data/trip lines/routes_sub_lines.shp')

# #%% route points chris

# route_points = gpd.read_file(r'C:/Users/tpassmore6/Documents/ridership_data/fromchris/CycleAtlanta/CycleAtlanta/9-10-16 Trip Lines and Data/route points/routes_sub_0.shp')

# #old kalman filter code
# import numpy as np
# from pykalman import KalmanFilter


# #get rid of speed because i don't think it matters here

# #convert to metric projection
# sample.to_crs('epsg:26967',inplace=True)

# sample['x'] = sample.geometry.x
# sample['y'] = sample.geometry.y

# #convert our measurements to numpy
# measurements = sample[['x','y','speed','hAccuracy','dt']].to_numpy()

# #we have 5 states that we're tracking (use the intial values)
# #we don't need to keep track of haccuracy or speed or time, we just want latlon
# #initial_state_mean = [measurements[0,0],measurements[0,1]]
# initial_state_mean = [measurements[0,0],measurements[0,1],0,sample['hAccuracy'].median(),0]

# #we don't have a kinematic equation for explaining the expected next state (because no vector/bearing)
# #so instead we just assume a linear transition based on change in the variable

# #x_new = x + x_v*dt #since we don't know angle/bearing we have to assume position is constant
# #y_new = y + y_v*dt
# #speed_new = speed #assuming constant speed
# #haccuracy = haccuracy #assuming constant haccuracy


# transition_matrix = [[1,0,1,1,0],
#                      []]

# #these are the values we're getting from the phone
# observation_matrix = np.eye(5)

# #
# kf1 = KalmanFilter(transition_matrices = transition_matrix,
#                   observation_matrices = observation_matrix,
#                   initial_state_mean = initial_state_mean,
#                   )

# #
# kf1 = kf1.em(measurements, n_iter=5)
# (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

# #second run adds observation covariance matrix that smooths out the data even more, increase the multiplication factor to increase smoothing
# kf2 = KalmanFilter(transition_matrices = transition_matrix,
#                   observation_matrices = observation_matrix,
#                   initial_state_mean = initial_state_mean,
#                   observation_covariance = 50*kf1.observation_covariance,
#                   em_vars=['transition_covariance', 'initial_state_covariance'])

# kf2 = kf2.em(measurements, n_iter=5)
# (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)