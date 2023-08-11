# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:09:45 2021

@author: tpassmore6

This script is for processing and snapping the GPS traces from CycleAtlanta. It's based on the
GPS_Clean_Route_Snapping.sql script that Dr. Aditi Misra used. However, some additional steps
have been added.

Overview
- Import one of the coords.csv files
- Add column names and format data
- Remove all points outside the NAD83 UTM18 (west Georgia) bounds
- Remove all points with a low accuracy reading (need to figure out how to incorporate this with kalman filter)
- Remove all trips less than 5 minutes long
- Remove all points that are more than three hours past the starting time
- Use Kalman filter to update coordinates and fill in missing timestamps
- Search for unrealistic speed values
- Use RDP algorithim to simplify lines
- Reduce points to 10m apart but keep all points kept in the RDP algorithm
- Export for snapping

"""

#imports
import pandas as pd
import geopandas as gpd
import fiona
import glob
from shapely.geometry import shape, Point, LineString, MultiPoint, box
import datetime


from pathlib import Path

from tqdm import tqdm
import rdp
import contextily as cx
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import pickle

import numpy as np
import numpy.ma as ma
from pykalman import KalmanFilter

def log_print(log_entry,exportfp):    
    '''
    Function used for creating a quick log file to keep track of progress. Opens
    a text file and adds the "log_entry" and a new line
    '''
    with (exportfp/"output_log.txt").open("a") as file:
        file.write(log_entry + "\n")
    print(log_entry)

def turn_to_line(trips):
    #turn into gdf
    if turn_to_line:
        trips['geometry'] = points.sort_values(by=['tripid','datetime']).groupby('tripid')['geometry'].apply(lambda x: LineString(x.tolist()))
        trips = gpd.GeoDataFrame(trips,geometry='geometry',crs='epsg:4326')

#export filepath
export_fp = Path.home() / 'Downloads/cleaned_trips'


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
log_print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),export_fp)

#filepaths for traces
coords_fps = Path.home() / 'Documents/ridership_data/CycleAtlantaClean/9-10-16 Trip Lines and Data/raw data'
coords_fps = coords_fps.glob('coord*.csv')

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
log_print(f"{coords['tripid'].nunique()} initial trips found",export_fp)

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
num_of_points.rename('num_of_points',inplace=True)

#get average haccuracy
avg_accuracy = coords.groupby('tripid')['hAccuracy'].mean()
avg_accuracy.rename('avg_accuracy',inplace=True)

#turn into df
trips_df = pd.concat([start_time,end_time,start_lon,start_lat,end_lon,end_lat,num_of_points,avg_accuracy],axis=1)
trips_df.reset_index(inplace=True)
trips_df.rename(columns={'index':'tripid'},inplace=True)

#make status column
trips_df['status'] = 'keep'

#%% GPS cleaning

### find duplicates ###
#only keep first appearance of a trip
duplicates = trips_df.drop(columns=['tripid']).duplicated()
trips_df.loc[duplicates,'status'] = 'duplicate trip'
log_print(f"{duplicates.sum()} trips are duplicates",export_fp)
duplicate_tripids = trips_df[duplicates]['tripid'].tolist()
coords = coords[-coords['tripid'].isin(duplicate_tripids)]

### Remove trip if they have points outside NAD83 UTM 18 (west Georgia) ###
check1 = (coords.geometry.x > -85.0200) & (coords.geometry.x < -83.0000)
check2 = (coords.geometry.y > 30.6200) & (coords.geometry.y < 35.0000)
outside = coords[(check1 & check2)==False]['tripid'].drop_duplicates().tolist()
trips_df.loc[trips_df['tripid'].isin(outside),'status'] = 'outside study area'
coords = coords[check1 & check2]

#Project Data
coords.to_crs("epsg:2240",inplace=True)

### drop any trip that's less than five minutes ###
start = coords.groupby('tripid')['datetime'].min()
end = coords.groupby('tripid')['datetime'].max()
trip_length = end - start
five_mins = trip_length[trip_length < datetime.timedelta(minutes=5)].index.tolist()
trips_df.at[trips_df['tripid'].isin(five_mins),'status'] = 'less than 5 mins'
coords = coords[-coords['tripid'].isin(five_mins)]

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

### Drop idle points ###
'''
Drop points that have a speed less than 2 mph to clean up the traces
'''
coords = coords[coords['speed']>2]

# seperate trips and store in dictionary for further processing
coords_dict.update({tripid : df.reset_index(drop=True) for tripid, df in coords.groupby('tripid')})


#%% pause detection

# remove = []

# for key, item in tqdm(coords_dict.items()):
#     #trips with a gap greater than 5 minutes
#     #naming convention is tripid_0, tripid_1, etc.
#     '''
#     If the app was paused for more than 5 minutes, we'll split that trip into segments

#     steps
#     1: find points that are recorded 5 minutes or more after previous point

#     for every point before this one, change the trip id to _0 and for every point after until
#     next 5 minute gap make it trip_id _1, _2, etc


#     '''
    
    
#     item['split_tripid'] = None
    
#     ### Pauses ###
#     pause = item['datetime'].diff() > datetime.timedelta(minutes=15)
#     if pause.sum() > 0:
        
#         # ran this to see how many trips had long pauses
#         # for y in range(1,21):
#         #     x.append((coords[coords.groupby('tripid')['datetime'].diff() > datetime.timedelta(minutes=y)]['tripid'].nunique()))
        
#         #get list of the positions with a large pause
#         indices = pause[pause].index.tolist()
#         indices = [0] + indices + [item.shape[0]-1]
        
#         #apply names
#         for i in range(0,len(indices)-1):
#             item.loc[indices[i]:indices[i+1],'split_tripid'] = i
            
#         #populate new dict
#         for split_tripid in item['split_tripid'].unique():
#             coords_dict[(key,split_tripid)] = item[item['split_tripid']==split_tripid]
            
#         #add old trip id to remove list
#         remove.append(key)
   
# #remove these trips
# for key in remove:
#     coords_dict.pop(key)



#%% iterate through trip dictionary (not sure how to do with just groupby)

remove = []

for tripid, item in tqdm(coords_dict.items()):
    
    if isinstance(tripid,tuple):
        split_id = tripid[1]
        tripid = tripid[0]
    
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
        trips_df.at[trips_df['tripid']==tripid,'status'] = '1 or fewer points'
        continue
    
    #attach sequence and total points for easier GIS examination
    item['sequence'] = range(0,item.shape[0])
    item['tot_points'] = item.shape[0]
    
    #update dict
    coords_dict[tripid] = item
    #count number of points dropped
    trips_df.at[trips_df['tripid']==tripid,'post_filter'] = item.shape[0]

#remove trips with 1 or fewer points
for rem in remove:
    coords_dict.pop(rem)

#%%

#export all_coords
with (export_fp/'coords_dict.pkl').open('wb') as fh:
    pickle.dump(coords_dict,fh)

#export trip_df
trips_df.to_csv(export_fp/'trips.csv',index=False)

#%% import cleaned data

all_trips = pd.read_csv(export_fp/'trips.csv')

with (export_fp/'coords_dict.pkl').open('rb') as fh:
    coords_dict = pickle.load(fh)


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

simp_dict = {}

for key, item in coords_dict.items():
    print('Spacing...')
    spacing_ft = 50
    
    #start with first point
    keep = item.iloc[[0],:]
    
    #recalculate distances
    item['distance_from_prev'] = item.distance(item.shift(1))
    
    #get cumdist
    item['cumulative_dist'] = item['distance_from_prev'].cumsum()
    
    #keep looping until there are no points at or above the spacing threshold
    while item['cumulative_dist'] >= spacing_ft:
                
        try:
            #find one above 50 and add prev if possible (otherwise add first above 50 if prev value the same as starting)
            keep_this = item.loc[[item[(spacing_ft - item['cumulative_dist']) >= 0]['cumulative_dist'].idxmin()],:]
        except:
            #if there is no point within tolerance than select the max under 0
            keep_this = item.loc[[item[(spacing_ft - item['cumulative_dist']) < 0]['cumulative_dist'].idxmax()],:]
        
        #add point to keep
        keep = pd.concat([keep,keep_this],ignore_index=True)
        
        #remove rest (keep last kept point)
        item = item[item['sequence'] >= keep_this['sequence'].item()]
    
        #recalculate distances
        item['distance_from_prev'] = item.distance(item.shift(1))
        
        #get cumdist
        item['cumulative_dist'] = item['distance_from_prev'].cumsum()
    
    #drop extra columns
    keep.drop(columns=['distance_from_prev','cumulative_dist'])
    
    #add to simp dict for export
    simp_dict[key] = keep
    
#%%

orig = coords_dict[31800]
item = coords_dict[31800]



spacing_ft = 50

#start with first point
keep = item.iloc[[0],:]

#recalculate distances
item['distance_from_prev'] = item.distance(item.shift(1))

#get cumdist
item['cumulative_dist'] = item['distance_from_prev'].cumsum()

#keep looping until there are no points at or above the spacing threshold
while (item['cumulative_dist'] >= spacing_ft).any():
            
    #try:
        #find one above 50 and add prev if possible (otherwise add first above 50 if prev value the same as starting)
    keep_this = item.loc[[item[(spacing_ft - item['cumulative_dist']) >= 0]['cumulative_dist'].idxmin()],:]
# =============================================================================
#     except:
#         #if there is no point within tolerance than select the max under 0
#         keep_this = item.loc[[item[(spacing_ft - item['cumulative_dist']) < 0]['cumulative_dist'].idxmax()],:]
# =============================================================================
    
    #add point to keep
    keep = pd.concat([keep,keep_this],ignore_index=True)
    
    #remove rest (keep last kept point)
    item = item[item['sequence'] >= keep_this['sequence'].item()]

    #recalculate distances
    item['distance_from_prev'] = item.distance(item.shift(1))
    
    #get cumdist
    item['cumulative_dist'] = item['distance_from_prev'].cumsum()

#drop extra columns
keep.drop(columns=['distance_from_prev','cumulative_dist'])


#%%



#export to final 
with (export_fp/'simp_dict.pkl').open('wb') as fh:
    pickle.dump(simp_dict,fh)



#%%

def kalman_test(points,tripid,export_fp):


    '''
    Use kinematics (assume constant velocity) to smooth out data. Points with low accuracy (high hAccuracy) are
    removed, but there should be a way to account for this.
    
    Adapted code from this:
    https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data
    
    Read this for more about kalman filters: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
    
    '''   
      
    #make sure data is sorted
    points = points.sort_values('datetime')
    
    #export raw data
    #points.to_file(export_fp/f"selected_trips/{tripid}.gpkg",layer='haccuracy_and_deviation')
    
    #make a time elapsed in seconds column
    points['time_elapsed'] = points['datetime'].apply(lambda x: int((x - points['datetime'].min()).total_seconds()))
        
    #project to get x and y values
    #points.to_crs(crs,inplace=True)
    
    points['x'] = points.geometry.x
    points['y'] = points.geometry.y
    
    #create nan entries to fill in missing data
    fill = pd.DataFrame(data={'time_elapsed':range(0,points['time_elapsed'].max()+1)})
    fill = pd.merge(fill,points,on='time_elapsed',how='left')
    
    #convert our observations to numpy array
    observations = fill[['x','y']].to_numpy()
    
    #errors
    #errors = fill[['hAccuracy']].to_numpy()
    #errors = np.concatenate([errors,errors],axis=1)
    
    #use np.ma to mask missing data
    observations = ma.masked_array(observations , mask=np.isnan(observations))
    #rrors = ma.masked_array(errors, mask=np.isnan(errors))
    
    # the initial state of the cyclist (assuming starting from stop)
    # so initial position in x and y
    # assume 0 velocity (acceleration is an unaacounted for external influence)
    initial_state_mean = [observations[0,0],0,observations[0,0],0]
    
    #these are the kinematics of how we're moving (ignoring road network and grade)
    #assume that velocity is constant (not true) but we don't have acceleration information
    transition_matrix = [[1,1,0,0],
                          [0,1,0,0],
                          [0,0,1,1],
                          [0,0,0,1]]
    
    #these are the values we're getting from the phone (just x and y position, we don't know direction)
    #QUESTION how do we include the other measurements that we're taking? we know hAccuracy is influecing the position and speed is related to speed
    #also how to account for time skips?
    observation_matrix = [[1,0,0,0],
                          [0,0,1,0]]
        
    
    #how confident are we in our measurements (assume x and y don't affect each other)
    observation_covariance = fill['hAccuracy'].mean()**2 * np.array([[1,0],[0,1]])
    
    #using just this estimate a kalman filter
    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean,
                      observation_covariance= observation_covariance
                      #observation_offsets=errors
                      )
    
    #get new values
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(observations)
    
    #try smoothing again without those points
    
    #convert back to dataframe
    filtered = pd.DataFrame(smoothed_state_means,columns=['x','v_x','y','v_y'])
    
    #create an estimated speed column (mph)
    filtered['speed'] = (filtered['v_x']**2 + filtered['v_y']**2)**0.5
    
    #remove points with excessively high speed 47 mph (should probably do this recursively?)
    #filtered = filtered[filtered['speed']<47]
    
    #and remove excesively low speed
    #filtered = filtered[filtered['speed']>2]
    
    #reset index and rename to elapsed time
    filtered.reset_index(inplace=True)
    filtered.rename(columns={'index':'time_elapsed'},inplace=True)
    
    #merge with points to get all the rest of the data
    smoothed_df = pd.merge(filtered,points,on='time_elapsed',how='left',suffixes=('_est',None))
    
    #make geodataframe
    smoothed_df['geometry'] = gpd.points_from_xy(smoothed_df['x_est'], smoothed_df['y_est'], crs=points.crs)
    smoothed_df = gpd.GeoDataFrame(smoothed_df,geometry='geometry')
    
    #export
    smoothed_df.to_file(export_fp/f"selected_trips/{tripid}.gpkg",layer='smoothed_with_covar')



#%%


#plot if desired
#sample.plot(markersize=0.5)

for tripid, df in sample_coords.items():

    #tripid=9208
    points = df
    df.to_crs('epsg:2240',inplace=True)
    #crs='epsg:2240'
    
    # x_min, y_min, x_max, y_max = (2227925.733540652,1374549.6684689275,2228562.6084365235,1375118.3676399956)
    
    # #check if one point within bounding box
    # check1 = ((df.geometry.x < x_max) & (df.geometry.x > x_min))
    # check2 = ((df.geometry.y < y_max) & (df.geometry.y > y_min))
    
    # if (check1 & check2).any() == False:
    #     print(f'{tripid} does not have point within bbox')
    #     continue
    
    #check if average haccuracy is above 100
    if df['hAccuracy'].mean() < 100:
        print('too low haccuracy')
        continue
    
    #datetime to str
    points['datetime'] = points['datetime'].astype(str)
    points['time_from_start'] = points['datetime'].astype(str)
    
    points.to_file(export_fp/f"selected_trips/{tripid}.gpkg",layer='raw')
    
    points['datetime'] = pd.to_datetime(points['datetime'])
    points['time_from_start'] = pd.to_datetime(points['time_from_start'])
    
    #for key, coords in sample_coords.items():
    
    #make sure it's sorted
    points.sort_values('datetime')

    #use kalman filter using error in the offset one
    
    #use kalman filter using errro in the covariance
    kalman_test(points,tripid,export_fp)
    
    #use kalman filter when dropping high error values and inplausible points
    points = filter_points(points)
    kalman_smoothing(points,tripid,export_fp)
        







#%%try running
smoothed_dict={}
for key in tqdm(sample_coords.keys()):
    smoothed_dict[key] = kalman_smoothing(sample_coords[key],key,haccuracy_max=50,crs="epsg:2240",export_fp=export_fp)


#%% 

'''
In some cases, the kalman filter can result in worse results if the observations are accurate
'''
def kalman_smoothing(points,tripid,export_fp):
    '''
    Use kinematics (assume constant velocity) to smooth out data. Points with low accuracy (high hAccuracy) are
    removed, but there should be a way to account for this.
    
    Adapted code from this:
    https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data
    
    Read this for more about kalman filters: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
    
    '''   
      
    #plot if desired
    #sample.plot(markersize=0.5)
    
    #make sure data is sorted
    points = points.sort_values('datetime')
    
    #export raw data
    #points.to_file(export_fp/f"selected_trips/{tripid}.gpkg",layer='raw')
    
    # #remove points according to haccuracy
    # remove_points = points['hAccuracy']<haccuracy_max
    
    # if remove_points.sum() < 2:
    #     print(f'Too high of hAccuracy max for trip {tripid}')
    # else:
    #     points = points[points['hAccuracy']<haccuracy_max]
    
    #make a time elapsed in seconds column
    points['time_elapsed'] = points['datetime'].apply(lambda x: int((x - points['datetime'].min()).total_seconds()))
        
    #project to get x and y values
    #points.to_crs(crs,inplace=True)
    
    points['x'] = points.geometry.x
    points['y'] = points.geometry.y
    
    #create nan entries to fill in data
    fill = pd.DataFrame(data={'time_elapsed':range(0,points['time_elapsed'].max()+1)})
    fill = pd.merge(fill,points,on='time_elapsed',how='left')
    
    #convert our observations to numpy array
    observations = fill[['x','y']].to_numpy()
    
    #use np.ma to mask missing data
    observations = ma.masked_array(observations , mask=np.isnan(observations))
    
    # the initial state of the cyclist (assuming starting from stop)
    # so initial position in x and y
    # assume 0 velocity (acceleration is an unaacounted for external influence)
    initial_state_mean = [observations[0,0],0,observations[0,0],0]
    
    #these are the kinematics of how we're moving (ignoring road network and grade)
    #assume that velocity is constant (not true) but we don't have acceleration information
    transition_matrix = [[1,1,0,0],
                          [0,1,0,0],
                          [0,0,1,1],
                          [0,0,0,1]]
    
    #these are the values we're getting from the phone (just x and y position, we don't know direction)
    #QUESTION how do we include the other measurements that we're taking? we know hAccuracy is influecing the position and speed is related to speed
    #also how to account for time skips?
    observation_matrix = [[1,0,0,0],
                          [0,0,1,0]]
        
    #using just this estimate a kalman filter
    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean
                      )
    
    #figure out what em is doing (something about learning new parameters like the two covariance matrices)
    #kf1 = kf1.em(observations, n_iter=5)
    #get new values
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(observations)
    
    # #can do second filter where we multiply the observation_covariance matrix to essentially say we have less confidence in our measurements
    # #second run adds observation covariance matrix that smooths out the data even more, increase the multiplication factor to increase smoothing
    # kf2 = KalmanFilter(transition_matrices = transition_matrix,
    #                   observation_matrices = observation_matrix,
    #                   initial_state_mean = initial_state_mean,
    #                   observation_covariance = 10*kf1.observation_covariance,
    #                   em_vars=['transition_covariance', 'initial_state_covariance'])

    # kf2 = kf2.em(observations, n_iter=5)
    # (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(observations)

    # Plot original and smoothed GPS points
    # plt.figure(figsize=(8, 6))
    # plt.plot(measurements[:,0], measurements[:,1], 'bo-', label='Original')
    # plt.plot(smoothed_state_means[:,0], smoothed_state_means[:,2], 'r.-', label='Smoothed')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Comparison of Original and Smoothed GPS Points')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
     
    #try smoothing again without those points

    #convert back to dataframe
    filtered = pd.DataFrame(smoothed_state_means,columns=['x','v_x','y','v_y'])
    
    #create an estimated speed column (mph)
    filtered['speed'] = (filtered['v_x']**2 + filtered['v_y']**2)**0.5
    
    #remove points with excessively high speed 40 mph (should probably do this recursively?)
    #filtered = filtered[filtered['speed']<40]
    
    #and remove excesively low speed
    #filtered = filtered[filtered['speed']>2]

    #reset index and rename to elapsed time
    filtered.reset_index(inplace=True)
    filtered.rename(columns={'index':'time_elapsed'},inplace=True)
    
    #merge with points to get all the rest of the data
    smoothed_df = pd.merge(filtered,points,on='time_elapsed',how='left',suffixes=('_est',None))
    
    #make geodataframe
    smoothed_df['geometry'] = gpd.points_from_xy(smoothed_df['x_est'], smoothed_df['y_est'], crs=points.crs)
    smoothed_df = gpd.GeoDataFrame(smoothed_df,geometry='geometry')
    
    #export
    smoothed_df.to_file(export_fp/f"selected_trips/{tripid}.gpkg",layer='smoothed_wo_outliers')
    
    return smoothed_df

#%% look at speeds




#don't do this because kalman should be accounting for this?
#throw out high haccuracy
#trying 15 for now
# haccuracy_max = 15
# count = coords.shape[0]
# coords = coords[coords['hAccuracy']<haccuracy_max]
# print(f'{count-coords.shape[0]} have above an hAccuracy above {haccuracy_max} meters')

### Sort and add timestamps ###

#use datetime to create sequence column
#coords['sequence'] = coords.groupby(['tripid']).cumcount()

#make total points column for reference
#tot_points = coords['tripid'].value_counts().rename('tot_points')
#coords = pd.merge(coords, tot_points, left_on='tripid',right_index=True)

#sort by sequence and tripid

# Drop trips that are less than five minutes long


# need a find duplicate trips code




# move these .gpkg to a new directory

# import os
# import shutil

# def copy_files(tripid):
    
#     source_file = Path.home() / f'Downloads/smoothed_traces/{tripid}.gpkg'
#     destination_dir = Path.home() / f'Downloads/selected_traces/{tripid}.gpkg'
    
#     # Copy the file to the destination directory
#     shutil.copy2(source_file, destination_dir)
    
# [copy_files(x) for x in coords['tripid'].unique().tolist()]


    

#%%


for tripid in tqdm(coords['tripid'].unique()):
    
    #subset to that specific trip
    sample = coords[coords['tripid']==tripid]
    
    #plot if desired
    #sample.plot(markersize=0.5)
    
    #export raw data
    sample.to_file(Path.home() / f'Downloads/selected_traces/{tripid}.gpkg',layer='raw')
    
    #make a time elapsed in seconds column
    sample['time_elapsed'] = sample['datetime'].apply(lambda x: int((x - sample['datetime'].min()).total_seconds()))
    
    # Kalman filtering and smoothing
    
    '''
    
    With the given info we have (lat-lon in degrees,velocity in m/s, time in secs, and haccuracy in m), we dont have a way of adding kinematic equations to improve the fit
    so this is basically maximum likelihood estimation.
    
    Adapted code from this:
    https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data
    
    Read this for more about kalman filters: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
    
    '''
    import numpy as np
    import numpy.ma as ma
    from pykalman import KalmanFilter
    
    #convert to metric projection to match up with other units
    #sample.to_crs('epsg:26967',inplace=True)
    sample.to_crs('epsg:2240',inplace=True)
    
    sample['x'] = sample.geometry.x
    sample['y'] = sample.geometry.y
    
    #create nan entries to fill in data
    fill = pd.DataFrame(data={'time_elapsed':range(0,sample['time_elapsed'].max()+1)})
    fill = pd.merge(fill,sample,on='time_elapsed',how='left')
    
    #convert our observations to numpy array
    observations = fill[['x','y','hAccuracy','speed']].to_numpy()
    
    #use np.ma to mask missing data
    observations = ma.masked_array(observations , mask=np.isnan(observations))
    
    #only keep x and y column for observations
    observations = observations[:,0:2]
    
    # the initial state of the cyclist (assuming starting from stop)
    # so initial position in x and y
    # assume 0 velocity (acceleration is an unaacounted for external influence)
    initial_state_mean = [observations[0,0],0,observations[0,0],0]
    
    #these are the kinematics of how we're moving (ignoring road network and grade)
    #assume that velocity is constant (not true) but we don't have acceleration information
    transition_matrix = [[1,1,0,0],
                          [0,1,0,0],
                          [0,0,1,1],
                          [0,0,0,1]]
    
    #these are the values we're getting from the phone (just x and y position, we don't know direction)
    #QUESTION how do we include the other measurements that we're taking? we know hAccuracy is influecing the position and speed is related to speed
    #also how to account for time skips?
    observation_matrix = [[1,0,0,0],
                          [0,0,1,0]]
    
    #sensor noise (speed and hAccuracy)
    #
    #observation_offsets = [1,1]
    
    #using just this estimate a kalman filter
    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean
                      )
    
    #get new values
    smoothed_state_means = kf1.smooth(observations)[0]
        
    #convert back to dataframe
    filtered = pd.DataFrame(smoothed_state_means,columns=['x','v_x','y','v_y'])
    
    #create an estimated speed column (mph)
    filtered['speed'] = (filtered['v_x']**2 + filtered['v_y']**2)**0.5 * 60 * 60 / 5280
    
    #reset index and rename to elapsed time
    filtered.reset_index(inplace=True)
    filtered.rename(columns={'index':'time_elapsed'},inplace=True)
    
    #merge with sample to get all the rest of the data
    check = pd.merge(filtered,sample,on='time_elapsed',how='left',suffixes=('_est',None))
    
    #make geodataframe
    check['geometry'] = gpd.points_from_xy(check['x_est'], check['y_est'], crs='epsg:2240')
    check = gpd.GeoDataFrame(check,geometry='geometry')
    
    #export
    check.to_file(Path.home()/f'Downloads/selected_traces/{tripid}.gpkg',layer='smoothed_w_outliers')
    
    
#%%i should figure out what em does
#kf1 = kf1.em(measurements, n_iter=5)
(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

# #second run adds observation covariance matrix that smooths out the data even more, increase the multiplication factor to increase smoothing
# kf2 = KalmanFilter(transition_matrices = transition_matrix,
#                   observation_matrices = observation_matrix,
#                   initial_state_mean = initial_state_mean,
#                   observation_covariance = 10*kf1.observation_covariance,
#                   em_vars=['transition_covariance', 'initial_state_covariance'])

# kf2 = kf2.em(measurements, n_iter=5)
# (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)

# # Plot original and smoothed GPS points
# plt.figure(figsize=(8, 6))
# plt.plot(measurements[:,0], measurements[:,1], 'bo-', label='Original')
# plt.plot(smoothed_state_means[:,0], smoothed_state_means[:,2], 'r.-', label='Smoothed')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Comparison of Original and Smoothed GPS Points')
# plt.legend()
# plt.grid(True)
# plt.show()




#%%



#%%


# Create initial state
initial_state = [latitude[0], 0, longitude[0], 0]

# Create observation matrix
observation_matrix = np.array([
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0]
])

# Create transition matrices
transition_matrices = np.array([
    [1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])

# Create Kalman filter
kf = KalmanFilter(
    transition_matrices=transition_matrices,
    observation_matrices=observation_matrix,
    initial_state_mean=initial_state,
    em_vars=['transition_covariance', 'observation_covariance']
)

# Create empty arrays to store smoothed values
smoothed_latitude = np.zeros_like(latitude)
smoothed_longitude = np.zeros_like(longitude)

# Smoothing using Kalman filter
for t in range(len(latitude)):
    if t == 0:
        smoothed_state_means, smoothed_state_covariances = kf.filter_update(
            initial_state_mean=initial_state,
            observation=np.array([latitude[t], speed[t], longitude[t], datetime[t], hAccuracy[t]])
        )
    else:
        smoothed_state_means, smoothed_state_covariances = kf.filter_update(
            filtered_state_mean=smoothed_state_means,
            filtered_state_covariance=smoothed_state_covariances,
            observation=np.array([latitude[t], speed[t], longitude[t], datetime[t], hAccuracy[t]])
        )
    smoothed_latitude[t] = smoothed_state_means[0]
    smoothed_longitude[t] = smoothed_state_means[2]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(datetime, latitude, 'r-', label='Original Latitude')
plt.plot(datetime, longitude, 'b-', label='Original Longitude')
plt.plot(datetime, smoothed_latitude, 'g-', label='Smoothed Latitude')
plt.plot(datetime, smoothed_longitude, 'm-', label='Smoothed Longitude')
plt.xlabel('Datetime')
plt.ylabel('Coordinate')
plt.title('GPS Data Smoothing')
plt.legend()
plt.grid(True)
plt.show()









#%% see what happens when running multiple times
new_coords = coords.copy()

#run 20 times to see what happends
for x in range(0,19):
    new_coords = gps_deviation(new_coords)

#%%turn to lines for examining

#doesn't work don't bother

#merge data
beforeafter_coords = pd.merge(coords[['tripid','datetime','geometry']],new_coords[['tripid','datetime','geometry']],on=['tripid','datetime'],suffixes=('_before','_after'),how='left')

#make subplots
fig, (ax1,ax2) = plt.subplots(1,2)

# print to examine
for x in tqdm(beforeafter_coords['tripid'].unique()):
    #subset
    df = beforeafter_coords[beforeafter_coords['tripid']==x]
    
    #seperate?
    
    #make first plot
    df.set_geometry('geometry_before',inplace=True)
    ax1.plot(df.geometry.x,df.geometry.y)
    #cx.add_basemap(ax1, crs=df.crs)
    
    #make second plot
    df.set_geometry('geometry_after',inplace=True)
    df.dropna(inplace=True)
    ax2.plot(df.geometry.x,df.geometry.y)
    #cx.add_basemap(ax2, crs=df.crs)
    
    fig.savefig(Path.home()/f'Downloads/gps_processing/{x}.png')
    ax1.cla()
    ax2.cla()



#%%
import pickle
#pickle
with (Path.home() / 'Downloads/cleaned_coords.pkl').open(mode='wb') as fh:
    pickle.dump(clean_coords,fh)
    
#%%

with (Path.home() / 'Downloads/cleaned_coords.pkl').open(mode='rb') as fh:
    clean_coords = pickle.load(fh)

#%%Route Simplify
'''
-Use douglass peucker alogrithim with a 5 foot tolerance to drop points
-Remove all trips that are less than 5 minutes long
-Remove all points recorded more than three hours after the first
'''


clean_coords = clean_coords[-clean_coords['tripid'].isin(less_than_five)]

#greater than three hours
clean_coords


#%%

def turn_into_linestring(points):
    #turn all trips into lines
    lines = points.sort_values(by=['tripid','datetime']).groupby('tripid')['geometry'].apply(lambda x: LineString(x.tolist()))
    
    #turn timestamps into list
    timestamps = points.sort_values(by=['tripid','datetime']).groupby('tripid')['datetime'].apply(list)

    #get start time
    start_time = points.groupby('tripid')['datetime'].min()

    #get end time
    end_time = points.groupby('tripid')['datetime'].max()
    
    #turn into gdf
    linestrings = gpd.GeoDataFrame({'start_time':start_time,'end_time':end_time,'geometry':lines}, geometry='geometry',crs="epsg:4326")
    
    return linestrings

clean_coords.to_crs('epsg:2240',inplace=True)

lines = turn_into_linestring(clean_coords)



#%%

from shapely.geometry import LineString
from math import sqrt

def douglas_peucker(line, epsilon):
    if len(line.coords) < 3:
        return line.coords[:], line.metadata[:]

    dmax = 0.0
    index = 0

    for i in range(1, len(line.coords)-1):
        d = shortest_distance(line.coords[i], line.coords[0], line.coords[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        coords_left, metadata_left = douglas_peucker(LineString(line.coords[:index+1]), epsilon)
        coords_right, metadata_right = douglas_peucker(LineString(line.coords[index:]), epsilon)
        
        coords = coords_left[:-1] + coords_right
        metadata = metadata_left[:-1] + metadata_right
    else:
        coords = [line.coords[0], line.coords[-1]]
        metadata = [line.metadata[0], line.metadata[-1]]

    return coords, metadata

def shortest_distance(p, line_start, line_end):
    x1, y1 = line_start
    x2, y2 = line_end
    x0, y0 = p

    if x1 == x2 and y1 == y2:
        return sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)

    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return numerator / denominator


test = list()
for row in lines.itertuples():
    line = row[-1]
    timestamps = row[0]
    
    epsilon = 5
    
    simplified_coords, removed_metadata = douglas_peucker(line, epsilon)
    simplified_line = LineString(simplified_coords)
    removed_points = [
        (p, metadata) for p, metadata in zip(line.coords, line.metadata)
        if p not in simplified_coords
    ]
    list.append(removed_points)

#%%

# Example usage
line_coords = [(0, 0), (1, 1), (2, 2), (2, 2), (3, 1), (4, 0)]
line_metadata = ['A', 'B', 'C', 'D', 'E', 'F']
line = LineString(line_coords)
line.metadata = line_metadata
epsilon = 0.5

simplified_coords, removed_metadata = douglas_peucker(line, epsilon)
simplified_line = LineString(simplified_coords)
removed_points = [
    (coord, metadata) for coord, metadata in zip(line.coords, line.metadata)
    if coord not in simplified_coords
]

print("Removed Points:", removed_points)
print("Simplified Line:", simplified_line)



#%%go back to points

full_gdf = gpd.GeoDataFrame()

for row in lines.itertuples():
    tripid = row[0]
    #timestamps = row[-2]
    coords = list(zip(row[-1].coords))
    coords = [Point(x) for x in coords]
    #make geodataframe
    gdf = gpd.GeoDataFrame({'geometry':coords},geometry='geometry',crs='epsg:2240')
    #add trip id
    gdf['tripid'] = tripid
    #concat
    full_gdf = pd.concat([full_gdf,gdf],ignore_index=True)

#merge back other info
clean_coords.to_crs('epsg:2240',inplace=True)
clean_coords_cols = clean_coords.to_wkt()
before = clean_coords_cols.shape[0]

#convert to wkt for merging
full_gdf = full_gdf.to_wkt()

new = pd.merge(clean_coords_cols,full_gdf,on=['tripid','geometry'],how="inner")    
print(f"{before - new.shape[0]} points removed")    

new = new.from_wkt()
    
#%%





#%%


myid_list = gdf_line.index.to_list()
repeat_list = [len(line.coords) for line in gdf_line['geometry'].unary_union] #how many points in each Linestring
coords_list = [line.coords for line in gdf_line['geometry'].unary_union]

#make new gdf
gdf = gpd.GeoDataFrame(columns=['myid', 'order', 'geometry'])

for myid, repeat, coords in zip(myid_list, repeat_list, coords_list):
    index_num = gdf.shape[0]
    for i in range(repeat):
        gdf.loc[index_num+i, 'geometry'] = Point(coords[i])
        gdf.loc[index_num+i, 'myid'] = myid
        gdf.loc[index_num+i, 'order'] = i+1

#%%
gdf['order'] = range(1, 1+len(df))

#you can use groupby method
gdf.groupby('myid')['geometry'].apply(list)








#%% import trip info
trip = pd.read_csv(path+"trip.csv", header = None)
col_names = ['tripid','userid','trip_type','description','starttime','endtime','notsure']
trip.columns = col_names

# these don't seem too accurate
# #convert to datetime
# trip['starttime'] = pd.to_datetime(trip['starttime'])
# trip['endtime'] = pd.to_datetime(trip['endtime'])

# #trip time
# trip['triptime'] = trip['endtime'] - trip['starttime']

#drop these
trip.drop(columns=['description','notsure','starttime','endtime'], inplace = True)

#change tripid and userid to str
trip['tripid'] = trip['tripid'].astype(str)
trip['userid'] = trip['userid'].astype(str)

#%% import user info
user = pd.read_csv(path+"user.csv", header=None)

user_col = ['userid','created_date','device','email','age','gender','income','ethnicity','homeZIP','schoolZip','workZip','cyclingfreq','rider_history','rider_type','app_version']
user.columns = user_col

user.drop(columns=['device','app_version','app_version','email'], inplace=True)

#change userid to str
user['userid'] = user['userid'].astype(str)

#%% merge trip and users

#join the user information with trip information
trip_and_user = pd.merge(trip,user,on='userid')

#%% import notes (optional)
note = pd.read_csv(path+"note.csv", header=None)

#%% pre-filter for creating sample dataset

#only get traces that cross COA borders
coa = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/base_shapefiles/bikewaysim_study_area/bikewaysim_study_area.shp')

#dissolve points by trip id
all_coords_dissolved = all_coords.dissolve('tripid').reset_index()

#find traces that are completely within the coa
#trips_within = all_coords_dissolved.sjoin(coa,predicate='crosses')['tripid'] # use if just crosses
trips_within = all_coords_dissolved.sjoin(coa,predicate='within')['tripid']

#only keep original columns
all_coords = all_coords[all_coords['tripid'].isin(trips_within)]

#%% create sample data


#select n random trips
#n = 500

#random_trips = all_coords['tripid'].drop_duplicates().sample(n)
#trip_mask = all_coords['tripid'].isin(random_trips)
#sample_coords = all_coords[trip_mask]

#use all
n = all_coords['tripid'].nunique()
random_trips = all_coords['tripid']
sample_coords = all_coords

#export gdf
sample_coords.to_file(rf'sample_trips/sample_coords_{n}.geojson',driver='GeoJSON')


#get user table
list_of_trips = sample_coords['tripid'].astype(str).drop_duplicates()

#%% now get user/trip info

#drop trips that aren't represented
sample_trip_and_user = trip_and_user[trip_and_user['tripid'].isin(random_trips)]

#tot datapoints
sample_trip_and_user['tot_points'] = pd.merge(sample_trip_and_user, tot_points, left_on='tripid',right_index=True)['tot_points']

#average speed
speed_stats = sample_coords.groupby('tripid').agg({'speed':['mean','median','max']})
speed_stats.columns = ['_'.join(col).strip() for col in speed_stats.columns.values]
sample_trip_and_user = pd.merge(sample_trip_and_user,speed_stats,left_on='tripid',right_index=True)

#average distance
def turn_into_linestring(points):
    #turn all trips into lines
    lines = points.sort_values(by=['tripid','datetime']).groupby('tripid')['geometry'].apply(lambda x: LineString(x.tolist()))

    #get start time
    start_time = points.groupby('tripid')['datetime'].min()

    #get end time
    end_time = points.groupby('tripid')['datetime'].max()
    
    #turn into gdf
    linestrings = gpd.GeoDataFrame({'start_time':start_time,'end_time':end_time,'geometry':lines}, geometry='geometry',crs="epsg:4326")
    
    return linestrings

lines = turn_into_linestring(sample_coords)

#project
lines = lines.to_crs(epsg='2240')

mean_dist_ft = lines.geometry.length.mean()
median_dist = lines.geometry.length.median()

#time difference
#get min, max index
idx_max = sample_coords.groupby('tripid')['datetime'].transform(max) == sample_coords['datetime']
idx_min = sample_coords.groupby('tripid')['datetime'].transform(min) == sample_coords['datetime']

#get min, max dfs
coords_max = sample_coords[idx_max][['tripid','datetime']].rename(columns={'datetime':'maxtime'})
coords_min = sample_coords[idx_min][['tripid','datetime']].rename(columns={'datetime':'mintime'})

#join these
coords_dif = pd.merge(coords_max, coords_min, on='tripid')

#find diffrence
coords_dif['duration'] = coords_dif['maxtime'] - coords_dif['mintime']

#add to trip and user df
sample_trip_and_user = pd.merge(sample_trip_and_user, coords_dif[['tripid','duration']], on='tripid')


#drop geometry
sample_coords.drop(columns=['geometry'],inplace=True)
#export csv
sample_coords.to_csv(rf'sample_trips/sample_coords_{n}.csv',index=False)


#export user
sample_trip_and_user.to_csv(rf'sample_trips/sample_trip_and_user_{n}.csv',index=False)

#%% get summary stats




#%% simplify gps traces




#%% export step


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
#     linestrings = gpd.GeoDataFrame({'start_time':start_time,'end_time':end_time,'geometry':lines}, geometry='geometry',crs="epsg:2240")
    
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