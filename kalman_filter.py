# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 21:24:41 2023

@author: tpassmore6
"""

import pandas as pd
import geopandas as gpd
import numpy.ma as ma
from pykalman import KalmanFilter

def kalman_speed(df_with_speed_and_time):
    '''
    Use kalman filter to smooth out speed data and interpolate acceleration data using
    recorded speed and timestamps. Uses basic kinematics with a constant speed assumption.

    Take in series/arrays of speed and timestamp and output the smoothed speeds and estimated
    accelerations.

    Assumes second by second data, so any missing timestamps and filled in and the speed is
    interpolated.
    '''
    
    #make a time elapsed in seconds column
    df_with_speed_and_time['time_elapsed'] = df_with_speed_and_time['datetime'].apply(lambda x: int((x - df_with_speed_and_time['datetime'].min()).total_seconds()))

    measurements = np.concat([speeds,timestamps])
    
    #use np.ma to mask missing data
    measurements = ma.masked_array(measurements, mask=np.isnan(measurements))
    
    # the initial state of the cyclist (speed = 0 and acceleration = 0)
    initial_state_mean = [0,0]
    
    #these are the kinematics of how we're moving assuming a constant speed (ignores elevation, signals, stop signs, etc)
    transition_matrix = [[1,1], # position_t-1 + speed_t-1 * t
                         [0,1]] # speed_t = 0 + speed_t-1
    
    #which 
    observation_matrix = [[1,0],
                          [0,1]]
        
    #estimate a kalman filter
    kf1 = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean,
                      )
    
    #get new values
    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(observations)
    
    #convert to dataframe
    filtered = pd.DataFrame(smoothed_state_means,columns=['speed','v_x','y','v_y'])
    
    #reset index and rename to elapsed time
    filtered.reset_index(inplace=True)
    filtered.rename(columns={'index':'time_elapsed'},inplace=True)
    
    return filtered

def kalman_geo(points,tripid,export_fp):
    '''
    Use kinematics (assume constant velocity) to smooth out positional data using XY and timestamps.
    
    . Points with low accuracy (high hAccuracy) are
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
    #smoothed_df.to_file(export_fp/f"selected_trips/{tripid}.gpkg",layer='smoothed_with_covar')



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