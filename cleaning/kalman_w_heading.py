# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:20:29 2023

@author: tpassmore6
"""

import numpy as np
import numpy.ma as ma
from pykalman import KalmanFilter
import pickle
import pandas as pd
import geopandas as gpd
from pathlib import Path

export_fp = Path.home() / 'Downloads/cleaned_trips/justin'

with (export_fp/'coords_dict.pkl').open('rb') as fh:
    coords_dict = pickle.load(fh)
    
#points = coords_dict[2200285601053]
#points = coords_dict[2207683501069]
points = coords_dict[2219927301082]

#LA
#points.to_crs('EPSG:2229',inplace=True)

#SF
#points.to_crs('EPSG:2227',inplace=True)

#chi
points.to_crs('EPSG:3435',inplace=True)

#use great circle distance instead here

points['X'] = points.geometry.x
points['Y'] = points.geometry.y

points['rad'] = points['bearing'] / 180 * np.pi
points['Vx'] = points['speed'] * np.sin(points['rad'])
points['Vy'] = points['speed'] * np.cos(points['rad'])

#throw out first and last point
points = points.iloc[1:-1]

points.to_file(export_fp/"kalman2.gpkg",layer='raw')



#%%


'''
Use kinematics (assume constant velocity) to smooth out data. Points with low accuracy (high hAccuracy) are
removed, but there should be a way to account for this.

Adapted code from this:
https://stackoverflow.com/questions/43377626/how-to-use-kalman-filter-in-python-for-location-data

Read this for more about kalman filters: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

'''   
  
#make sure data is sorted
points = points.sort_values('datetime')

#make a time elapsed in seconds column
points['time_elapsed'] = points['datetime'].apply(lambda x: int((x - points['datetime'].min()).total_seconds()))
    
#create nan entries to fill in missing data
fill = pd.DataFrame(data={'time_elapsed':range(0,points['time_elapsed'].max()+1)})
fill = pd.merge(fill,points,on='time_elapsed',how='left')

#convert our observations to numpy array
observations = fill[['X','Vx','Y','Vy']].to_numpy()

#errors
#errors = fill[['hAccuracy']].to_numpy()
#errors = np.concatenate([errors,errors],axis=1)

#use np.ma to mask missing data
observations = ma.masked_array(observations , mask=np.isnan(observations))
#rrors = ma.masked_array(errors, mask=np.isnan(errors))

# the initial state of the cyclist (assuming starting from stop)
# so initial position in x and y
# assume 0 velocity (acceleration is an unaacounted for external influence)
initial_state_mean = [observations[0,0],0,observations[0,2],0]

#these are the kinematics of how we're moving (ignoring road network and grade)
#assume that velocity is constant (not true) but we don't have acceleration information
transition_matrix = [[1,1,0,0], # x position * t + x velocity
                      [0,1,0,0], # x velocity
                      [0,0,1,1], # y position * t + y velocity
                      [0,0,0,1]] # y velocity

#these are the values we're getting from the phone (just x and y position, we don't know direction)
#QUESTION how do we include the other measurements that we're taking? we know hAccuracy is influecing the position and speed is related to speed
#also how to account for time skips?
observation_matrix = [[1,0,0,0], # first column is x position
                      [0,1,0,0], # second column is x velocity
                      [0,0,1,0], # third column is y position
                      [0,0,0,1]] # fourth column is y velocity

   
# #how confident are we in our measurements (assume x and y don't affect each other)
# observation_covariance = fill['hAccuracy'].mean()**2 * np.array([[1,0],[0,1]])

#using just this estimate a kalman filter
kf1 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean,
                  #observation_covariance= observation_covariance
                  #observation_offsets=errors
                  )

#get new values
(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(observations)

#try smoothing again without those points

#convert back to dataframe
filtered = pd.DataFrame(smoothed_state_means,columns=['X_est','Vx_est','Y_est','Vy_est'])

#create an estimated speed column (mph)
filtered['speed_est'] = (filtered['Vx_est']**2 + filtered['Vy_est']**2)**0.5

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
smoothed_df['geometry'] = gpd.points_from_xy(smoothed_df['X_est'], smoothed_df['Y_est'], crs=points.crs)
smoothed_df = gpd.GeoDataFrame(smoothed_df,geometry='geometry')

#export
smoothed_df.to_file(export_fp/"kalman2.gpkg",layer='smoothed')

#%%





#%%

