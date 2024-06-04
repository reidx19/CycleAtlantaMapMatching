import datetime

def reduce_spacing(coords,speed_mph_thresh=2,spacing_ft=50):
    
    '''
    Drop points that have a speed less than 2 mph to clean up the traces (except for first)
    Then reduce number of points 
    '''

    # not everything has accurate speed data
    # if coords['speed_mph'].abs()>speed_mph_thresh:
    #     coords = coords[coords['speed_mph'].abs()>speed_mph_thresh]

    coords = calculate_coordinate_metrics(coords)
    
    current_spacing = spacing_ft
    
    #start with first point
    keep = [coords.index.tolist()[0]]
    
    for index, value in coords['traversed_distance_ft'].items():
        if value > current_spacing:
            keep.append(index)
            current_spacing = value + spacing_ft
    #remove points
    coords = coords.loc[keep]

    return coords

def calculate_coordinate_metrics(coords):
    '''
    Calculates acceleration, time differences, distance differences, time elapsed, etc.
    '''
    coords = coords.copy()
    
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
    coords.loc[:,'calculated_speed_mph'] = (coords['delta_distance_ft'] / feet_per_mile) / ((coords['delta_time'] / datetime.timedelta(seconds=1)) / seconds_per_hour)

    #attach sequence and total points for easier GIS examination
    coords.loc[:,'sequence'] = range(0,coords.shape[0])

    return coords

def calculate_trip_metrics(key,coords,trips_df):
    '''
    Updates the trip dataframe using the trace data
    '''
    
    # update the basic parameters
    trips_df.at[trips_df['tripid'] == key, 'start_time'] = coords['datetime'].min()
    trips_df.at[trips_df['tripid'] == key, 'end_time'] = coords['datetime'].max()
    trips_df.at[trips_df['tripid'] == key, 'start_lon'] = coords.at[coords['datetime'].idxmin(),'lon']
    trips_df.at[trips_df['tripid'] == key, 'start_lat'] = coords.at[coords['datetime'].idxmin(),'lat']
    trips_df.at[trips_df['tripid'] == key, 'end_lon'] = coords.at[coords['datetime'].idxmax(),'lon']
    trips_df.at[trips_df['tripid'] == key, 'end_lat'] = coords.at[coords['datetime'].idxmax(),'lat']
    trips_df.at[trips_df['tripid'] == key, 'start_X'] = coords.at[coords['datetime'].idxmin(),'X']
    trips_df.at[trips_df['tripid'] == key, 'start_Y'] = coords.at[coords['datetime'].idxmin(),'Y']
    trips_df.at[trips_df['tripid'] == key, 'end_X'] = coords.at[coords['datetime'].idxmax(),'X']
    trips_df.at[trips_df['tripid'] == key, 'end_Y'] = coords.at[coords['datetime'].idxmax(),'Y']
    trips_df.at[trips_df['tripid'] == key, 'duration'] = coords['datetime'].max() - coords['datetime'].min()
    trips_df.at[trips_df['tripid'] == key, 'total_points'] = coords.shape[0]
    trips_df.at[trips_df['tripid'] == key, 'avg_accuracy'] = coords['hAccuracy_ft'].mean()

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
