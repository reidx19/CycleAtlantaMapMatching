# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:11:36 2023

@author: tpassmore6
"""

import pickle
from pathlib import Path
import numpy as np


export_fp = Path.home() / 'Downloads/cleaned_trips'

#load existing matches/if none then create a new dict
if (export_fp/'matched_traces.pkl').exists():
    with (export_fp/'matched_traces.pkl').open('rb') as fh:
        matched_traces = pickle.load(fh)
else:
    print('no matched traces')
    
    
#print total run time and average run time per trace if complete
run_times = np.array([item['match_time_sec'] for key, item in matched_traces.items() if item['match_ratio'] > 0.95])
print(f"{len(run_times)} trips with match ratio above 0.95")
print(f"Total run time: {np.round(run_times.sum()/60**2,decimals=1)} hours")
print(f"Average match time: {np.round(run_times.mean(),decimals=1)} seconds")

# #add to matched_traces dictionary
# matched_traces[tripid] = {
#     'nodes':match_nodes, #list of the matched node ids
#     'edges':states, #list of the matched edge ids
#     'last_matched':last_matched, #last gps point reached
#     'match_ratio':match_ratio, #percent of points matched
#     'max_lattice_width':matcher.max_lattice_width, # record the final lattice width
#     'matched_trip': matched_trip, #gdf of matched lines
#     'match_lines': match_lines, # gdf of distances between interpolated point and gps point
#     'interpolated_points': interpolated_points, # gdf of interpolated points on match line
#     'match_time_sec': time.time() - start, #time it took to match
#     'match_distance': matcher.path_pred_distance(), # distance along network
#     'gps_distance': matcher.path_distance(), # total euclidean distance between all gps points
#     'time': datetime.datetime.now(), # record when it was last matched
#     }


#%%

good_trips = np.array([key for key, item in matched_traces.items() if item['match_ratio'] > 0.95])
np.random.choice(good_trips,1)
