import pandas as pd
import geopandas as gpd
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
import pickle
import time
import datetime
from pathlib import Path
from tqdm import tqdm
from shapely.ops import Point, LineString

def make_network(edges,nodes,n_col,lat_col,lon_col,a_col,b_col,use_latlon):
    
    # create network graph needed for map matching (using a projected coordinate system so latlon false)
    map_con = InMemMap("marta_osm", use_latlon = matching_settings['use_latlon'])
    
    #make map_con (the network graph)
    #add edges and nodes to leuven graph network object (make sure latlon is in same order and crs as traces)
    for row in nodes[[n_col,lat_col,lon_col]].itertuples(index=False):
        map_con.add_node(row[0], (row[1], row[2]))
    for row in edges[[a_col,b_col]].itertuples(index=False):
        map_con.add_edge(row[0], row[1])

    return map_con

def leuven_match(trace:gpd.GeoDataFrame,tripid:int,matched_traces:dict,matching_settings:dict,edges:gpd.GeoDataFrame,map_con):
    #record start time for matching
    start = time.time()
    
    #get list of points
    gps_trace = list(zip(trace.geometry.y,trace.geometry.x))
    
    #turn network into dict to quickly retrieve geometries
    edges['tup'] = list(zip(edges['A'],edges['B']))
    geos_dict = dict(zip(edges['tup'],edges['geometry']))
    
    #set up matching
    matcher = DistanceMatcher(map_con,
                     max_dist=matching_settings['max_dist'],
                     min_prob_norm=matching_settings['min_prob_norm'],
                     non_emitting_length_factor=matching_settings['non_emitting_states_factor'], 
                     non_emitting_states=matching_settings['non_emitting_states'],
                     obs_noise=matching_settings['obs_noise'],
                     max_lattice_width=matching_settings['max_lattice_width'])

    #perform initial matching
    states, last_matched = matcher.match(gps_trace)
    match_nodes = matcher.path_pred_onlynodes

    #match ratio
    match_ratio = last_matched / (len(gps_trace)-1)

    #repeat match if failed and update states
    while matcher.max_lattice_width < 20:
        if match_ratio < 1:
            states, last_matched = matcher.increase_max_lattice_width(matcher.max_lattice_width+1, unique=False)
            match_ratio = last_matched / (len(gps_trace)-1)
        else:
            break

    #get reduced states
    reduced_states = list(set(states))

    #check if match ratio here is better than the one in records and overwrite if yes
    if tripid in matched_traces.keys():
        if matched_traces[tripid]['match_ratio'] > match_ratio:
            return matched_traces

    #retreive matched edges from network and turn into geodataframe
    geos_list = [geos_dict.get(id,0) for id in reduced_states]
    matched_trip = gpd.GeoDataFrame(data={'A_B':reduced_states,'geometry':geos_list},geometry='geometry',crs=matching_settings['crs'])

    #turn tuple to str for exporting to gpkg
    matched_trip['A_B'] = matched_trip['A_B'].apply(lambda row: f'{row[0]}_{row[1]}')

    #reset index to add an edge sequence column
    matched_trip.reset_index(inplace=True)
    matched_trip.rename(columns={'index':'edge_sequence'},inplace=True)

    #export traces and matched line to gpkg for easy examination
    matched_trip.to_file(export_fp/f"matched_traces/{tripid}.gpkg",layer='matched_trace')

    #turn datetime to str for exporting to gpkg
    trace['datetime'] = trace['datetime'].astype(str)
    trace['time_from_start'] = trace['time_from_start'].astype(str)

    #export gps points
    trace.to_file(export_fp/f"matched_trace/{tripid}.gpkg",layer='gps_points')

    #get interpolated points (need to flip coords because leuven uses YX order)
    trace['interpolated_point'] = pd.Series([ Point(x.edge_m.pi[1],x.edge_m.pi[0]) for x in matcher.lattice_best ])
    #only get ones that matched
    trace = trace.loc[0:last_matched]

    #create match lines
    trace['match_lines'] = trace.apply(lambda row: LineString([row['geometry'],row['interpolated_point']]),axis=1)

    #create gdf for interpolated points
    interpolated_points = trace[['sequence','interpolated_point']]
    interpolated_points = gpd.GeoDataFrame(interpolated_points,geometry='interpolated_point')

    #create gdf for match lines
    match_lines = trace[['sequence','match_lines']]
    match_lines = gpd.GeoDataFrame(match_lines,geometry='match_lines')
    match_lines['length'] = match_lines.length

    #export both
    interpolated_points.to_file(export_fp/f"matched_trace/{tripid}.gpkg",layer='interpolated_points')
    match_lines.to_file(export_fp/f"matched_trace/{tripid}.gpkg",layer='match_lines')

    #add to matched_traces dictionary
    matched_traces[tripid] = {
        'nodes':match_nodes, #list of the matched node ids
        'edges':reduced_states, #list of the matched edge ids
        'last_matched':last_matched, #last gps point reached
        'match_ratio':match_ratio, #percent of points matched
        'lattice_stats': matcher.print_lattice_stats(), # get lattice stats
        'matched_trip': matched_trip, #gdf of matched lines
        'match_lines': match_lines, # gdf of distances between interpolated point and gps point
        'interpolated_points': interpolated_points, # gdf of interpolated points on match line
        'match_time_sec': time.time() - start, #time it took to match
        'time': datetime.datetime.now(), # record when it was last matched
        }

    #export DOT file for lattice visualization (testing)
    if matching_settings['export_graph'] == True:
        fh =(export_fp/f'dot_files/{tripid}.dot').open('w')
        matcher.lattice_dot(file=fh, precision=3, render=False)

    return matched_traces

#%% run

#export filepath
export_fp = Path.home() / 'Downloads/cleaned_trips'

#load dict of traces (replace with database)
with (export_fp/'coords_dict.pkl').open('rb') as fh:
    simp_dict = pickle.load(fh)
    
#load csv of trips
trips_df = pd.read_csv(export_fp/'trips.csv')

#load network
network_fp = Path.home() / "Downloads/cleaned_trips/networks/final_network.gpkg"
#project if neccessary
edges = gpd.read_file(network_fp,layer="links")
nodes = gpd.read_file(network_fp,layer="nodes")

#redo the latlon columns
nodes['X'] = nodes.geometry.x
nodes['Y'] = nodes.geometry.y
 
#load traces
with (export_fp/'coords_dict.pkl').open('rb') as fh:
    coords_dict = pickle.load(fh)

matching_settings = {
    'max_dist':200,  # maximum distance for considering a link a candidate match for a GPS point
    'min_prob_norm':0.001, # drops routes that are below a certain normalized probability  
    'non_emitting_length_factor': 0.75, # reduces the prob of non-emitting states if the longer the sequence is
    'non_emitting_states': False, # allow for states that don't have matching GPS points
    'obs_noise': 50, # the standard error in GPS measurement
    'max_lattice_width': 1, # limits the number of possible routes to consider, can increment if no solution is found
    'increase_max_lattice_width' : True, # increases max lattice width by one until match is complete or max lattice width is 20
    'crs': 'epsg:2240', # set to epsg:4326 for unprojected latlon
    'export_graph': False # set to true to export a .DOT file to visualize in graphviz or gephi
}

# #load existing matches/if none then create a new dict
# if (export_fp/'matched_traces.pkl').exists():
#     with (export_fp/'matched_traces.pkl').open('rb') as fh:
#         matched_traces = pickle.load(fh)
# else:
#     matched_traces = dict()

#load trips_df
matched_traces = dict()

map_con = make_network(edges,nodes,n_col='N',lat_col='Y',lon_col='X',a_col='A',b_col='B',use_latlon=False)

#%% one match

tripid = 17825#14011
trace = coords_dict[tripid]

leuven_match(trace,tripid,matched_traces,matching_settings,edges,map_con)

#%% multi-match

for tripid in tqdm(coords_dict.keys()):
    try:
        trace = coords_dict[tripid]
        matched_traces = leuven_match(trace,tripid,matched_traces,matching_settings,edges,map_con)
    except:
        #if anything breaks export the matched_traces dict as is
        with (export_fp/'matched_traces.pkl').open('wb') as fh:
            pickle.dump(matched_traces,fh)

#%% export matched traces
        with (export_fp/'matched_traces.pkl').open('wb') as fh:
            pickle.dump(matched_traces,fh)


#%% update trips_df

#import

    #updates trips_df with relevant info
    trips_df.at[trips_df['tripid']==tripid,['match_ratio','match_time','match_status']] = [match_ratio,str(datetime.datetime.now()),'matched']
