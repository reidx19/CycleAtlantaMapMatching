import pandas as pd
import geopandas as gpd
import numpy as np
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
import pickle
import time
import datetime
from pathlib import Path
from tqdm import tqdm
from shapely.ops import Point, LineString
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def make_network(edges,nodes,n_col,lat_col,lon_col,a_col,b_col,use_latlon):
    '''
    Function for making network graph that is specific to leuven
    '''    
    
    # create network graph needed for map matching (using a projected coordinate system so latlon false)
    map_con = InMemMap("osm", use_latlon = use_latlon)
    
    #make map_con (the network graph)
    #add edges and nodes to leuven graph network object (make sure latlon is in same order and crs as traces)
    for row in nodes[[n_col,lat_col,lon_col]].itertuples(index=False):
        map_con.add_node(row[0], (row[1], row[2]))
    for row in edges[[a_col,b_col]].itertuples(index=False):
        map_con.add_edge(row[0], row[1])
    #add reverse links
    for row in edges[[b_col,a_col]].itertuples(index=False):
        map_con.add_edge(row[0], row[1])
    return map_con

def explode_network(links,nodes,tolerance_ft):

    '''
    problem: leuven assumes straight connecting lines between nodes to match to
    and that will not always be a good assumption
    
    this code takes the given network and breaks it down to include the curvature of the road
    
    to avoid adding too many edges and vertices, the rdp algorithm is used to remove vertices within
    a tolerence
    
    this function first simplifies edge geomtry and removes uneccessary points (excluding the start and end)
    
    it then assigns these new vertices temporary node IDs starting with +1 after the highest osm node id
    
    these added nodes are then used to split up existing links and add new reference ids
    
    when map matching is finished, all node ids with a temporary id can be removed and the real nodes used can be retrieved
    
    note this takes a while, so it should be used with caution, not usually needed for dense street networks
    
    Inputs:
        edges: dataframe with column A inidcating starting node and B indicating ending node and a geometry column
        nodes: dataframe with N column and geometry columm
        tolerance_ft: float with tolerance in units of CRS to perform RDP
    
    '''

    #make copies and subset
    links0 = links.copy()
    nodes0 = nodes.copy()
    links0 = links0[['A','B','geometry']]
    nodes0 = nodes0[['N','geometry']]

    # #first remove two way links
    # df_dup = pd.DataFrame(
    #     np.sort(links0[["A","B"]], axis=1),
    #         columns=["A","B"]
    #         )
    # df_dup = df_dup.drop_duplicates()
    
    #merge on index to get geometry
    #links0 = links0.loc[df_dup.index.tolist(),:]
    
    #rdp algorithm to simplify points
    links0['geometry'].apply(lambda row: row.simplify(tolerance_ft))
    links0.set_geometry('geometry',inplace=True)
    
    #sets up the numbering
    numbering = nodes0['N'].max() + 1
    
    new_links = {}
    new_nodes = {}
    rem = []
    
    for idx, row in links0.iterrows():
        
        #only if there are more than origin and end
        if len(row['geometry'].coords) > 2:
            #remove these links
            rem.append((row['A'],row['B']))
            
            line = row['geometry'].coords
            
            for i in range(0,len(line)-1):
                
                #form the new link
                new_link = LineString([line[i],line[i+1]])
                
                #start node
                if i == 0:
                    new_links[(row['A'],numbering+1)] = new_link
                
                #end node
                elif i == len(line) - 2:
                    new_links[(numbering+i,row['B'])] = new_link
                    
                    #add last node
                    new_nodes[numbering+i] = Point(line[i])
    
                #interstitial new_nodes    
                else:
                    new_links[(numbering+i,numbering+i+1)] = new_link
                    
                    #add first node to list
                    new_nodes[numbering+i] = Point(line[i])
    
            # number of unique ids assigned
            numbering += len(line) - 2
            
    #from dict to gdf
    new_links = pd.DataFrame.from_dict(new_links,orient='index',columns=['geometry']).reset_index()
    new_links['A'] = new_links['index'].apply(lambda row: row[0])
    new_links['B'] = new_links['index'].apply(lambda row: row[1])
    new_links = new_links[['A','B','geometry']]
    new_links = gpd.GeoDataFrame(new_links,geometry='geometry',crs=links0.crs)
    
    #from dict to gdf
    new_nodes = pd.DataFrame.from_dict(new_nodes,orient='index',columns=['geometry']).reset_index()
    new_nodes.columns = ['N','geometry']
    new_nodes = gpd.GeoDataFrame(new_nodes,geometry='geometry',crs=links0.crs)
    
    #remove links that were split up
    links0['tup'] = list(zip(links0['A'],links0['B']))
    links0 = links0[-links0['tup'].isin(rem)]
    links0.drop(columns=['tup'],inplace=True)
    
    #nodes
    nodes0['added'] = False
    new_nodes['added'] = True
    
    #add new links and nodes
    links0 = pd.concat([links0,new_links],ignore_index=True)
    nodes0 = pd.concat([nodes0,new_nodes],ignore_index=True)

    return links0, nodes0

#for exporting to gpkg
def convert_datetime_columns_to_str(dataframe):
    for column in dataframe.select_dtypes(include=['datetime64','timedelta64']):
        dataframe[column] = dataframe[column].astype(str)
    return dataframe

def leuven_match(trace:gpd.GeoDataFrame,tripid:int,matched_traces:dict,matching_settings:dict,edges:gpd.GeoDataFrame,map_con):

    #tripid = 11069#17825#10803#9999999999999999#17825#14011
    
    #trace = coords_dict[tripid].copy()
    
    trace.drop(columns=['sequence'],inplace=True)
    
    #reset index and sequence (do in pre-processing)
    trace.reset_index(inplace=True,drop=True)
    trace.reset_index(inplace=True)
    #trace.drop(columns=['sequence'],inplace=True)
    trace.rename(columns={'index':'sequence'},inplace=True)
    
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
                     non_emitting_length_factor=matching_settings['non_emitting_length_factor'], 
                     non_emitting_states=matching_settings['non_emitting_states'],
                     obs_noise=matching_settings['obs_noise'],
                     max_lattice_width=matching_settings['max_lattice_width'])
    
    #perform initial matching
    states, last_matched = matcher.match(gps_trace)
    match_nodes = matcher.path_pred_onlynodes
    
    #match ratio
    match_ratio = last_matched / (len(gps_trace)-1)
    
    #repeat match if failed and update states
    if match_ratio < .95:
        states, last_matched = matcher.increase_max_lattice_width(matcher.max_lattice_width+45, unique=False)
        match_nodes = matcher.path_pred_onlynodes
        match_ratio = last_matched / (len(gps_trace)-1)
    
    #remove interstitial nodes (nodes that are only in the exploded network)
    #have to deal with case where it begins/ends in interstitial node
    #look up states and replace states with appropriate one based on direction
    check_list = set(nodes['N'].tolist())
    match_nodes = [match_node for match_node in match_nodes if match_node in check_list]
    
    #get reduced states without removing valid backtracking
    reduced_states = []
    for i in range(len(match_nodes)-1):
        if match_nodes[i]!=match_nodes[i+1]:
            one_link = (match_nodes[i],match_nodes[i+1])
            reduced_states.append(one_link)
    
    # check if match ratio here is better than the one in records and overwrite if yes
    if len(matched_traces) > 0:
        if tripid in matched_traces.keys():
            if matched_traces[tripid]['match_ratio'] > match_ratio:
                print('no improvement')
    
    #retreive matched edges from network and turn into geodataframe
    geos_list = []
    for state in reduced_states:
        geo = geos_dict.get(state,0)
        #check if it was a reverse link
        if geo == 0:
            state = tuple(reversed(state))
            geo = geos_dict.get(state,0)
        geos_list.append(geo)
    
    #form geoadataframe of matched trip
    matched_trip = gpd.GeoDataFrame(data={'A_B':reduced_states,'geometry':geos_list},geometry='geometry',crs=trace.crs)
    
    #turn tuple to str for exporting to gpkg
    matched_trip['A_B'] = matched_trip['A_B'].apply(lambda row: f'{row[0]}_{row[1]}')
    
    #reset index to add an edge sequence column
    matched_trip.reset_index(inplace=True)
    matched_trip.rename(columns={'index':'edge_sequence'},inplace=True)
    
    #export traces and matched line to gpkg for easy examination
    matched_trip.to_file(export_fp/f"matched_traces/{tripid}.gpkg",layer='matched_trace')
    
    #turn datetime to str for exporting to gpkg
    trace = convert_datetime_columns_to_str(trace)
    
    #export gps points
    trace.to_file(export_fp/f"matched_traces/{tripid}.gpkg",layer='gps_points')
    
    #get interpolated points (need to flip coords because leuven uses YX order)
    trace['interpolated_point'] = pd.Series([ Point(x.edge_m.pi[1],x.edge_m.pi[0]) for x in matcher.lattice_best ])
    
    #drop non-matched
    trace = trace.dropna()
    
    #create match lines
    trace['match_lines'] = trace.apply(lambda row: LineString([row['geometry'],row['interpolated_point']]),axis=1)
    
    #create gdf for interpolated points
    interpolated_points = trace[['sequence','interpolated_point']]
    interpolated_points = gpd.GeoDataFrame(interpolated_points,geometry='interpolated_point',crs=trace.crs)
    
    #create gdf for match lines
    match_lines = trace[['sequence','match_lines']]
    match_lines = gpd.GeoDataFrame(match_lines,geometry='match_lines',crs=trace.crs)
    match_lines['length'] = match_lines.length
    
    #export both
    interpolated_points.to_file(export_fp/f"matched_traces/{tripid}.gpkg",layer='interpolated_points')
    match_lines.to_file(export_fp/f"matched_traces/{tripid}.gpkg",layer='match_lines')
    
    #add to matched_traces dictionary
    matched_traces[tripid] = {
        'nodes':match_nodes, #list of the matched node ids
        'edges':states, #list of the matched edge ids
        'last_matched':last_matched, #last gps point reached
        'match_ratio':match_ratio, #percent of points matched
        'max_lattice_width':matcher.max_lattice_width, # record the final lattice width
        'matched_trip': matched_trip, #gdf of matched lines
        'match_lines': match_lines, # gdf of distances between interpolated point and gps point
        'interpolated_points': interpolated_points, # gdf of interpolated points on match line
        'match_time_sec': time.time() - start, #time it took to match
        'match_distance': matcher.path_pred_distance(), # distance along network
        'gps_distance': matcher.path_distance(), # total euclidean distance between all gps points
        'time': datetime.datetime.now(), # record when it was last matched
        }
    
    #update trips_df
    trips_df.at[trips_df['tripid']==tripid,'match_ratio'] = match_ratio
    trips_df.at[trips_df['tripid']==tripid,'last_matched'] = datetime.datetime.now()
    
    # #export DOT file for lattice visualization (testing)
    # if matching_settings['export_graph'] == True:
    #     fh =(export_fp/f'dot_files/{tripid}.dot').open('w')
    #     matcher.lattice_dot(file=fh, precision=3, render=False)
    
    return matched_traces, trips_df


#%% load data

#file paths
export_fp = Path.home() / 'Downloads/cleaned_trips'
network_fp = Path.home() / "Downloads/cleaned_trips/networks/final_network.gpkg"

#import network
edges = gpd.read_file(network_fp,layer="links")[['A','B','geometry']]
nodes = gpd.read_file(network_fp,layer="nodes")[['N','geometry']]

#explode network
exploded_edges, exploded_nodes = explode_network(edges, nodes, tolerance_ft = 50)

#add latlon columns
exploded_nodes['X'] = exploded_nodes.geometry.x
exploded_nodes['Y'] = exploded_nodes.geometry.y

#load all traces
with (export_fp/'cleaned_traces.pkl').open('rb') as fh:
    coords_dict, trips_df = pickle.load(fh)

map_con = make_network(exploded_edges,exploded_nodes,n_col='N',lat_col='Y',lon_col='X',a_col='A',b_col='B',use_latlon=False)

#note: distance units will be units of projected CRS
matching_settings = {
    'max_dist':700,  # maximum distance for considering a link a candidate match for a GPS point
    'min_prob_norm':0.001, # drops routes that are below a certain normalized probability  
    'non_emitting_length_factor': 0.75, # reduces the prob of non-emitting states the longer the sequence is
    'non_emitting_states': True, # allow for states that don't have matching GPS points
    'obs_noise': 50, # the standard error in GPS measurement 50 ft (could use hAccuracy)
    'max_lattice_width': 5, # limits the number of possible routes to consider, can increment if no solution is found
    'increase_max_lattice_width' : True, # increases max lattice width by one until match is complete or max lattice width is 50
    'export_graph': False # set to true to export a .DOT file to visualize in graphviz or gephi (this feature is messy)
}

#load existing matches/if none then create a new dict
if (export_fp/'matched_traces.pkl').exists():
    with (export_fp/'matched_traces.pkl').open('rb') as fh:
        matched_traces, trips_df, failed_match = pickle.load(fh)
else:
    matched_traces = dict()
    failed_match = []

#%%single match
#matched_traces = leuven_match(trace,tripid,matched_traces,matching_settings,edges,map_con)

#%% batch match
for tripid, trace in tqdm(coords_dict.items()):
    try:
        matched_traces, trips_df = leuven_match(trace,tripid,matched_traces,matching_settings,edges,map_con)
        #update trips_df
    except:
        if tripid in matched_traces.keys():
            failed_match.append(tripid)
        export_files = (matched_traces,trips_df,failed_match)
        with (export_fp/'matched_traces.pkl').open('wb') as fh:
            pickle.dump(export_files,fh)
        
        
#export
export_files = (matched_traces,trips_df,failed_match)
with (export_fp/'matched_traces.pkl').open('wb') as fh:
    pickle.dump(export_files,fh)
