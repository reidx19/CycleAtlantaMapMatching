# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:19:04 2023

@author: tpassmore6
"""

import pandas as pd
import geopandas as gpd
from shapely.ops import Point, LineString
import numpy as np
from pathlib import Path

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

    #first remove two way links
    df_dup = pd.DataFrame(
        np.sort(links0[["A","B"]], axis=1),
            columns=["A","B"]
            )
    df_dup = df_dup.drop_duplicates()
    
    #merge on index to get geometry
    links0 = links0.loc[df_dup.index.tolist(),:]
    
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
    links['tup'] = list(zip(links['A'],links['B']))
    links = links[-links['tup'].isin(rem)]
    links.drop(columns=['tup'],inplace=True)
    
    #nodes
    nodes['added'] = False
    new_nodes['added'] = True
    
    #add new links and nodes
    links = pd.concat([links,new_links],ignore_index=True)
    nodes = pd.concat([nodes,new_nodes],ignore_index=True)

    return links, nodes

#run script
if __name__ == "__main__":
    
    export_fp = Path.home() / 'Downloads/cleaned_trips'
    
    #load network
    network_fp = Path.home() / "Downloads/cleaned_trips/networks/final_network.gpkg"

    #project if neccessary
    edges = gpd.read_file(network_fp,layer="links")[['A','B','geometry']]
    nodes = gpd.read_file(network_fp,layer="nodes")[['N','geometry']]
    
    #run function
    links, nodes = explode_network(edges, nodes, 50)
    
    #export
    links.to_file(export_fp/'networks/match_network.gpkg',layer='links')
    nodes.to_file(export_fp/'networks/match_network.gpkg',layer='nodes')


