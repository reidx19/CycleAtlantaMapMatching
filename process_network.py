# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:19:04 2023

@author: tpassmore6
"""

import pandas as pd
import geopandas as gpd
from shapely.ops import Point, LineString
import numpy as np



# #run script
# if __name__ == "__main__":
    
#     export_fp = Path.home() / 'Downloads/cleaned_trips'
    
#     #load network
#     network_fp = Path.home() / "Downloads/cleaned_trips/networks/final_network.gpkg"

#     #project if neccessary
#     edges = gpd.read_file(network_fp,layer="links")[['A','B','geometry']]
#     nodes = gpd.read_file(network_fp,layer="nodes")[['N','geometry']]
    
#     #run function
#     links, nodes = explode_network(edges, nodes, 50)
    
#     #export
#     links.to_file(export_fp/'networks/match_network.gpkg',layer='links')
#     nodes.to_file(export_fp/'networks/match_network.gpkg',layer='nodes')


