#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import geopandas as gpd
import pyproj
import numpy as np
import geojson
from shapely.geometry import Point, LineString, MultiPoint
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

def sample_road_segments(line_string, line_length, distance_delta = 16, sample_prop = .33):
    '''
    Returns a list of sampled subsegments given a road segment. 
    Each subsegment is a shapely LineString.

    - line_string is the geometry of the road segment from shapefile
    - line_length is the length of the road segment, in US feet.
    - distance_delta is the length of the subsegments, and should be in US feet.
    E.g., distance_delta = 16 ft is about 5 meters.
    - sample_prop is the proportion of the total number of segments to sample.
    '''
    lnstr = line_string
    lnlng = line_length 
    # get equidistant points along the line
    equdist = np.arange(0, lnlng, distance_delta)
    # get the point for value in equdist
    equdist_points = [lnstr.interpolate(d) for d in equdist] 
    try:
        # add end point
        end_point = lnstr.boundary.geoms[1]
        equdist_points.append(end_point)
    except IndexError:
        print(f'There are no boundaries for {lnstr}.')
        # there's a geometry with no boundaries, looks like a polygon
    all_points = MultiPoint(equdist_points)
    n_subsegments = len(equdist_points) - 1
    print(f'N road segments of {distance_delta} ft: {n_subsegments}')
    # sample some ratio from the line
    if n_subsegments <= 3:
        n_sample = 1
    else:
        n_sample = int(n_subsegments * sample_prop)
    sampled_points = np.random.choice(equdist_points[1:], size=n_sample, replace=False)
    # get the line segments between the sampled points
    sampled_segments = []
    for s in sampled_points:
        s_ind = equdist_points.index(s)
        s_segment = LineString(equdist_points[s_ind-1:s_ind+1])
        sampled_segments.append(s_segment)
    return(sampled_segments)

def transform_NAD83_to_WGS84(line_segment):
    '''
    Need to use geographic coordinates (not projections) for geodetic calculations in pyproj 
    '''
    x1, y1, x2, y2 = line_segment.coords[0][0], line_segment.coords[0][1], line_segment.coords[1][0], line_segment.coords[1][1]
    # transform coordinates from NAD83 to WGS84
    transformer = pyproj.Transformer.from_crs("epsg:2263", "epsg:4326")
    new_x1, new_y1 = transformer.transform(x1, y1)
    new_x2, new_y2 = transformer.transform(x2, y2)
    new_coords = [(new_x1, new_y1), (new_x2, new_y2)] 
    return new_coords

def calculate_midpoint(subsegment):
    '''
    Get the midpoint of a sampled road subsegment. 

    Input is in lat/long (WGS84)
    
    Using formula found here: https://www.movable-type.co.uk/scripts/latlong.html
    It uses Haversine to calculate distance.
    '''
    # break down coordinates 
    lat1, lng1 = subsegment[0][0], subsegment[0][1]
    lat2, lng2 =  subsegment[1][0], subsegment[1][1]

    # convert to radians
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)

    lng_diff = lng2_rad - lng1_rad

    Bx = math.cos(lat2_rad) * math.cos(lng_diff)
    By = math.cos(lat2_rad) * math.sin(lng_diff)

    # formula
    lat3 = math.atan2(math.sin(lat1_rad) + math.sin(lat2_rad),
                      math.sqrt((math.cos(lat1_rad) + Bx) * (math.cos(lat1_rad) + Bx) + By * By) 
                      )
    lng3 = lng1_rad + math.atan2(By, math.cos(lat1_rad) + Bx)
    lat3, lng3 = math.degrees(lat3), math.degrees(lng3)
    midpoint = (lat3, lng3)
    return midpoint

def calculate_bearing(subsegment):
    '''
    Get the heading from the sampled road segments by performing geodetic calculations.
    Input is a line segment has two coordinates (i, j) and (k, l).
    Output of interest is forward azimuth, which is effectively the heading for GSV param. Ranges from 0 to 360 degrees.
    '''
     # extract the correct geod ellipsoid given WGS84
    geodesic = pyproj.Geod(ellps='WGS84') 
    # break down coordinates 
    x1, y1 = subsegment[0][0], subsegment[0][1]
    x2, y2 =  subsegment[1][0], subsegment[1][1]
    # get the azimuth from the first point to the second point
    # back azimuth is the bearing in the reverse direction
    fwd_azimuth, back_azimuth, distance = geodesic.inv(y1, x1, y2, x2 )
    # make back azimuth positive
    if fwd_azimuth <= 180:
        back_azimuth = fwd_azimuth + 180
    elif fwd_azimuth > 180:
        back_azimuth = fwd_azimuth - 180
    result = {'fwd_azimuth': fwd_azimuth, 
              'back_azimuth': back_azimuth, 
              'distance_metres': distance}
    return result

if __name__ == '__main__':
    # ----- DATA
    # this is road centerline from NYC Open Data 
    centerfile_shp = gpd.read_file("./data/Centerline_20230109/Centerline.shp")
    # basic info about the data
    print(centerfile_shp.crs) # coordinate system = NAD83 EPSG:2263
    # get the road segments from the shapefile
    input_lines = centerfile_shp.geometry
    input_lengths = centerfile_shp.SHAPE_Leng
    sampled_subsegments = [sample_road_segments(i, j) for i, j in zip(input_lines, input_lengths)]

    # For each sampled road segment:
    # - transform coordinates from NAD83 to WGS84 
    # - and get the heading/midpoint 
    sampled_roads = []
    for i in tqdm(range(len(sampled_subsegments))):
        road_subsegments = sampled_subsegments[i]
        road_subsegments_geo = [transform_NAD83_to_WGS84(j) for j in road_subsegments]
        # compile list of all the bearings for each sampled road
        segment_bearings = []
        segment_midpoints = []
        for k in range(len(road_subsegments_geo)):
            segment_bearings.append(calculate_bearing(road_subsegments_geo[k]))
            # also get the midpoint of the sampled road segments
            segment_midpoints.append(calculate_midpoint(road_subsegments_geo[k]))
        sampled_road_info = {"road_index" : i,
                             #"road_projection" : road_subsegments, # nad83 epsg:2263
                             "road_geographic" : road_subsegments_geo, # wgs84 epsg:4326
                             "bearings" : segment_bearings, # azimuths
                             "midpoints" : segment_midpoints} # midpoints
        sampled_roads.append(sampled_road_info)
   
    # convert to json and write out 
    # (classical json cannot serialize LINESTRING objects, need geojson)
    for i in tqdm(range(len(sampled_roads))):
        sampled_roads_json = geojson.dumps(sampled_roads[i], sort_keys=True)
        with open(f"./data/clean_centerline/centerline_sampled-{i}.json", "w") as f:
            f.write(sampled_roads_json)






