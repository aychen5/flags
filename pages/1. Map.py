#%%
import importlib.util
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np

import os, sys
import dotenv
from itertools import chain
from PIL import Image
from shapely.geometry import Point

import folium
from folium.plugins import Fullscreen
import altair as alt
from branca.colormap import linear, LinearColormap
from streamlit_folium import st_folium
from utils.map_utils import (
    mapillary_thumb_from_id, add_detection_markers,
    mapillary_viewer_link, make_census_layer, resolve_click
)

print("CWD:", os.getcwd())
print("Sys path[0]:", sys.path[0])
print("Utils dir exists?", os.path.isdir(os.path.join(os.getcwd(), "utils")))
print("__init__.py exists?", os.path.isfile(os.path.join("utils", "__init__.py")))
print("map_utils.py exists?", os.path.isfile(os.path.join("utils", "map_utils.py")))
spec = importlib.util.find_spec("utils.map_utils")
print("find_spec(utils.map_utils):", spec)

#import mapillary.interface as mly
#%%

# -----------------------------
# Configuration
# -----------------------------
CENSUS_CMAP = linear.YlGnBu_06.scale(0, 100)
CENSUS_CMAP.caption = "Census share (%)"
st.set_page_config(page_title="Map", page_icon=":round_pushpin:", layout="wide")

#st.markdown("# Plotting Demo")
st.sidebar.header("Who flies the flag?")
st.sidebar.write(
    """This map displays all American flags detected in NYC. The detection threshold was set to 0.85."""
)

#DEFAULT_DATA_PATH = "C:/Users/ANNIE CHEN/Box Sync/Personal/flags/"
DEFAULT_DATA_PATH = "https://storage.googleapis.com/streamlit-app-data/"
MAX_POPUPS = 3000               # max markers that get image popups (to avoid huge HTML)
THUMBNAIL_MAX_W = 320           # px
THUMBNAIL_MAX_H = 240           # px
NYC_CENTER = (40.7128, -74.0060)
DETECTION_THRESHOLD = 0.85

# for map col scheme
BINS = list(np.linspace(0, 100, 6))  # 0,20,40,60,80,100
PALETTE = 'YlGnBu'      

dotenv.load_dotenv(".env") 
# MLY_KEY = os.getenv("MLY_KEY")
#mly.set_access_token(os.getenv("MLY_KEY"))
# -----------------------------
#%%
# -----------------------------
# Data loading
# -----------------------------
census_tracts = gpd.read_file(f"{DEFAULT_DATA_PATH}2020_Census_Tracts_20250901.geojson")
census_geo_df = gpd.read_file(f"{DEFAULT_DATA_PATH}census_geo_df.geojson")
detected_flags_tracts_geo = gpd.read_file(f"{DEFAULT_DATA_PATH}detected_flags_tracts_geo.geojson")
clean_voter_ed_df = gpd.read_file(f"{DEFAULT_DATA_PATH}nyc_voter_data.geojson")
#pluto_shape = gpd.read_file(f'{DEFAULT_DATA_PATH}MapPLUTO.shp')

# Cache a tiny copy + spatial index
if "tract_gdf" not in st.session_state:
    # enforce CRS and id type for spatial tests
    if census_geo_df.crs is None or str(census_geo_df.crs).upper() != "EPSG:4326":
        census_geo_df = census_geo_df.to_crs(4326)
    st.session_state.tract_gdf = census_geo_df[["geoid", "geometry"]].copy()
    _ = st.session_state.tract_gdf.sindex  # build sindex

# ===== Some munging
# set threshold
markers_df = detected_flags_tracts_geo[detected_flags_tracts_geo['score'] >= DETECTION_THRESHOLD]

# build counts for choropleth coloring
tract_counts = (markers_df 
                .dropna(subset=["geoid"])
                .groupby("geoid").size()
                .rename("n_flags")
                .reset_index())

# join counts → tracts
tracts_plot = census_geo_df.merge(tract_counts, on="geoid", how="left")
tracts_plot["n_flags"] = tracts_plot["n_flags"].fillna(0).astype(int)
tracts_plot.set_geometry("geometry", inplace=True)
tracts_plot["geometry"] = tracts_plot.geometry.simplify(0.0002, preserve_topology=True)
tracts_plot.to_crs("EPSG:4326", inplace=True)
 
# quantile scale (robust to all-zero) 
vals = tracts_plot["n_flags"].to_numpy()
qs = np.quantile(vals, [0, .2, .4, .6, .8, 1.0]) if vals.size and vals.max() > 0 else np.array([0,0,0,0,0,0])

# only active voters
clean_active_voter_ed_df = clean_voter_ed_df[clean_voter_ed_df.STATUS == "Active"]

# define colormaps
dem_cmap = LinearColormap(["#f7fbff", "#c6dbef", "#6baed6", "#2171b5"], vmin=0, vmax=100)
rep_cmap = LinearColormap(["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d"], vmin=0, vmax=100)

# 3) PRECOMPUTE per-feature colors and store as properties
clean_active_voter_ed_df["fillColor_dem"] = clean_active_voter_ed_df["perc_dem"].apply(lambda v: dem_cmap(v) if np.isfinite(v) else "transparent")
clean_active_voter_ed_df["fillColor_rep"] = clean_active_voter_ed_df["perc_rep"].apply(lambda v: rep_cmap(v) if np.isfinite(v) else "transparent")

#%%

# --- Base map
nyc_map = folium.Map(location=[40.7, -74], zoom_start=10, tiles=None)
folium.TileLayer(
    'CartoDB positron',
    name='Map layers',  
    control=True
).add_to(nyc_map)
Fullscreen(
    position="topright",
    force_separate_button=True,
    title="Full screen",
    title_cancel="Exit full screen"
).add_to(nyc_map)

# layers
folium.map.CustomPane('polygons', z_index=450).add_to(nyc_map)
tracts_group = folium.FeatureGroup(name="Census Tracts", show=False)
#ed_group = folium.FeatureGroup(name="Electoral Districts", show=True)
#dem_group = folium.FeatureGroup(name="EDs — % Democrat", show=False)
#rep_group = folium.FeatureGroup(name="EDs — % Republican", show=False)

# --- census tracts
folium.GeoJson(
    census_geo_df,
    name='Census Tracts',
    style_function=lambda feature: {
        'fillColor': 'transparent',
        'color': 'darkgray',
        'weight': 1,
        'dashArray': '5, 5'
    },
    #pane = "polygons",
    interactive=True,
    tooltip=folium.GeoJsonTooltip(fields=['geoid'], aliases=['TRACT ID'])
).add_to(tracts_group)
tracts_group.add_to(nyc_map)

# --- electoral districts
# folium.GeoJson(
#     clean_active_voter_ed_df,
#     name='Electoral Districts',
#     style_function=lambda feature: {
#         'fillColor': 'transparent',
#         'color': 'darkgray',
#         'weight': 1,
#         'dashArray': '5, 5'
#     },
#     interactive=True,
#     tooltip=folium.GeoJsonTooltip(fields=['ELECTION_DIST'], aliases=['DISTRICT ID'])
# ).add_to(ed_group)
# ed_group.add_to(nyc_map)

# folium.GeoJson(
#     clean_active_voter_ed_df.to_json(),
#     name="EDs — % Democrat",
#     style_function=lambda feature: {
#                 "fillColor": feature["properties"].get("fillColor_dem", "transparent"),
#         "fillOpacity": 0.75,
#         "color": "darkgray",
#         "weight": 1,
#         "dashArray": "5, 5",
#     },
#     tooltip=folium.GeoJsonTooltip(fields=["ELECTION_DIST","perc_dem"],
#                                   aliases=["District","% Dem"], localize=True),
# ).add_to(dem_group)
# folium.GeoJson(
#     clean_active_voter_ed_df.to_json(),
#     name="EDs — % Republican",
#     style_function=lambda feature: {
#                 "fillColor": feature["properties"].get("fillColor_rep", "transparent"),
#         "fillOpacity": 0.75,
#         "color": "darkgray",
#         "weight": 1,
#         "dashArray": "5, 5",
#     },
#     tooltip=folium.GeoJsonTooltip(fields=["ELECTION_DIST","perc_rep"],
#                                   aliases=["District","% Rep"], localize=True),
# ).add_to(rep_group)
# dem_group.add_to(nyc_map)
# rep_group.add_to(nyc_map)

# --- num flags density
# folium.Choropleth(
#     geo_data=tracts_plot,
#     name='Number of flags',
#     data=tracts_plot,
#     columns=['geoid', 'n_flags'],
#     key_on='feature.id',
#     fill_color='YlOrRd',
#     fill_opacity=0.7,
#     line_opacity=0.2,
#     line_color='white', 
#     line_weight=0,
#     highlight=False, 
#     smooth_factor=1.0,
#     show=True,
#     #threshold_scale=[100, 250, 500, 1000, 2000],
#     legend_name= 'Number of flags').add_to(nyc_map)

# --- build census 
census_groups = ['hispanic_alone_pct', 
                 'white_alone_pct', 
                 'black_alone_pct'#,
                 #'amind_asknat_alone_pct', 
                 #'asian_alone_pct', 
                 #'hwi_pacisl_alone_pct',
                 #'other_alone_pct']
                ]

for group in census_groups:
    layer = make_census_layer(census_geo_df, group, group.replace("_alone_pct", " %").title())
    layer.add_to(nyc_map)
# add a single legend at the end
CENSUS_CMAP.add_to(nyc_map)

# --- flag markers
add_detection_markers(nyc_map, 
                      markers_df)


# layer control
folium.LayerControl(collapsed=False).add_to(nyc_map)

# two column layout
left, right = st.columns([2.3, 1], gap="large")

with left:
    out = st_folium(nyc_map, 
                    width=None, height=1000, 
                    returned_objects=['last_clicked', 'last_object_clicked'],
                    key="nyc_map")


with right:
    st.subheader("Census Tract Profile", divider=True)

    # Use robust lat/lon resolver (no popup parsing)
    geoid_cur = st.session_state.get("selected_geoid")
    geoid_new, marker_row = resolve_click(out, markers_df)

    if geoid_new:
        geoid_cur = geoid_new
        st.session_state.selected_geoid = geoid_new

    if not geoid_cur:
        st.caption("Click a census tract to see its demographics and detections.")
    else:
        tract_row = census_geo_df.loc[census_geo_df["geoid"].astype(str) == str(geoid_cur)].head(1)
        flags_in_tract = markers_df.loc[markers_df["geoid"].astype(str) == str(geoid_cur)]

        # Show once (no duplicates)
        st.metric("Tract ID",  int(geoid_cur) )
        st.metric("Number of Detections in tract", int(flags_in_tract.shape[0]))

        pct_cols_all = [
            'hispanic_alone_pct','white_alone_pct','black_alone_pct','asian_alone_pct',
            #'amind_asknat_alone_pct','hwi_pacisl_alone_pct','other_alone_pct'
        ]
        pct_cols = [c for c in pct_cols_all if c in census_geo_df.columns]
        if not tract_row.empty and pct_cols:
            tidy = (
                tract_row[pct_cols]
                .melt(value_name="pct", var_name="group")
                .assign(label=lambda d: d["group"].str.replace("_alone_pct","",regex=False)
                                        .str.replace("_"," ").str.title())
                )
            bar_h = 30  # px per category (28–32 works well)    
            h = max(180, bar_h * len(tidy))  # ensure enough room for all labels
            st.altair_chart(
                alt.Chart(tidy).mark_bar().encode(
                    x=alt.X("pct:Q", title="Share (%)", scale=alt.Scale(domain=[0,100])),
                    y=alt.Y(
                        "label:N",
                        # keep your sort by value; this preserves order
                        sort='-x',
                        title=None,
                        axis=alt.Axis(
                            labelOverlap=False,  
                            labelLimit=0,        
                            ticks=False     
                        )
                    ),
                    tooltip=[alt.Tooltip("label:N"), alt.Tooltip("pct:Q", format=".1f")]
                ).properties(height=h, width='container'),
                use_container_width=True
            )
        else:
            st.info("No demographics columns found to chart.")

    st.subheader("Selected detection", divider=True)
    if marker_row is not None:
        pid = marker_row.get("properties.id")
        url = mapillary_thumb_from_id(pid, res=1024)
        if url:
            st.image(url, width="stretch")
            st.caption(f"[Open in Mapillary ↗]({mapillary_viewer_link(pid)})")
        else:
            st.link_button("Open in Mapillary ↗", mapillary_viewer_link(pid), width="stretch")
    else:
        st.caption("Click a flag marker to preview its street view here.")
