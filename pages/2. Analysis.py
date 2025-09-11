#%%
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np

import os, re, json
import dotenv
from itertools import chain
from PIL import Image

import plotnine
import mapillary.interface as mly
#%%
# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(page_title="Plots", page_icon=":bar_chart:", layout="wide")

#st.markdown("# Plotting Demo")
st.sidebar.header("Who flies the flag?")
st.sidebar.write(
    """This map displays all American flags detected in NYC. The detection threshold was set to 0.8."""
)

DEFAULT_DATA_PATH = "C:/Users/ANNIE CHEN/Box Sync/Personal/flags/"


dotenv.load_dotenv(".env")
mly.set_access_token(os.getenv("MLY_KEY"))
#%%
# ---------------

n_flags_df = detected_flags_tracts_geo.groupby(['year'])['image_id'].count().reset_index()
n_flags_df.columns = [ 'year', 'n_flags']


plotnine.ggplot(n_flags_df) + \
    plotnine.aes(x='year', 
                 y='n_flags') + \
    plotnine.geom_line(size = 2,
                       alpha = .5,
                       show_legend = False) + \
    plotnine.geom_point(size = 10,
                       show_legend = False) + \
    plotnine.geom_text(plotnine.aes(label='n_flags'), 
                       color = "white",
                       size = 8,
                       show_legend = False) + \
    plotnine.labs(x = "", y = "Number of flags") + \
    plotnine.scale_y_continuous(breaks=range(0, n_flags_df['n_flags'].max()+100, 200)) + \
    plotnine.theme_minimal()
#%%