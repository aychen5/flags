#%%
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np

import os, re, json
import html
import dotenv
from itertools import chain
from PIL import Image
from shapely.geometry import Point

import folium
from folium.plugins import Fullscreen
import altair as alt
from branca.colormap import linear, LinearColormap
from streamlit_folium import st_folium
#import mapillary.interface as mly
#%%
# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(page_title="Map", page_icon=":round_pushpin:", layout="wide")

#st.markdown("# Plotting Demo")
st.sidebar.header("Who flies the flag?")
st.sidebar.write(
    """This map displays all American flags detected in NYC. The detection threshold was set to 0.85."""
)

# st.markdown("""
#     <style>
#     .block-container { padding: 0; }
#     header { visibility: hidden; }
#     footer { visibility: hidden; }
#     </style>
# """, unsafe_allow_html=True)

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
#mly.set_access_token(os.getenv("MLY_KEY"))
#%%
# -----------------------------
# Utilities
# -----------------------------

#@st.cache_data(show_spinner=False)
def mapillary_thumb_from_id(pid: str, res: int = 1024) -> str | None:
    s = str(pid).strip()                     
    try:
        # Mapillary photo key -> short-lived thumbnail URL
        #return mly.image_thumbnail(image_id=s, resolution=res)
        return f"https://images.mapillary.com/{s}/thumb-{res}.jpg" if s else None
    except Exception:
        return None

def mapillary_viewer_link(pid):
    s = str(pid).strip()
    return s if s.startswith("http") else f"https://www.mapillary.com/app/?pKey={s}&focus=photo"


def popup_html(row, thumb_url: str | None = None) -> str:
    """Return a styled HTML card for Folium popup.
       Embeds data-pid / data-iid attributes for click selection."""
    pid = str(row.get("properties.id", ""))
    iid = str(row.get("image_id", ""))
    score = row.get("score")
    tract = row.get("geoid") 
    dt = row.get("datetime")  # pandas Timestamp or None
    when = pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else ""
    link = f"https://www.mapillary.com/app/?pKey={pid}&focus=photo"

    # plain-text payload to parse on click
    payload_txt = html.escape(json.dumps({"pid": pid}, separators=(",", ":")))
    hidden = f"<div style='display:none'>__PAYLOAD__{payload_txt}__END__</div>"

    # badges
    score_badge = f"<span style='background:#fee2e2;border:1px solid #fecaca;color:#991b1b;padding:1px 6px;border-radius:999px;font-size:11px;'>score: {score:.2f}</span>" if isinstance(score,(float,int)) else ""
    tract_badge = f"<span style='margin-left:6px;background:#e0e7ff;border:1px solid #c7d2fe;color:#3730a3;padding:1px 6px;border-radius:999px;font-size:11px;'>tract: {html.escape(str(tract))}</span>" if tract else ""
    time_line   = f"<div style='margin-top:6px;color:#6b7280;font-size:12px'>{when}</div>" if when else ""

    thumb_img = f"<img src='{html.escape(thumb_url)}' style='width:84px;height:84px;object-fit:cover;border-radius:8px;border:1px solid #ddd;'/>" if thumb_url else ""

    title = f"Photo {html.escape(pid[-8:] or pid)}"  # short id tail
    return f"""
        {hidden}
        <div class="payload" data-pid="{html.escape(pid)}" data-iid="{html.escape(iid)}"></div>
        <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;width:260px">
        <div style="display:flex;gap:8px">
            {thumb_img}
            <div style="flex:1">
            <div style="font-weight:600;font-size:14px">{title}</div>
            <div style="margin-top:4px">{score_badge}{tract_badge}</div>
            {time_line}
            <div style="margin-top:8px">
                <a href="{link}" target="_blank" style="text-decoration:none;font-size:12px">Open in Mapillary ↗</a>
            </div>
            </div>
        </div>
        </div>
        """.strip()

def us_flag_icon(height_px: int = 12, add_stars: bool = True,
                 stripe_red="#B22234", union_blue="#3C3B6E", star_white="#FFFFFF"):
    """
    Returns a folium.CustomIcon of a small U.S. flag.
    height_px: icon height in pixels (width auto = height * 1.9)
    add_stars: draw white dots in the canton (turn off for ultra-tiny icons)
    """

    # Encode colors for data URI (# -> %23)
    def enc(c): return c.replace("#", "%23")
    red, blue, white = enc(stripe_red), enc(union_blue), enc(star_white)

    # Official proportions (A=flag height)
    A = 130.0                 # arbitrary "units" for clean math
    B = 1.9 * A               # flag length
    C = 7/13 * A              # canton height (7 stripes)
    D = 0.76 * A              # canton length

    # Stripe height
    stripe_h = A / 13.0

    # Build stripes (top stripe is red)
    stripes = []
    for i in range(13):
        y = i * stripe_h
        color = red if i % 2 == 0 else white
        stripes.append(f'<rect x="0" y="{y:.3f}" width="{B:.3f}" height="{stripe_h:.3f}" fill="{color}"/>')

    # Canton (union)
    canton = f'<rect x="0" y="0" width="{D:.3f}" height="{C:.3f}" fill="{blue}"/>'

    # Stars (use tiny circles to keep SVG light/legible at small sizes)
    stars = []
    if add_stars:
        # 9 rows: 6,5,6,5,6,5,6,5,6
        row_count = 9
        row_spacing = C / (row_count + 1)             # evenly spaced vertically
        # 6-star rows
        col6 = 6
        col6_spacing = D / (col6 + 1)
        # 5-star rows
        col5 = 5
        col5_spacing = D / (col5 + 1)
        # star radius ~ 18% of min spacing
        star_r = 0.18 * min(row_spacing, col6_spacing)

        for r in range(row_count):
            y = (r + 1) * row_spacing
            if r % 2 == 0:  # 6-star row
                for k in range(col6):
                    x = (k + 1) * col6_spacing
                    stars.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{star_r:.3f}" fill="{white}"/>')
            else:           # 5-star row
                for k in range(col5):
                    x = (k + 1) * col5_spacing
                    stars.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{star_r:.3f}" fill="{white}"/>')

    # Compose SVG (transparent background)
    svg = (
        'data:image/svg+xml;utf8,'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {B:.3f} {A:.3f}">'
        + "".join(stripes) + canton + "".join(stars) +
        '</svg>'
    )

    # Icon pixel size (keep 10:19 ratio). Anchor near bottom center so it "points" nicely.
    h = int(height_px)
    w = int(round(h * 1.9))
    anchor = (w // 2, int(h * 0.85))

    return folium.CustomIcon(icon_image=svg, icon_size=(w, h), icon_anchor=anchor)

def add_detection_markers(map, df, icon_px = 10):
    for _, r in df.iterrows():
        card = popup_html(r, thumb_url=r.get("thumb_url"))
        icon = us_flag_icon(height_px=icon_px, add_stars=True)
        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            icon=icon,
            #icon_size=(100,100),
            popup=folium.Popup(card, max_width=160)
        ).add_to(map)

CENSUS_CMAP = linear.YlGnBu_06.scale(0, 100)
CENSUS_CMAP.caption = "Census share (%)"
def make_census_layer(df, value_col, name, cmap = CENSUS_CMAP):
    df["geoid"] = df["geoid"].astype(str).str.zfill(11)

    def style_fn(feat):
        v = feat["properties"].get(value_col, 0) or 0
        try: v = float(v)
        except: v = 0.0
        return {
            "fillColor": cmap(v),
            "color": "#888",
            "weight": 0.6,
            "fillOpacity": 0.7 if v > 0 else 0.4,
        }
    layer = folium.GeoJson(
        data=json.loads(df.to_json()),
        name=name, overlay=True, control=True, show=False,
        style_function=style_fn,
        pane = "polygons",
        highlight_function=lambda f: {"weight": 2, "color": "#000"},
    )
    folium.GeoJsonPopup(
        fields=["geoid", value_col],
        aliases=["GEOID", name],
        labels=False
    ).add_to(layer)
    return layer

def style_fn(feat, var):
    '''Style function for folium choropleth layer
       var: property name to color by (e.g. 'n_flags')
    '''
    geoid = feat["properties"]["geoid"]
    c = int(feat["properties"].get(var, 0))
    is_sel = (st.session_state.get("selected_geoid") and str(geoid) == str(st.session_state["selected_geoid"]))
    return {
        "fillColor": color_for(c),
        "color": "#111" if is_sel else "#555",
        "weight": 3 if is_sel else 0.7,
        "fillOpacity": 0.65,
        "opacity": 1.0,
    }

# color scale (quantiles)
def color_for(v):
    palette = PALETTE
    # map value into a bucket
    for i, t in enumerate(qs[1:]):  # compare to 20,40,60,80,100%
        if v <= t:
            return palette[i]
    return palette[-1]

def geoid_from_latlon(lat: float, lon: float) -> str | None:
    gdf = st.session_state.tract_gdf
    pt = Point(float(lon), float(lat))  # x=lon, y=lat
    try:
        cand_idx = list(gdf.sindex.query(pt, predicate="intersects"))
    except Exception:
        cand_idx = list(gdf.sindex.query(pt))
    if not cand_idx:
        return None
    cand = gdf.iloc[cand_idx]
    hit = cand.loc[cand.geometry.covers(pt)]   # covers = includes boundary clicks
    if not hit.empty:
        return str(hit.iloc[0]["geoid"])
    # nearest as last resort
    nearest = cand.distance(pt).sort_values().index[:1]
    return str(gdf.loc[nearest[0], "geoid"]) if len(nearest) else None


def nearest_marker_within(lat: float, lon: float, df: pd.DataFrame, max_meters: float = 40.0):
    pts = df.dropna(subset=["latitude","longitude"])
    if pts.empty:
        return None
    lat1 = np.deg2rad(lat)
    dlat_m = (pts["latitude"] - lat) * 111_132.0
    dlon_m = (pts["longitude"] - lon) * (111_320.0 * np.cos(lat1))
    d_m = np.sqrt(dlat_m**2 + dlon_m**2)
    idx = d_m.idxmin()
    if float(d_m.loc[idx]) <= max_meters:
        return pts.loc[idx]
    return None

def resolve_click(out, markers_df):
    """Return (geoid, marker_row) using lat/lon; popup text not required."""
    click_obj = out.get("last_object_clicked")
    click_ll  = out.get("last_clicked")
    lat = lon = None
    if click_obj and {"lat","lng"} <= set(click_obj.keys()):
        lat, lon = float(click_obj["lat"]), float(click_obj["lng"])
    elif click_ll and {"lat","lng"} <= set(click_ll.keys()):
        lat, lon = float(click_ll["lat"]), float(click_ll["lng"])
    if lat is None or lon is None:
        return (None, None)
    marker_row = nearest_marker_within(lat, lon, markers_df, max_meters=40.0)
    geoid = geoid_from_latlon(lat, lon)
    return (geoid, marker_row)

#%%
# -----------------------------
# Data loading
# -----------------------------
census_tracts = gpd.read_file(f"{DEFAULT_DATA_PATH}2020_Census_Tracts_20250901.geojson")
census_geo_df = gpd.read_file(f"{DEFAULT_DATA_PATH}census_geo_df.geojson")
detected_flags_tracts_geo = gpd.read_file(f"{DEFAULT_DATA_PATH}detected_flags_tracts_geo.geojson")
clean_voter_ed_df = gpd.read_file(f"{DEFAULT_DATA_PATH}nyc_voter_data.geojson")

# Cache a tiny copy + spatial index
if "tract_gdf" not in st.session_state:
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
        st.markdown(f"**GEOID {geoid_cur}**")
        st.metric("Detections in tract", int(flags_in_tract.shape[0]))

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
            st.image(url, width='stretch')
            st.caption(f"[Open in Mapillary ↗]({mapillary_viewer_link(pid)})")
        else:
            st.link_button("Open in Mapillary ↗", mapillary_viewer_link(pid), width='stretch')
    else:
        st.caption("Click a flag marker to preview its street view here.")
