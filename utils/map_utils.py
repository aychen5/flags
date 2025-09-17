import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np

import os, re, json
import requests
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


def get_mapillary_token():
    return (
        st.secrets.get("MLY_KEY")
        or os.getenv("MLY_KEY")         
    )

#@st.cache_data(show_spinner=False)
MLY_KEY = get_mapillary_token()
def mapillary_thumb_from_id(pid: str, res: int = 1024) -> str | None:
    s = str(pid).strip()          
    # was coerced to float like '12345.0', strip the .0
    if re.match(r"^\d+\.0$", s):
        return s[:-2]
    # try:
    #     # Mapillary photo key -> short-lived thumbnail URL
    #     #return mly.image_thumbnail(image_id=s, resolution=res)
    #     return f"https://images.mapillary.com/{s}/thumb-{res}.jpg" if s else None
    if not pid or not MLY_KEY:
        return None
    try:
        header = {'Authorization' : 'OAuth {}'.format(MLY_KEY)}
        url = 'https://graph.mapillary.com/{}?fields=thumb_1024_url'.format(s)
        r = requests.get(url, headers=header)
        data = r.json()
        image_url = data[f'thumb_{res}_url']
    except Exception:
        image_url = None
    return image_url

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
    time_badge = f"<span style='margin-left:6px;background:#f1f5f9;border:1px solid #e2e8f0;color:#334155;padding:1px 6px;border-radius:999px;font-size:11px;'>captured: {html.escape(when)}</span>" if when else ""
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
            <div style="margin-top:4px">{time_badge}</div>
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
        aliases=["Tract ID", name],
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
