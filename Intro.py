import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Flags App",
    page_icon=":us:",
)

st.write("# :us: Who flies the flag?")

st.sidebar.success("Start exploring by selecting a tab above.")

st.markdown(
    """
### 🏳️ Getting Started

This app explores where American flags tend to be displayed in New York City using street-level imagery and object detection models to locate flags across the five boroughs.

Flag display is not evenly distributed across the city, but instead reflects broader social, economic, and political patterns. Flags may be more common in neighborhoods with higher rates of homeownership, specific racial or ethnic compositions, or strong partisan affiliations. They may also cluster around public institutions or civic spaces. This analysis provides a window into how symbols of national identity—like the American flag—are woven into the urban fabric of New York City, and how their display might reflect deeper narratives about belonging, pride, and politics.


### 🗺️ What You'll Find in This App

#### Map Tab
Visualizes the geographic distribution of flag detections across the city. You can zoom into different neighborhoods, compare across boroughs, and see spatial patterns in civic symbolism.
""")

IMG = Path(__file__).parent / "static" / "ssmap.png"
st.image(IMG)

st.markdown(
    """
#### Analysis Tab
Shows how flag density correlates with neighborhood characteristics such as:
- Racial and ethnic composition
- Median household income
- Education levels
- Homeownership rates
- Voter registration

#### Methodology Tab

"""
)
