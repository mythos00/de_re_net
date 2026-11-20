import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide")
st.title("Folium xəritə testi")

m = folium.Map(
    location=[40.2, 47.8],
    zoom_start=7,
    tiles="OpenStreetMap",
    control_scale=True
)
folium.Marker([40.2, 47.8], tooltip="Test nöqtəsi").add_to(m)

st.write("Xəritə aşağıda görünməlidir:")
result = st_folium(m, width=1100, height=650, key="test_map")
st.write("Klik nəticəsi:", result.get("last_clicked") if result else None)
