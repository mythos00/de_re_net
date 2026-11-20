# -*- coding: utf-8 -*-
"""Single-page multi-index forest change analysis: DE / RE / Net Change."""
from pathlib import Path
import base64
from io import BytesIO
import json
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds, transform as rio_transform, reproject
from rasterio.features import geometry_mask
from pyproj import Transformer
from shapely.geometry import Point, Polygon, shape, mapping
from shapely.ops import transform as shp_transform, unary_union

import streamlit as st
import plotly.express as px
import folium
from folium.plugins import Fullscreen, MousePosition
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Forest Change | DE RE Net Change",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🌳 Forest Change Monitoring")
st.subheader("Deforestation • Reforestation • Net Change")
st.caption("Tək səhifəlik analiz: bütün 7 Sentinel-2 indeksinin kəsilməz modeli, xəritə, nöqtə, bufer və poliqon analizi.")

DEFAULT_ROOT = str(
    Path(__file__).parent / "feature_stack"
)
INDEX_BANDS = {"NDVI": 11, "EVI": 12, "NDMI": 13, "NBR": 14, "SAVI": 15, "BSI": 16, "NDWI": 17}
NODATA_FALLBACK = -9999.0
MAX_PREVIEW_SIDE = 1200

# Continuous normalization ranges (low_val -> high_val maps to 0.0 -> 1.0).
# For BSI and NDWI, low_val > high_val so lower values (less bare soil / less open water) score higher.
INDEX_CONFIG = {
    "NDVI": {"low": 0.20, "high": 0.75, "weight": 0.20, "delta_sign": +1.0},
    "EVI":  {"low": 0.15, "high": 0.55, "weight": 0.18, "delta_sign": +1.0},
    "NDMI": {"low": -0.10, "high": 0.35, "weight": 0.16, "delta_sign": +1.0},
    "NBR":  {"low": -0.10, "high": 0.45, "weight": 0.16, "delta_sign": +1.0},
    "SAVI": {"low": 0.15, "high": 0.55, "weight": 0.14, "delta_sign": +1.0},
    "BSI":  {"low": 0.25, "high": -0.10, "weight": 0.10, "delta_sign": -1.0},
    "NDWI": {"low": 0.40, "high": -0.10, "weight": 0.06, "delta_sign": -1.0},
}

CLASS_ORDER = ["Deforestation", "Reforestation", "Stabil meşə", "Stabil qeyri-meşə"]
CLASS_COLORS = {
    "Deforestation": "#dc2d2d",
    "Reforestation": "#1ea555",
    "Stabil meşə": "#2369be",
    "Stabil qeyri-meşə": "#919ba5",
}


@st.cache_data(show_spinner=False)
def find_stacks(root: str):
    p = Path(root)
    return [str(x) for x in sorted(p.rglob("*feature_stack.tif"))] if p.exists() else []


def valid(a, nodata):
    m = np.isfinite(a)
    if nodata is not None:
        m &= a != nodata
    m &= a != NODATA_FALLBACK
    return m


def scene_year(path: str):
    match = re.search(r"(20\d{2})\d{4}", Path(path).name)
    if match:
        return match.group(1)
    parent = Path(path).parent.name
    return parent if re.fullmatch(r"20\d{2}", parent) else parent


def parse_geojson_bytes(raw_bytes: bytes):
    obj = json.loads(raw_bytes.decode("utf-8-sig"))
    crs = obj.get("crs") or "EPSG:4326"
    if isinstance(crs, dict):
        crs = crs.get("properties", {}).get("name", "EPSG:4326")
    if obj.get("type") == "FeatureCollection":
        geoms = [f["geometry"] for f in obj.get("features", []) if f.get("geometry")]
    elif obj.get("type") == "Feature":
        geoms = [obj["geometry"]]
    else:
        geoms = [obj]
    if not geoms:
        raise ValueError("GeoJSON daxilində poliqon tapılmadı.")
    return geoms, crs


def _parse_kml_coords(text: str):
    coords = []
    for token in text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            coords.append((lon, lat))
    return coords


def parse_kml_bytes(kml_bytes: bytes):
    root_xml = ET.fromstring(kml_bytes)
    geoms = []
    for elem in root_xml.iter():
        if elem.tag.endswith("Polygon"):
            outer_coords = None
            inner_rings = []
            for child in elem:
                if child.tag.endswith("outerBoundaryIs"):
                    for c in child.iter():
                        if c.tag.endswith("coordinates") and c.text:
                            ring = _parse_kml_coords(c.text)
                            if len(ring) >= 3:
                                outer_coords = ring
                elif child.tag.endswith("innerBoundaryIs"):
                    for c in child.iter():
                        if c.tag.endswith("coordinates") and c.text:
                            ring = _parse_kml_coords(c.text)
                            if len(ring) >= 3:
                                inner_rings.append(ring)
            if outer_coords:
                poly = Polygon(outer_coords, holes=inner_rings if inner_rings else None)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    geoms.append(mapping(poly))
    if not geoms:
        raise ValueError("KML/KMZ faylı daxilində poliqon tapılmadı.")
    return geoms, "EPSG:4326"


def parse_polygon_file(upload):
    suffix = Path(upload.name).suffix.lower()
    raw = upload.getvalue()
    if suffix in (".geojson", ".json"):
        return parse_geojson_bytes(raw)
    if suffix == ".kml":
        return parse_kml_bytes(raw)
    if suffix == ".kmz":
        with zipfile.ZipFile(BytesIO(raw), "r") as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("KMZ arxivi daxilində .kml faylı tapılmadı.")
            preferred = "doc.kml" if "doc.kml" in kml_names else kml_names[0]
            return parse_kml_bytes(zf.read(preferred))
    raise ValueError("Dəstəklənən formatlar: GeoJSON, KML, KMZ.")


def project_geometries(geoms, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return [mapping(shp_transform(transformer.transform, shape(g))) for g in geoms]


def geometries_centroid_latlon(geoms_wgs84):
    shapes = [shape(g) for g in geoms_wgs84 if g]
    if not shapes:
        return None
    merged = unary_union(shapes)
    c = merged.centroid if not merged.is_empty else shapes[0].centroid
    if c.is_empty:
        return None
    return float(c.y), float(c.x)


def build_point_buffer_geom(lat: float, lon: float, radius_m: float, dst_crs):
    to_dst = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    px, py = to_dst.transform(lon, lat)
    buf_dst = Point(px, py).buffer(float(radius_m), resolution=48)
    to_wgs = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True)
    buf_wgs = shp_transform(to_wgs.transform, buf_dst)
    return mapping(buf_dst), mapping(buf_wgs)


def parse_coordinate(text: str):
    if not text or not text.strip():
        return None
    parts = [x.strip() for x in text.replace(";", ",").split(",")]
    if len(parts) != 2:
        raise ValueError("Format belə olmalıdır: Latitude, Longitude")
    lat, lon = float(parts[0]), float(parts[1])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Koordinat diapazondan kənardadır.")
    return lat, lon


def normalize_01(val, low: float, high: float):
    denom = high - low
    if denom == 0:
        return np.zeros_like(val, dtype="float32") if isinstance(val, np.ndarray) else 0.0
    if isinstance(val, np.ndarray):
        return np.clip((val - low) / denom, 0.0, 1.0).astype("float32")
    return float(np.clip((val - low) / denom, 0.0, 1.0))


def continuous_forest_score_from_dict(vals: dict) -> float:
    total_w = sum(cfg["weight"] for cfg in INDEX_CONFIG.values())
    acc = 0.0
    for name, cfg in INDEX_CONFIG.items():
        n = normalize_01(float(vals[name]), cfg["low"], cfg["high"])
        acc += cfg["weight"] * n
    return float(acc / total_w) if total_w > 0 else 0.0


def continuous_forest_score_from_bands(bands: dict, shape) -> np.ndarray:
    total_w = sum(cfg["weight"] for cfg in INDEX_CONFIG.values())
    acc = np.zeros(shape, dtype="float32")
    for name, cfg in INDEX_CONFIG.items():
        n = normalize_01(bands[name], cfg["low"], cfg["high"])
        acc += cfg["weight"] * n
    return (acc / total_w).astype("float32") if total_w > 0 else acc


def continuous_delta_signal_from_bands(bands1: dict, bands2: dict, shape) -> np.ndarray:
    """Bütün 7 indeksin dəyişməsindən (Δ) yaranan ümumi meşə dəyişimi siqnalı."""
    total_w = sum(cfg["weight"] for cfg in INDEX_CONFIG.values())
    sig = np.zeros(shape, dtype="float32")
    for name, cfg in INDEX_CONFIG.items():
        d = (bands2[name] - bands1[name]) * cfg["delta_sign"]
        # Normalize roughly by expected range width
        span = max(0.05, abs(cfg["high"] - cfg["low"]))
        sig += cfg["weight"] * np.clip(d / span, -1.5, 1.5).astype("float32")
    return (sig / total_w).astype("float32") if total_w > 0 else sig


def continuous_delta_signal_from_dicts(before_vals: dict, after_vals: dict) -> float:
    total_w = sum(cfg["weight"] for cfg in INDEX_CONFIG.values())
    sig = 0.0
    for name, cfg in INDEX_CONFIG.items():
        d = (float(after_vals[name]) - float(before_vals[name])) * cfg["delta_sign"]
        span = max(0.05, abs(cfg["high"] - cfg["low"]))
        sig += cfg["weight"] * float(np.clip(d / span, -1.5, 1.5))
    return float(sig / total_w) if total_w > 0 else 0.0


@st.cache_data(show_spinner=False)
def read_preview_stack(path: str, max_side: int = MAX_PREVIEW_SIDE):
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        scale = min(1.0, max_side / max(h, w))
        oh, ow = max(1, int(h * scale)), max(1, int(w * scale))
        bands = {}
        common_valid = None
        for name, band in INDEX_BANDS.items():
            arr = ds.read(band, out_shape=(oh, ow), resampling=Resampling.average).astype("float32")
            mask = valid(arr, ds.nodatavals[band - 1])
            common_valid = mask if common_valid is None else common_valid & mask
            bands[name] = arr
        transform = ds.transform * ds.transform.scale(w / ow, h / oh)
        bounds = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
        crs = ds.crs
        source_pixel_area_ha = abs(ds.transform.a * ds.transform.e - ds.transform.b * ds.transform.d) / 10000.0
        preview_factor = (w / ow) * (h / oh)
    return bands, common_valid, transform, crs, bounds, source_pixel_area_ha * preview_factor


def aligned_bands_and_scores(path: str, reference_shape=None, reference_transform=None, reference_crs=None):
    bands, mask, transform, crs, bounds, pixel_area = read_preview_stack(path)
    score = continuous_forest_score_from_bands(bands, bands["NDVI"].shape)
    score[~mask] = np.nan

    if reference_shape is None:
        return bands, score, mask, transform, crs, bounds, pixel_area

    if score.shape == reference_shape and transform == reference_transform and crs == reference_crs:
        return bands, score, mask, transform, crs, bounds, pixel_area

    aligned_bands = {}
    for name, arr in bands.items():
        arr_masked = np.where(mask, arr, np.nan).astype("float32")
        dst_b = np.full(reference_shape, np.nan, dtype="float32")
        reproject(
            source=arr_masked,
            destination=dst_b,
            src_transform=transform,
            src_crs=crs,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        aligned_bands[name] = dst_b

    dst_s = np.full(reference_shape, np.nan, dtype="float32")
    reproject(
        source=score,
        destination=dst_s,
        src_transform=transform,
        src_crs=crs,
        dst_transform=reference_transform,
        dst_crs=reference_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    dst_mask = np.isfinite(dst_s)
    return aligned_bands, dst_s, dst_mask, reference_transform, reference_crs, bounds, pixel_area


def mean_indices_in_mask(bands1, bands2, mask):
    if not np.any(mask):
        return {}, {}
    m1, m2 = {}, {}
    for name in INDEX_BANDS:
        v1 = bands1[name][mask]
        v2 = bands2[name][mask]
        v1 = v1[np.isfinite(v1)]
        v2 = v2[np.isfinite(v2)]
        m1[name] = float(np.mean(v1)) if len(v1) > 0 else None
        m2[name] = float(np.mean(v2)) if len(v2) > 0 else None
    return m1, m2


def compute_change_stats(score1, score2, delta_sig, mask, threshold, pixel_area_ha):
    # All 7 indices contribute continuously to score1, score2, and delta_sig.
    forest1 = score1 >= threshold
    forest2 = score2 >= threshold
    loss = mask & forest1 & ~forest2 & (delta_sig <= 0.05)
    gain = mask & ~forest1 & forest2 & (delta_sig >= -0.05)
    stable_forest = mask & forest1 & forest2 & ~loss & ~gain
    stable_nonforest = mask & ~forest1 & ~forest2 & ~loss & ~gain

    de_ha = float(loss.sum() * pixel_area_ha)
    re_ha = float(gain.sum() * pixel_area_ha)
    sf_ha = float(stable_forest.sum() * pixel_area_ha)
    snf_ha = float(stable_nonforest.sum() * pixel_area_ha)
    net_ha = re_ha - de_ha

    df = pd.DataFrame({
        "Sinif": ["Deforestation", "Reforestation", "Stabil meşə", "Stabil qeyri-meşə"],
        "Sahə (ha)": [de_ha, re_ha, sf_ha, snf_ha],
    })
    total = df["Sahə (ha)"].sum()
    df["Pay (%)"] = np.where(total > 0, df["Sahə (ha)"] / total * 100.0, 0.0)

    classes = np.full(score1.shape, -1, dtype=np.int8)
    classes[stable_nonforest] = 0
    classes[loss] = 1
    classes[gain] = 2
    classes[stable_forest] = 3

    return {
        "de_ha": de_ha,
        "re_ha": re_ha,
        "sf_ha": sf_ha,
        "snf_ha": snf_ha,
        "net_ha": net_ha,
        "df": df,
        "classes": classes,
    }


def classify_point_change(before_vals: dict, after_vals: dict, threshold: float):
    if any(before_vals.get(k) is None or after_vals.get(k) is None for k in INDEX_BANDS):
        return None, None, None, None
    s1 = continuous_forest_score_from_dict(before_vals)
    s2 = continuous_forest_score_from_dict(after_vals)
    d_sig = continuous_delta_signal_from_dicts(before_vals, after_vals)
    f1 = s1 >= threshold
    f2 = s2 >= threshold
    if f1 and (not f2) and (d_sig <= 0.05):
        label = "Deforestation"
    elif (not f1) and f2 and (d_sig >= -0.05):
        label = "Reforestation"
    elif f1 and f2:
        label = "Stabil meşə"
    else:
        label = "Stabil qeyri-meşə"
    return label, s1, s2, d_sig


def rgba_uint8_array(class_array: np.ndarray) -> np.ndarray:
    palette = {
        0: (145, 155, 165, 110),
        1: (220, 45, 45, 220),
        2: (30, 165, 85, 220),
        3: (35, 105, 190, 150),
    }
    rgba = np.zeros((*class_array.shape, 4), dtype=np.uint8)
    for cls, color in palette.items():
        rgba[class_array == cls] = color
    return rgba


def rgba_to_data_uri(rgba_u8: np.ndarray) -> str:
    img = Image.fromarray(rgba_u8, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def point_values(path: str, lon: float, lat: float):
    values = {}
    with rasterio.open(path) as ds:
        x, y = rio_transform("EPSG:4326", ds.crs, [lon], [lat])
        row, col = ds.index(x[0], y[0])
        if not (0 <= row < ds.height and 0 <= col < ds.width):
            return values
        for name, band in INDEX_BANDS.items():
            v = float(ds.read(band, window=((row, row + 1), (col, col + 1)))[0, 0])
            nd = ds.nodatavals[band - 1]
            values[name] = v if valid(np.array([v]), nd)[0] else None
    return values


def make_change_map(rgba_u8: np.ndarray, bounds, marker_point=None, polygon_wgs84=None, buffer_wgs84=None):
    south, west, north, east = bounds[1], bounds[0], bounds[3], bounds[2]
    center = list(marker_point) if marker_point else [(south + north) / 2, (west + east) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[south, west], [north, east]], padding=(20, 20))

    folium.raster_layers.ImageOverlay(
        image=rgba_to_data_uri(rgba_u8),
        bounds=[[south, west], [north, east]],
        opacity=1.0,
        interactive=True,
        origin="upper",
        name="DE / RE / Net Change",
    ).add_to(m)

    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#ff7800",
        weight=2,
        fill=False,
        tooltip="Sentinel-2 raster coverage",
    ).add_to(m)

    if polygon_wgs84:
        fc = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": g, "properties": {}} for g in polygon_wgs84],
        }
        folium.GeoJson(
            fc,
            name="Yüklənmiş Poliqon",
            style_function=lambda _: {
                "color": "#ffd000",
                "weight": 3,
                "fillColor": "#ffd000",
                "fillOpacity": 0.14,
            },
        ).add_to(m)

    if buffer_wgs84:
        buf_fc = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": buffer_wgs84, "properties": {}}],
        }
        folium.GeoJson(
            buf_fc,
            name="Nöqtə Buferi",
            style_function=lambda _: {
                "color": "#00e5ff",
                "weight": 2.5,
                "dashArray": "6, 4",
                "fillColor": "#00e5ff",
                "fillOpacity": 0.12,
            },
        ).add_to(m)

    if marker_point:
        folium.Marker(
            location=list(marker_point),
            tooltip="Seçilmiş / mərkəz nöqtə",
            icon=folium.Icon(color="red"),
        ).add_to(m)

    Fullscreen().add_to(m)
    MousePosition(position="bottomleft", separator=" | ", prefix="Lat/Lon:").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_class_charts(df: pd.DataFrame, title_prefix: str, key_suffix: str):
    col_bar, col_pie = st.columns([1.35, 1.0])
    with col_bar:
        fig_bar = px.bar(
            df,
            x="Sinif",
            y="Sahə (ha)",
            color="Sinif",
            text_auto=".2f",
            category_orders={"Sinif": CLASS_ORDER},
            color_discrete_map=CLASS_COLORS,
            title=f"{title_prefix} — Sahə (ha)",
        )
        fig_bar.update_layout(showlegend=False, height=360, margin=dict(t=45, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{key_suffix}")

    with col_pie:
        fig_pie = px.pie(
            df,
            names="Sinif",
            values="Sahə (ha)",
            hole=0.45,
            category_orders={"Sinif": CLASS_ORDER},
            color="Sinif",
            color_discrete_map=CLASS_COLORS,
            title=f"{title_prefix} — Pay bölgüsü (%)",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, height=360, margin=dict(t=45, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{key_suffix}")


def render_index_comparison_charts(before_vals: dict, after_vals: dict, base_yr: str, target_yr: str, title_label: str, key_prefix: str):
    rows = []
    long_rows = []
    for name in INDEX_BANDS:
        v1, v2 = before_vals.get(name), after_vals.get(name)
        diff = (v2 - v1) if (v1 is not None and v2 is not None) else None
        rows.append({
            "İndeks": name,
            "Çəki": INDEX_CONFIG[name]["weight"],
            f"{base_yr}": v1,
            f"{target_yr}": v2,
            "Fərq (Δ)": diff,
        })
        if v1 is not None:
            long_rows.append({"İndeks": name, "İl": str(base_yr), "Dəyər": v1})
        if v2 is not None:
            long_rows.append({"İndeks": name, "İl": str(target_yr), "Dəyər": v2})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if long_rows:
        c_left, c_right = st.columns([1.3, 1.0])
        with c_left:
            fig_idx = px.bar(
                pd.DataFrame(long_rows),
                x="İndeks",
                y="Dəyər",
                color="İl",
                barmode="group",
                title=f"{title_label} — 7 indeks müqayisəsi ({base_yr} vs {target_yr})",
            )
            fig_idx.update_layout(height=350, margin=dict(t=45, b=20, l=20, r=20))
            st.plotly_chart(fig_idx, use_container_width=True, key=f"{key_prefix}_bar")

        with c_right:
            diff_df = pd.DataFrame([
                {"İndeks": r["İndeks"], "Fərq (Δ)": r["Fərq (Δ)"]}
                for r in rows
                if r["Fərq (Δ)"] is not None
            ])
            if not diff_df.empty:
                fig_diff = px.bar(
                    diff_df,
                    x="İndeks",
                    y="Fərq (Δ)",
                    color="Fərq (Δ)",
                    color_continuous_scale=["#dc2d2d", "#e5e7eb", "#1ea555"],
                    color_continuous_midpoint=0.0,
                    title=f"{title_label} — İndeks dəyişimi (Δ = Son - Başlanğıc)",
                )
                fig_diff.update_layout(height=350, margin=dict(t=45, b=20, l=20, r=20))
                st.plotly_chart(fig_diff, use_container_width=True, key=f"{key_prefix}_diff")


# ---- Sidebar controls ----
with st.sidebar:
    st.header("Məlumat və müqayisə")
    root = st.text_input("Feature Stack qovluğu", DEFAULT_ROOT)
    if st.button("Rasterləri yenilə"):
        find_stacks.clear()
        read_preview_stack.clear()
        st.rerun()

    stack_files = find_stacks(root)
    if not stack_files:
        st.error("Feature Stack TIFF tapılmadı. Qovluq yolunu yoxla.")
        st.stop()

    available_years = sorted(set(scene_year(p) for p in stack_files))
    if len(available_years) < 2:
        st.error("DE / RE hesablaması üçün ən azı iki ilin Feature Stack faylı lazımdır.")
        st.stop()

    base_year = st.selectbox("Başlanğıc il", available_years, index=0)
    target_year = st.selectbox("Müqayisə ili", available_years, index=len(available_years) - 1)
    base_files = [p for p in stack_files if scene_year(p) == base_year]
    target_files = [p for p in stack_files if scene_year(p) == target_year]
    base_path = st.selectbox("Başlanğıc raster", base_files, format_func=lambda p: Path(p).name)
    target_path = st.selectbox("Son raster", target_files, format_func=lambda p: Path(p).name)
    threshold = st.slider("Meşə uyğunluq həddi (0–1)", 0.0, 1.0, 0.55, 0.01)
    buffer_radius_m = st.slider("Nöqtə bufer radiusu (metr)", 50, 5000, 500, 50)
    show_buffer_on_map = st.checkbox("Xəritədə bufer dairəsini göstər", value=True)
    map_renderer = st.radio("Xəritə mühərriki", ["Folium (Xəritə üzərində)", "Plotly (Rastr baxışı)"], index=0)
    st.caption("7 indeks: NDVI (0.20), EVI (0.18), NDMI (0.16), NBR (0.16), SAVI (0.14), BSI (0.10), NDWI (0.06).")

c_inp1, c_inp2 = st.columns([1.2, 1.0])
with c_inp1:
    coord_input = st.text_input("Koordinat (Latitude, Longitude)", placeholder="40.123456, 48.654321")
with c_inp2:
    polygon_file = st.file_uploader("Poliqon faylı yüklə (GeoJSON, KML, KMZ)", type=["geojson", "json", "kml", "kmz"])

coord = None
if coord_input.strip():
    try:
        coord = parse_coordinate(coord_input)
    except Exception as exc:
        st.error(f"Koordinat xətası: {exc}")

uploaded_geometries = None
uploaded_crs = "EPSG:4326"
polygon_wgs84 = None
polygon_centroid = None
if polygon_file is not None:
    try:
        uploaded_geometries, uploaded_crs = parse_polygon_file(polygon_file)
        polygon_wgs84 = project_geometries(uploaded_geometries, uploaded_crs, "EPSG:4326")
        polygon_centroid = geometries_centroid_latlon(polygon_wgs84)
        st.success(f"Poliqon yükləndi: **{polygon_file.name}** ({len(uploaded_geometries)} obyekt)")
    except Exception as exc:
        st.error(f"Poliqon faylı xətası: {exc}")

# ---- Compute continuous 7-index forest change ----
with st.spinner("Bütün 7 indeks kəsilməz olaraq oxunur və müqayisə hazırlanır..."):
    bands1, score1, mask1, transform1, crs1, bounds1, pixel_area_ha = aligned_bands_and_scores(base_path)
    bands2, score2, mask2, _, _, _, _ = aligned_bands_and_scores(
        target_path,
        reference_shape=score1.shape,
        reference_transform=transform1,
        reference_crs=crs1,
    )
    delta_sig = continuous_delta_signal_from_bands(bands1, bands2, score1.shape)
    base_valid = np.isfinite(score1) & np.isfinite(score2) & np.isfinite(delta_sig)
    overall_stats = compute_change_stats(score1, score2, delta_sig, base_valid, threshold, pixel_area_ha)
    has_polygon_filter = bool(uploaded_geometries)

# ---- Interactive map ----
st.subheader("İnteraktiv dəyişiklik xəritəsi")
st.caption("Xəritə rasterin coğrafi sərhədlərinə uyğun açılır. Narıncı çərçivə analiz sahəsidir.")
st.markdown("🟥 Deforestation &nbsp;&nbsp; 🟩 Reforestation &nbsp;&nbsp; 🟦 Stabil meşə &nbsp;&nbsp; ⬜ Stabil qeyri-meşə")

marker_point = coord if coord is not None else polygon_centroid

pre_buf_dst, pre_buf_wgs84 = (None, None)
if marker_point is not None and show_buffer_on_map:
    pre_buf_dst, pre_buf_wgs84 = build_point_buffer_geom(marker_point[0], marker_point[1], buffer_radius_m, crs1)

pre_mask = base_valid.copy()
if has_polygon_filter:
    projected_poly = project_geometries(uploaded_geometries, uploaded_crs, crs1)
    pre_mask &= geometry_mask(projected_poly, out_shape=score1.shape, transform=transform1, invert=True)

pre_stats = compute_change_stats(score1, score2, delta_sig, pre_mask, threshold, pixel_area_ha)
overlay_u8 = rgba_uint8_array(pre_stats["classes"])

clicked_coord = None
if map_renderer.startswith("Folium"):
    map_obj = make_change_map(
        overlay_u8,
        bounds1,
        marker_point=marker_point,
        polygon_wgs84=polygon_wgs84,
        buffer_wgs84=pre_buf_wgs84 if show_buffer_on_map else None,
    )
    map_state = st_folium(
        map_obj,
        use_container_width=True,
        height=620,
        key=f"map_{Path(base_path).name}_{Path(target_path).name}_{threshold:.2f}_{polygon_file.name if polygon_file else 'none'}_{buffer_radius_m}_{show_buffer_on_map}_{marker_point}",
        returned_objects=["last_clicked"],
    )
    clicked = map_state.get("last_clicked") if isinstance(map_state, dict) else None
    if clicked and "lat" in clicked and "lng" in clicked:
        clicked_coord = (float(clicked["lat"]), float(clicked["lng"]))
else:
    fig_map = px.imshow(overlay_u8, origin="upper", title="Dəyişiklik rastrı (Preview)")
    fig_map.update_layout(height=620)
    st.plotly_chart(fig_map, use_container_width=True, key="plotly_raster_map")

active_coord = coord if coord is not None else (clicked_coord if clicked_coord is not None else polygon_centroid)

buf_dst_geom = None
buf_wgs84_geom = None
buffer_stats = None
if active_coord is not None:
    buf_dst_geom, buf_wgs84_geom = build_point_buffer_geom(
        active_coord[0], active_coord[1], buffer_radius_m, crs1
    )
    buf_mask = base_valid & geometry_mask([buf_dst_geom], out_shape=score1.shape, transform=transform1, invert=True)
    buffer_stats = compute_change_stats(score1, score2, delta_sig, buf_mask, threshold, pixel_area_ha)

selected_valid = base_valid.copy()
if has_polygon_filter:
    projected_poly = project_geometries(uploaded_geometries, uploaded_crs, crs1)
    selected_valid &= geometry_mask(projected_poly, out_shape=score1.shape, transform=transform1, invert=True)
    scope_title = f"Seçilmiş poliqon: {polygon_file.name}"
elif coord is not None or clicked_coord is not None:
    selected_valid &= geometry_mask([buf_dst_geom], out_shape=score1.shape, transform=transform1, invert=True)
    scope_title = f"Nöqtə buferi ({buffer_radius_m} m): {active_coord[0]:.5f}, {active_coord[1]:.5f}"
else:
    scope_title = "Bütün rastr sahəsi"

selected_stats = (
    compute_change_stats(score1, score2, delta_sig, selected_valid, threshold, pixel_area_ha)
    if (has_polygon_filter or coord is not None or clicked_coord is not None)
    else overall_stats
)

# ---- Active / Selected scope KPIs & charts ----
st.markdown(f"### Meşə dəyişikliklərinin nəticəsi — **{scope_title}**")
c1, c2, c3 = st.columns(3)
c1.metric("Deforestation (DE)", f"{selected_stats['de_ha']:,.2f} ha")
c2.metric("Reforestation (RE)", f"{selected_stats['re_ha']:,.2f} ha")
c3.metric("Net Change", f"{selected_stats['net_ha']:+,.2f} ha")

build_class_charts(
    selected_stats["df"],
    title_prefix=scope_title,
    key_suffix="selected",
)

if has_polygon_filter:
    poly_mean1, poly_mean2 = mean_indices_in_mask(bands1, bands2, selected_valid)
    if poly_mean1 and poly_mean2:
        st.markdown(f"#### Poliqon daxilində 7 indeksin orta dəyərləri ({len(uploaded_geometries)} obyekt)")
        render_index_comparison_charts(
            poly_mean1,
            poly_mean2,
            base_year,
            target_year,
            title_label="Poliqon ortalaması",
            key_prefix="poly_mean_indices",
        )

# ---- Selected point / centroid + buffer forest change & 7 indices ----
st.divider()
st.markdown("### Nöqtə və bufer ətrafı üzrə meşə dəyişimi (7 indeks)")

if active_coord:
    lat, lon = active_coord
    if coord is not None:
        coord_source = "Daxil edilmiş koordinat"
    elif clicked_coord is not None:
        coord_source = "Xəritədən kliklənmiş nöqtə"
    else:
        coord_source = "Poliqon mərkəz nöqtəsi (avtomatik)"

    st.caption(f"{coord_source}: **{lat:.7f}, {lon:.7f}** • Bufer radiusu: **{buffer_radius_m} m**")

    before = point_values(base_path, lon, lat)
    after = point_values(target_path, lon, lat)
    pt_class, pt_s1, pt_s2, pt_d_sig = classify_point_change(before, after, threshold)

    if pt_class is not None:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Nöqtə statusu (piksel)", pt_class)
        p2.metric(f"7-İndeks Skoru ({base_year})", f"{pt_s1:.3f}")
        p3.metric(f"7-İndeks Skoru ({target_year})", f"{pt_s2:.3f} ({pt_s2 - pt_s1:+.3f})")
        p4.metric("7-İndeks Δ Siqnalı", f"{pt_d_sig:+.3f}")

    if buffer_stats is not None:
        st.markdown(f"#### Bufer daxilində dəyişiklik sahəsi ({buffer_radius_m} m radius)")
        b1, b2, b3 = st.columns(3)
        b1.metric("Bufer DE", f"{buffer_stats['de_ha']:,.2f} ha")
        b2.metric("Bufer RE", f"{buffer_stats['re_ha']:,.2f} ha")
        b3.metric("Bufer Net Change", f"{buffer_stats['net_ha']:+,.2f} ha")
        build_class_charts(
            buffer_stats["df"],
            title_prefix=f"Bufer ({buffer_radius_m} m)",
            key_suffix="buffer_point",
        )

    render_index_comparison_charts(
        before,
        after,
        base_year,
        target_year,
        title_label="Seçilmiş nöqtə",
        key_prefix="point_7_indices",
    )
else:
    st.info("Nöqtə üzrə meşə dəyişimini və 7 indeks qrafiklərini görmək üçün koordinat daxil et, poliqon yüklə və ya xəritədə bir nöqtəyə kliklə.")

# ---- Export ----
export_df = pd.DataFrame([{
    "Başlanğıc il": base_year,
    "Müqayisə ili": target_year,
    "Başlanğıc raster": Path(base_path).name,
    "Son raster": Path(target_path).name,
    "Analiz_sahəsi": scope_title,
    "DE_ha_preview_estimate": selected_stats["de_ha"],
    "RE_ha_preview_estimate": selected_stats["re_ha"],
    "Net_Change_ha_preview_estimate": selected_stats["net_ha"],
    "Buffer_radius_m": buffer_radius_m if active_coord else None,
    "Buffer_DE_ha": buffer_stats["de_ha"] if buffer_stats else None,
    "Buffer_RE_ha": buffer_stats["re_ha"] if buffer_stats else None,
    "Buffer_Net_Change_ha": buffer_stats["net_ha"] if buffer_stats else None,
    "Forest_score_threshold": threshold,
    "Polygon_filter": has_polygon_filter,
    "Polygon_file": polygon_file.name if polygon_file else "",
    "Selected_lat": active_coord[0] if active_coord else None,
    "Selected_lon": active_coord[1] if active_coord else None,
}])
st.download_button(
    "Nəticələri CSV yüklə",
    export_df.to_csv(index=False).encode("utf-8-sig"),
    "forest_change_DE_RE_net_change.csv",
    "text/csv",
)

st.warning(
    "Metodoloji qeyd: DE/RE 7 indeksdən qurulmuş kəsilməz meşə uyğunluq skorunun və indeks dəyişiminin (Δ) birləşməsidir; "
    "bu, GFW-nin rəsmi alqoritmi deyil. Hektar göstəriciləri sürətli preview rasterindən təxmin edilir. "
    "Rəsmi hesabat üçün tam ölçülü raster, bulud/kölgə maskası, eyni mövsüm kompozitləri və sahəvi validasiya lazımdır."
)