import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mapclassify
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.patheffects import Stroke, Normal
import os
import requests
import zipfile
import io
import numpy as np

# ------------------------------------------------------------
# PAGE AND FONT SETTINGS
# ------------------------------------------------------------
st.set_page_config(page_title="UK Regional Company Map", layout="wide")

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams["font.weight"] = "regular"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["axes.labelweight"] = "regular"

# ------------------------------------------------------------
# GOOGLE DRIVE CONFIG
# ------------------------------------------------------------
GDRIVE_FILE_ID = "1ip-Aip_rQNucgdJRvIBckSnYa_RBcRFU"
SHAPEFILE_DIR = "shapefile_data"
SHAPEFILE_NAME = "NUTS_Level_1__January_2018__Boundaries.shp"

@st.cache_resource
def download_shapefile():
    """Download shapefile from Google Drive if not already present"""
    shapefile_path = os.path.join(SHAPEFILE_DIR, SHAPEFILE_NAME)
    if os.path.exists(shapefile_path):
        return shapefile_path

    os.makedirs(SHAPEFILE_DIR, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

    with st.spinner("Downloading shapefile from Google Drive..."):
        r = requests.get(url)
        if r.status_code == 200:
            zip_path = os.path.join(SHAPEFILE_DIR, "shapefile.zip")
            with open(zip_path, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(SHAPEFILE_DIR)
            os.remove(zip_path)
            return shapefile_path
        else:
            st.error(f"Failed to download shapefile. Status code: {r.status_code}")
            st.error("Make sure the Google Drive file is publicly accessible.")
            return None

# ------------------------------------------------------------
# APP CONTENT
# ------------------------------------------------------------
st.title("UK Regional Company Distribution Map")

st.write("""
Upload a CSV or Excel file containing:
- **Head Office Address - Region**
- **Registered Address - Region**
""")

# Download shapefile (only once)
shapefile_path = download_shapefile()
if shapefile_path is None:
    st.stop()

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.stop()

ext = uploaded_file.name.split('.')[-1].lower()
if ext == "csv":
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

required_cols = ["Head Office Address - Region", "Registered Address - Region"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ------------------------------------------------------------
# CLEAN & MERGE REGIONS
# ------------------------------------------------------------
for c in required_cols:
    df[c] = (
        df[c]
        .astype(str)
        .str.strip()
        .replace({"nan": np.nan, "(no value)": np.nan, "": np.nan, "None": np.nan})
    )

df["Region (merged)"] = df["Head Office Address - Region"].fillna(
    df["Registered Address - Region"]
)

# ------------------------------------------------------------
# REGION MAPPING (group Scotland subregions)
# ------------------------------------------------------------
region_mapping = {
    "East Midlands": "East Midlands (England)",
    "East of England": "East of England",
    "London": "London",
    "North East": "North East (England)",
    "North West": "North West (England)",
    "Northern Ireland": "Northern Ireland",
    "Scotland": "Scotland",
    "South East": "South East (England)",
    "South West": "South West (England)",
    "Wales": "Wales",
    "West Midlands": "West Midlands (England)",
    "Yorkshire and The Humber": "Yorkshire and The Humber",
    # Scotland subregions → Scotland
    "West of Scotland": "Scotland",
    "East of Scotland": "Scotland",
    "South of Scotland": "Scotland",
    "Highlands and Islands": "Scotland",
    "Tayside": "Scotland",
    "Aberdeen": "Scotland",
}

df["Region_Mapped"] = df["Region (merged)"].map(region_mapping).fillna("Unknown")

# Counts for mapping (exclude Unknown from the map)
region_counts = (
    df[df["Region_Mapped"] != "Unknown"]
    .groupby("Region_Mapped")
    .size()
    .reset_index(name="Company_Count")
)

# ------------------------------------------------------------
# LOAD SHAPEFILE & MERGE
# ------------------------------------------------------------
os.environ["SHAPE_RESTORE_SHX"] = "YES"
gdf = gpd.read_file(shapefile_path)
g = gdf.merge(region_counts, left_on="nuts118nm", right_on="Region_Mapped", how="left")
g["Company_Count"] = g["Company_Count"].fillna(0)

# ------------------------------------------------------------
# SMART 1–2–5 BINNING
# ------------------------------------------------------------
colors = ["#B5E7F4", "#90DBEF", "#74D1EA", "#4BB5CF", "#2B8EAA"]  # your palette

def nice_bins(vals, target=5):
    """Return UserDefined upper-bound bins using a 1–2–5 progression."""
    arr = np.asarray(vals)
    pos = arr[arr > 0]
    if len(pos) == 0:
        return [0, np.inf]
    lo, hi = float(pos.min()), float(pos.max())
    lo_e = int(np.floor(np.log10(lo))) - 1
    hi_e = int(np.ceil(np.log10(hi))) + 1
    candidates = []
    for e in range(lo_e, hi_e + 1):
        for m in (1, 2, 5):
            v = m * (10 ** e)
            if lo <= v <= hi:
                candidates.append(v)
    candidates.sort()
    if len(candidates) <= (target - 1):
        picks = candidates
    else:
        step = int(np.ceil(len(candidates) / (target - 1)))
        picks = candidates[::step]
    return [0] + picks + [np.inf]

bins = nice_bins(g["Company_Count"], target=5)
cls = mapclassify.UserDefined(g["Company_Count"].values, bins)
g["bin"] = cls.yb  # may contain -1 when value doesn't fall in bins (e.g. zeros)

# ------------------------------------------------------------
# PLOT MAP (safe bin handling + London border)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 14))

for i, r in g.iterrows():
    # Safely handle missing/out-of-range bins and zeros
    bin_idx = int(r["bin"]) if pd.notna(r["bin"]) else -1
    if bin_idx < 0 or bin_idx >= len(colors) or r["Company_Count"] == 0:
        facecolor = "#F0F0F0"  # light grey for zero/invalid
    else:
        facecolor = colors[bin_idx]

    # London border highlight
    if r["nuts118nm"] == "London":
        edge_color = "#B0B0B0"  # light grey
        edge_width = 1.2
    else:
        edge_color = "#4D4D4D"
        edge_width = 0.5

    g.iloc[[i]].plot(ax=ax, color=facecolor, edgecolor=edge_color, linewidth=edge_width)

bounds = g.total_bounds

# ------------------------------------------------------------
# LABELS & CALLOUTS
# ------------------------------------------------------------
label_positions = {
    "North East": ("right", 650000),
    "North West": ("left", 400000),
    "Yorkshire and The Humber": ("right", 480000),
    "East Midlands": ("right", 380000),
    "West Midlands": ("left", 320000),
    "East of England": ("right", 280000),
    "London": ("right", 180000),
    "South East": ("right", 80000),
    "South West": ("left", 120000),
    "Wales": ("left", 220000),
    "Scotland": ("left", 750000),
    "Northern Ireland": ("left", 500000),
}

for _, r in g.iterrows():
    centroid = r.geometry.centroid
    cx, cy = centroid.x, centroid.y
    name = r["nuts118nm"].replace(" (England)", "")
    count = int(r["Company_Count"])

    if name not in label_positions:
        continue

    side, ty = label_positions[name]
    if side == "left":
        line_end_x = bounds[0] - 30000
        text_x = line_end_x - 5000
        text_ha = "right"
    else:
        line_end_x = bounds[2] + 30000
        text_x = line_end_x + 5000
        text_ha = "left"

    circle = Circle((cx, cy), 5000, facecolor="#FFD40E", edgecolor="black", linewidth=0.5, zorder=10)
    circle.set_path_effects([Stroke(linewidth=1.2, foreground="black"), Normal()])
    ax.add_patch(circle)
    ax.add_line(Line2D([cx, cx], [cy, ty], color="black", linewidth=0.8))
    ax.add_line(Line2D([cx, line_end_x], [ty, ty], color="black", linewidth=0.8))

    ax.text(text_x, ty, name, fontsize=16, va="bottom", ha=text_ha, fontweight="regular", zorder=11)
    ax.text(text_x, ty - 8000, f"{count}", fontsize=16, va="top", ha=text_ha, fontweight="bold", zorder=11)

# ------------------------------------------------------------
# LEGEND (numeric ranges)
# ------------------------------------------------------------
# Build readable bin labels from 'bins' (upper-bound edges)
bin_labels = []
for i in range(len(bins) - 2):
    low = int(bins[i]) + (1 if i > 0 else 0)  # 0–x includes 0; others start at +1
    high = int(bins[i + 1])
    bin_labels.append(f"{low}–{high}")
# Last bin is open-ended
bin_labels[-1] = f">{int(bins[-2])}"

# Draw color swatches
box_w = 0.025
start_x = 0.04
start_y = 0.90
for i, col in enumerate(colors):
    rect = Rectangle(
        (start_x + i * box_w, start_y), box_w, box_w,
        transform=fig.transFigure, fc=col, ec="none"
    )
    fig.patches.append(rect)

# Add min/max anchors above the legend bar
ax.text(
    start_x - 0.005, start_y + box_w / 2, bin_labels[0].split("–")[0],
    transform=fig.transFigure, fontsize=16, va="center", ha="right"
)
ax.text(
    start_x + len(colors) * box_w + 0.005, start_y + box_w / 2, bin_labels[-1].replace(">", ""),
    transform=fig.transFigure, fontsize=16, va="center", ha="left"
)

# Add textual ranges
ax.text(
    start_x, start_y - 0.03, " | ".join(bin_labels),
    transform=fig.transFigure, fontsize=12
)

ax.set_title("UK Company Distribution by NUTS Level 1 Region", fontsize=16, fontweight="bold", pad=20)
ax.axis("off")
plt.tight_layout()

# ------------------------------------------------------------
# SHOW & EXPORT
# ------------------------------------------------------------
st.pyplot(fig, use_container_width=True)

st.subheader("Export Map")
svg_buffer = io.BytesIO()
png_buffer = io.BytesIO()

fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
svg_buffer.seek(0)

fig.savefig(png_buffer, format="png", bbox_inches="tight", dpi=300)
png_buffer.seek(0)

col1, col2 = st.columns(2)
with col1:
    st.download_button("📥 Download SVG", data=svg_buffer, file_name="uk_company_map.svg", mime="image/svg+xml")
with col2:
    st.download_button("📥 Download PNG (300 dpi)", data=png_buffer, file_name="uk_company_map.png", mime="image/png")
