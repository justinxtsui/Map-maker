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
        response = requests.get(url)
        if response.status_code == 200:
            zip_path = os.path.join(SHAPEFILE_DIR, "shapefile.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(SHAPEFILE_DIR)
            os.remove(zip_path)
            return shapefile_path
        else:
            st.error(f"Failed to download shapefile. Status code: {response.status_code}")
            st.error("Make sure the Google Drive file is publicly accessible.")
            return None

# ------------------------------------------------------------
# APP CONTENT
# ------------------------------------------------------------
st.title("UK Regional Company Distribution Map")

st.write("""
Upload a CSV or Excel file containing company data with the following columns:
- **Head Office Address - Region**
- **Registered Address - Region**
""")

# Download shapefile (only once)
shapefile_path = download_shapefile()
if shapefile_path is None:
    st.stop()

# ------------------------------------------------------------
# FILE UPLOAD SECTION
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Read file based on extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        st.error("Unsupported file format")
        st.stop()

    required_cols = ["Head Office Address - Region", "Registered Address - Region"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    # ------------------------------------------------------------
    # CLEAN & MERGE REGIONS
    # ------------------------------------------------------------
    for col in required_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "(no value)": np.nan, "": np.nan})
        )
    df["Region (merged)"] = df["Head Office Address - Region"].fillna(df["Registered Address - Region"])

    # ------------------------------------------------------------
    # REGION MAPPING (with Scotland grouping)
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
        # Scotland subregions
        "West of Scotland": "Scotland",
        "East of Scotland": "Scotland",
        "South of Scotland": "Scotland",
        "Highlands and Islands": "Scotland",
        "Tayside": "Scotland",
        "Aberdeen": "Scotland",
    }

    df["Region_Mapped"] = df["Region (merged)"].map(region_mapping)
    df["Region_Mapped"] = df["Region_Mapped"].fillna("Unknown")

    region_counts = (
        df.groupby("Region_Mapped")
        .size()
        .reset_index(name="Company_Count")
    )
    region_counts = region_counts[region_counts["Region_Mapped"] != "Unknown"]

    # ------------------------------------------------------------
    # LOAD SHAPEFILE AND MERGE DATA
    # ------------------------------------------------------------
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf_level1 = gpd.read_file(shapefile_path)
    merged_gdf = gdf_level1.merge(region_counts, left_on="nuts118nm", right_on="Region_Mapped", how="left")
    merged_gdf["Company_Count"] = merged_gdf["Company_Count"].fillna(0)

    # ------------------------------------------------------------
    # MAP VISUALIZATION
    # ------------------------------------------------------------
    custom_colors = ["#E6E6FA", "#C2C2F0", "#9999E6", "#6666CC", "#3333B3"]
    values_for_classify = merged_gdf["Company_Count"]

    if (values_for_classify > 0).sum() >= 5:
        classifier = mapclassify.Quantiles(values_for_classify, k=5)
        merged_gdf["color_bin"] = classifier.yb
    else:
        merged_gdf["color_bin"] = (values_for_classify > 0).astype(int)

    fig, ax = plt.subplots(figsize=(12, 14))
    for idx, row in merged_gdf.iterrows():
        color = "#F0F0F0" if row["Company_Count"] == 0 else custom_colors[int(row["color_bin"])]
        merged_gdf.iloc[[idx]].plot(ax=ax, color=color, edgecolor="#4D4D4D", linewidth=0.5)

    bounds = merged_gdf.total_bounds

    # ------------------------------------------------------------
    # LABELS
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

    for _, row in merged_gdf.iterrows():
        centroid = row["geometry"].centroid
        cx, cy = centroid.x, centroid.y
        region_name = row["nuts118nm"].replace(" (England)", "")
        count = int(row["Company_Count"])

        if region_name in label_positions:
            side, target_y = label_positions[region_name]
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
            ax.add_line(Line2D([cx, cx], [cy, target_y], color="black", linewidth=0.8))
            ax.add_line(Line2D([cx, line_end_x], [target_y, target_y], color="black", linewidth=0.8))
            ax.text(text_x, target_y, region_name, fontsize=16, va="bottom", ha=text_ha, fontweight="regular")
            ax.text(text_x, target_y - 8000, f"{count}", fontsize=16, va="top", ha=text_ha, fontweight="bold")

    ax.set_title("UK Company Distribution by NUTS Level 1 Region", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()

    # ------------------------------------------------------------
    # DISPLAY MAP AND EXPORT
    # ------------------------------------------------------------
    st.pyplot(fig, use_container_width=True)

    st.subheader("Export Map")
    svg_buffer = io.BytesIO()
    png_buffer = io.BytesIO()

    fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
    svg_buffer.seek(0)
    fig.savefig(png_buffer, format='png', bbox_inches='tight', dpi=300)
    png_buffer.seek(0)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download SVG", data=svg_buffer, file_name="uk_company_map.svg", mime="image/svg+xml")
    with col2:
        st.download_button("📥 Download PNG (300 dpi)", data=png_buffer, file_name="uk_company_map.png", mime="image/png")
