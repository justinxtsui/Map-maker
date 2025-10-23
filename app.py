import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import mapclassify
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.patheffects import Stroke, Normal
import os
import requests
import zipfile

# Set page config
st.set_page_config(page_title="UK Regional Company Map", layout="wide")

# Set matplotlib font settings
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "Public Sans"
mpl.rcParams["font.sans-serif"] = ["Public Sans", "Arial", "DejaVu Sans"]
mpl.rcParams["font.weight"] = "normal"

# Google Drive file ID
GDRIVE_FILE_ID = "1ip-Aip_rQNucgdJRvIBckSnYa_RBcRFU"
SHAPEFILE_DIR = "shapefile_data"
SHAPEFILE_NAME = "NUTS_Level_1__January_2018__Boundaries.shp"

@st.cache_resource
def download_shapefile():
    """Download shapefile from Google Drive if not already present"""
    shapefile_path = os.path.join(SHAPEFILE_DIR, SHAPEFILE_NAME)
    
    if os.path.exists(shapefile_path):
        return shapefile_path
    
    # Create directory if it doesn't exist
    os.makedirs(SHAPEFILE_DIR, exist_ok=True)
    
    # Download from Google Drive
    url = f"https://drive.google.com/uc?export=download&id={1ip-Aip_rQNucgdJRvIBckSnYa_RBcRFU}"
    
    with st.spinner("Downloading shapefile from Google Drive..."):
        response = requests.get(url)
        
        if response.status_code == 200:
            # Save as zip file
            zip_path = os.path.join(SHAPEFILE_DIR, "shapefile.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(SHAPEFILE_DIR)
            
            # Remove zip file
            os.remove(zip_path)
            
            return shapefile_path
        else:
            st.error(f"Failed to download shapefile. Status code: {response.status_code}")
            st.error("Make sure the Google Drive file is publicly accessible.")
            return None

st.title("UK Regional Company Distribution Map")

st.write("""
Upload a CSV file containing company data with the following columns:
- **Head Office Address - Region**
- **Registered Address - Region**

The app will automatically fill missing head office addresses with registered addresses and generate a map.
""")

# Download shapefile
shapefile_path = download_shapefile()

if shapefile_path is None:
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Check if required columns exist
    required_cols = ["Head Office Address - Region", "Registered Address - Region"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        # Step 1: Fill missing head office addresses with registered addresses
        df["Region (merged)"] = df["Head Office Address - Region"].fillna(df["Registered Address - Region"])
        
        st.success(f"Data loaded successfully! Total companies: {len(df)}")
        
        # Show data preview
        with st.expander("Preview Data"):
            st.dataframe(df[["Head Office Address - Region", "Registered Address - Region", "Region (merged)"]].head(10))
        
        # Region mapping
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
            "Yorkshire and The Humber": "Yorkshire and The Humber"
        }
        
        df["Region_Mapped"] = df["Region (merged)"].map(region_mapping)
        region_counts = df.groupby("Region_Mapped").size().reset_index(name="Company_Count")
        
        # Load shapefile
        os.environ["SHAPE_RESTORE_SHX"] = "YES"
        gdf_level1 = gpd.read_file(shapefile_path)
        
        # Merge data
        merged_gdf = gdf_level1.merge(region_counts, left_on="nuts118nm", right_on="Region_Mapped", how="left")
        
        # Create visualization
        custom_colors = ["#E6E6FA", "#C2C2F0", "#9999E6", "#6666CC", "#3333B3"]
        
        classifier = mapclassify.Quantiles(merged_gdf["Company_Count"], k=5)
        merged_gdf["color_bin"] = classifier.yb
        
        fig, ax = plt.subplots(figsize=(12, 14))
        
        # Plot regions
        for idx, row in merged_gdf.iterrows():
            if row["nuts118nm"] != "London":
                color = custom_colors[row["color_bin"]]
                merged_gdf[merged_gdf.index == idx].plot(ax=ax, color=color, edgecolor="#4D4D4D", linewidth=0.5)
        
        # Plot London separately
        london_row = merged_gdf[merged_gdf["nuts118nm"] == "London"]
        if len(london_row) > 0:
            london_color = custom_colors[london_row.iloc[0]["color_bin"]]
            london_row.plot(ax=ax, color=london_color, edgecolor="#D3D3D3", linewidth=0.5)
        
        bounds = merged_gdf.total_bounds
        
        # Label positions
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
            "Northern Ireland": ("left", 500000)
        }
        
        # Add labels and circles
        for idx, row in merged_gdf.iterrows():
            centroid = row["geometry"].centroid
            cx, cy = centroid.x, centroid.y
            
            region_name = row["nuts118nm"].replace(" (England)", "")
            count = row["Company_Count"]
            
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
                
                circle = Circle(
                    (cx, cy),
                    5000,
                    facecolor="#FFD40E",
                    edgecolor="black",
                    linewidth=0.5,
                    antialiased=True,
                    zorder=10
                )
                circle.set_path_effects([Stroke(linewidth=1.2, foreground="black"), Normal()])
                ax.add_patch(circle)
                
                line1 = Line2D([cx, cx], [cy, target_y], color="black", linewidth=0.8, zorder=9)
                ax.add_line(line1)
                
                line2 = Line2D([cx, line_end_x], [target_y, target_y], color="black", linewidth=0.8, zorder=9)
                ax.add_line(line2)
                
                ax.text(text_x, target_y, region_name, fontsize=16, 
                        verticalalignment="bottom", horizontalalignment=text_ha,
                        fontfamily="Public Sans", fontweight="normal", zorder=11, rasterized=False)
                
                ax.text(text_x, target_y - 8000, str(count), fontsize=16, 
                        verticalalignment="top", horizontalalignment=text_ha,
                        fontfamily="Public Sans", fontweight=600, zorder=11, rasterized=False)
        
        min_count = merged_gdf["Company_Count"].min()
        max_count = merged_gdf["Company_Count"].max()
        
        ax.set_title("UK Company Distribution by NUTS Level 1 Region", fontsize=16, fontweight="bold", pad=20, fontfamily="Public Sans")
        
        # Draw legend
        box_size = 0.025
        start_x = 0.04
        start_y = 0.90
        
        for i, color in enumerate(custom_colors):
            rect = Rectangle(
                (start_x + i * box_size, start_y), 
                box_size, box_size,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none"
            )
            fig.patches.append(rect)
        
        ax.text(start_x - 0.005, start_y + box_size/2, f"{min_count:.0f}", 
                transform=fig.transFigure,
                fontsize=16, verticalalignment="center", horizontalalignment="right",
                fontfamily="Public Sans")
        
        ax.text(start_x + len(custom_colors) * box_size + 0.005, start_y + box_size/2, f"{max_count:.0f}",
                transform=fig.transFigure,
                fontsize=16, verticalalignment="center", horizontalalignment="left",
                fontfamily="Public Sans")
        
        ax.axis("off")
        plt.tight_layout()
        
        # Display the map
        st.pyplot(fig)
        
        # Show statistics
        st.subheader("Regional Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Companies", f"{merged_gdf['Company_Count'].sum():.0f}")
        with col2:
            st.metric("Regions with Data", f"{merged_gdf['Company_Count'].notna().sum()}")
        with col3:
            st.metric("Company Range", f"{min_count:.0f} - {max_count:.0f}")
        
        # Show regional breakdown
        with st.expander("Regional Breakdown"):
            st.dataframe(region_counts.sort_values("Company_Count", ascending=False))
