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
    shapefile_path = os.path.join(SHAPEFILE_DIR, SHAPEFILE_NAME)
    if os.path.exists(shapefile_path):
        return shapefile_path
    os.makedirs(SHAPEFILE_DIR, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    with st.spinner("Downloading shapefile from Google Drive..."):
        r = requests.get(url)
        if r.status_code == 200:
            z = os.path.join(SHAPEFILE_DIR, "shapefile.zip")
            with open(z, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(SHAPEFILE_DIR)
            os.remove(z)
            return shapefile_path
        else:
            st.error(f"Download failed: {r.status_code}")
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

path = download_shapefile()
if path is None:
    st.stop()

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
f = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
if f is None:
    st.stop()

ext = f.name.split('.')[-1].lower()
df = pd.read_csv(f) if ext == "csv" else pd.read_excel(f, engine="openpyxl")

req = ["Head Office Address - Region", "Registered Address - Region"]
if any(c not in df.columns for c in req):
    st.error("Missing required columns.")
    st.stop()

for c in req:
    df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "(no value)": np.nan, "": np.nan})

df["Region (merged)"] = df["Head Office Address - Region"].fillna(df["Registered Address - Region"])

# ------------------------------------------------------------
# REGION MAPPING (group Scotland subregions)
# ------------------------------------------------------------
mapping = {
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
    "West of Scotland": "Scotland",
    "East of Scotland": "Scotland",
    "South of Scotland": "Scotland",
    "Highlands and Islands": "Scotland",
    "Tayside": "Scotland",
    "Aberdeen": "Scotland",
}
df["Region_Mapped"] = df["Region (merged)"].map(mapping).fillna("Unknown")
counts = df[df["Region_Mapped"] != "Unknown"].groupby("Region_Mapped").size().reset_index(name="Company_Count")

# ------------------------------------------------------------
# LOAD SHAPEFILE
# ------------------------------------------------------------
os.environ["SHAPE_RESTORE_SHX"] = "YES"
gdf = gpd.read_file(path)
g = gdf.merge(counts, left_on="nuts118nm", right_on="Region_Mapped", how="left").fillna({"Company_Count": 0})

# ------------------------------------------------------------
# SMART BINNING
# ------------------------------------------------------------
colors = ["#E6E6FA", "#C2C2F0", "#9999E6", "#6666CC", "#3333B3"]

def nice_bins(vals, target=5):
    pos = np.array(vals)[np.array(vals) > 0]
    if len(pos) == 0:
        return [0, np.inf]
    lo, hi = float(pos.min()), float(pos.max())
    lo_e, hi_e = int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1
    cands = sorted(m*(10**e) for e in range(lo_e, hi_e+1) for m in (1,2,5) if lo <= m*(10**e) <= hi)
    step = max(1, int(np.ceil(len(cands)/(target-1))))
    picks = cands[::step]
    return [0] + picks + [np.inf]

bins = nice_bins(g["Company_Count"], 5)
cls = mapclassify.UserDefined(g["Company_Count"].values, bins)
g["bin"] = cls.yb

# ------------------------------------------------------------
# PLOT MAP
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,14))
for i, r in g.iterrows():
    c = "#F0F0F0" if r["Company_Count"] == 0 else colors[int(r["bin"])]
    if r["nuts118nm"] == "London":
        ec, lw = "#B0B0B0", 1.2
    else:
        ec, lw = "#4D4D4D", 0.5
    g.iloc[[i]].plot(ax=ax, color=c, edgecolor=ec, linewidth=lw)

b = g.total_bounds
labels = {
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
    cx, cy = r.geometry.centroid.x, r.geometry.centroid.y
    name = r["nuts118nm"].replace(" (England)", "")
    c = int(r["Company_Count"])
    if name not in labels: continue
    side, ty = labels[name]
    if side == "left":
        lx, tx, ha = b[0]-30000, b[0]-35000, "right"
    else:
        lx, tx, ha = b[2]+30000, b[2]+35000, "left"
    circle = Circle((cx, cy), 5000, facecolor="#FFD40E", edgecolor="black", lw=0.5)
    circle.set_path_effects([Stroke(lw=1.2, foreground="black"), Normal()])
    ax.add_patch(circle)
    ax.add_line(Line2D([cx, cx], [cy, ty], color="black", lw=0.8))
    ax.add_line(Line2D([cx, lx], [ty, ty], color="black", lw=0.8))
    ax.text(tx, ty, name, fontsize=16, va="bottom", ha=ha)
    ax.text(tx, ty-8000, f"{c}", fontsize=16, va="top", ha=ha, fontweight="bold")

# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------
# Prepare readable bin labels
leg_labels = []
for i in range(len(bins)-2):
    low = int(bins[i]) + 1 if i > 0 else 0
    high = int(bins[i+1])
    leg_labels.append(f"{low}–{high}")
leg_labels[-1] = f">{int(bins[-2])}"

x0, y0, box = 0.04, 0.90, 0.025
for i, c in enumerate(colors):
    Rectangle((x0+i*box, y0), box, box, transform=fig.transFigure, fc=c, ec="none").set_zorder(5)
    fig.patches.append(Rectangle((x0+i*box, y0), box, box, transform=fig.transFigure, fc=c, ec="none"))
ax.text(x0-0.005, y0+box/2, leg_labels[0].split("–")[0], transform=fig.transFigure, fontsize=16, va="center", ha="right")
ax.text(x0+len(colors)*box+0.005, y0+box/2, leg_labels[-1].replace(">",""), transform=fig.transFigure, fontsize=16, va="center", ha="left")

# Add legend text row
ax.text(x0, y0-0.03, " | ".join(leg_labels), transform=fig.transFigure, fontsize=12)

ax.set_title("UK Company Distribution by NUTS Level 1 Region", fontsize=16, fontweight="bold", pad=20)
ax.axis("off")
plt.tight_layout()

# ------------------------------------------------------------
# SHOW & EXPORT
# ------------------------------------------------------------
st.pyplot(fig, use_container_width=True)
st.subheader("Export Map")
svg, png = io.BytesIO(), io.BytesIO()
fig.savefig(svg, format="svg", bbox_inches="tight"); svg.seek(0)
fig.savefig(png, format="png", bbox_inches="tight", dpi=300); png.seek(0)
c1, c2 = st.columns(2)
with c1:
    st.download_button("📥 Download SVG", data=svg, file_name="uk_company_map.svg", mime="image/svg+xml")
with c2:
    st.download_button("📥 Download PNG (300 dpi)", data=png, file_name="uk_company_map.png", mime="image/png")
