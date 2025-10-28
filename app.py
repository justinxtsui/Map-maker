import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mapclassify
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.patheffects import Stroke, Normal
import os, requests, zipfile, io, numpy as np

# --------------------------- Page & fonts ---------------------------
st.set_page_config(page_title="UK Regional Company Map", layout="wide")
mpl.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.weight": "regular",
    "axes.titleweight": "bold",
    "axes.labelweight": "regular",
})

# --------------------------- Data sources ---------------------------
GDRIVE_FILE_ID = "1ip-Aip_rQNucgdJRvIBckSnYa_RBcRFU"
SHAPEFILE_DIR = "shapefile_data"
SHAPEFILE_NAME = "NUTS_Level_1__January_2018__Boundaries.shp"

@st.cache_resource
def download_shapefile():
    shp_path = os.path.join(SHAPEFILE_DIR, SHAPEFILE_NAME)
    if os.path.exists(shp_path):
        return shp_path
    os.makedirs(SHAPEFILE_DIR, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    with st.spinner("Downloading shapefile from Google Drive..."):
        r = requests.get(url)
        if r.status_code != 200:
            st.error(f"Failed to download shapefile. Status code: {r.status_code}")
            return None
        zpath = os.path.join(SHAPEFILE_DIR, "shapefile.zip")
        with open(zpath, "wb") as f: 
            f.write(r.content)
        with zipfile.ZipFile(zpath, "r") as zf: 
            zf.extractall(SHAPEFILE_DIR)
        os.remove(zpath)
    return shp_path

# --------------------------- UI ---------------------------
st.title("Mapphew 🗺️")
st.write("Upload a CSV or Excel with **Head Office Address - Region** and **Registered Address - Region**. The app will merge these two columns and create the map.")

shp_path = download_shapefile()
if not shp_path: 
    st.stop()

uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
if not uploaded: 
    st.stop()

# Colour scheme selector
bin_mode = st.selectbox(
    "Change colour scheme",
    ["Tableau-like (Equal Interval)", "Quantiles", "Natural Breaks (Fisher-Jenks)", "Pretty (1–2–5)"],
    index=0
)

# Custom map title (user input)
map_title = st.text_input("Enter your custom map title:", "UK Company Distribution by NUTS Level 1 Region")

# --------------------------- Load file ---------------------------
ext = uploaded.name.split(".")[-1].lower()

if ext == "csv":
    df = pd.read_csv(uploaded)
else:
    # Excel: let user choose which sheet to use
    xls = pd.ExcelFile(uploaded, engine="openpyxl")
    sheet_name = st.selectbox("Choose a sheet", options=xls.sheet_names, index=0)
    df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")

req = ["Head Office Address - Region", "Registered Address - Region"]
missing = [c for c in req if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Clean & merge regions
for c in req:
    df[c] = (df[c].astype(str).str.strip()
             .replace({"nan": np.nan, "None": np.nan, "(no value)": np.nan, "": np.nan}))
df["Region (merged)"] = df["Head Office Address - Region"].fillna(df["Registered Address - Region"])

# --------------------------- Optional filtering ---------------------------
st.subheader("Optional Filter")
filter_col = st.selectbox("Select a column to filter:", options=df.columns, index=0)
unique_vals = df[filter_col].dropna().unique()
if len(unique_vals) > 100:
    st.warning("Too many unique values — showing only the first 100 distinct values.")
    unique_vals = unique_vals[:100]
selected_vals = st.multiselect("Select values:", options=sorted(unique_vals, key=lambda x: str(x)))
filter_mode = st.radio("Filter mode:", ["Include", "Exclude"], horizontal=True)
if selected_vals:
    if filter_mode == "Include":
        df = df[df[filter_col].isin(selected_vals)]
    else:
        df = df[~df[filter_col].isin(selected_vals)]
    st.success(f"Filtered to {len(df)} rows based on **{filter_col}** ({filter_mode}).")

# --------------------------- Region mapping ---------------------------
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
    # Scotland subregions -> Scotland
    "West of Scotland": "Scotland",
    "East of Scotland": "Scotland",
    "South of Scotland": "Scotland",
    "Highlands and Islands": "Scotland",
    "Tayside": "Scotland",
    "Aberdeen": "Scotland",
}
df["Region_Mapped"] = df["Region (merged)"].map(region_mapping).fillna("Unknown")

counts = (df[df["Region_Mapped"] != "Unknown"]
          .groupby("Region_Mapped").size()
          .reset_index(name="Company_Count"))

# --------------------------- Join shapes ---------------------------
os.environ["SHAPE_RESTORE_SHX"] = "YES"
gdf = gpd.read_file(shp_path)
g = gdf.merge(counts, left_on="nuts118nm", right_on="Region_Mapped", how="left")
g["Company_Count"] = g["Company_Count"].fillna(0)

# --------------------------- Binning ---------------------------
def bins_equal_interval(pos_vals, k=5):
    lo, hi = float(np.min(pos_vals)), float(np.max(pos_vals))
    if lo == hi:
        edges = [hi] * (k-1)
    else:
        step = (hi - lo) / k
        edges = [lo + step * i for i in range(1, k)]
    return [*edges, np.inf]

def bins_quantiles(pos_vals, k=5):
    qs = np.quantile(pos_vals, np.linspace(0, 1, k+1))[1:-1].tolist()
    return qs + [np.inf]

def bins_fisher_jenks(pos_vals, k=5):
    u = np.unique(pos_vals)
    k_eff = int(min(k, max(2, len(u))))
    try:
        fj = mapclassify.FisherJenks(pos_vals, k=k_eff)
        bins = list(fj.bins[:-1]) + [np.inf]
        return bins
    except Exception:
        return bins_equal_interval(pos_vals, k)

def bins_pretty_125(pos_vals, k=5):
    lo, hi = float(np.min(pos_vals)), float(np.max(pos_vals))
    lo_e, hi_e = int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1
    cands = sorted({m*(10**e) for e in range(lo_e, hi_e+1) for m in (1,2,5) if lo <= m*(10**e) <= hi})
    if len(cands) >= (k-1):
        step = len(cands)/(k-1)
        edges = [cands[int(round((i+1)*step))-1] for i in range(k-1)]
    else:
        gs = np.geomspace(lo, hi, num=k).tolist()[1:-1]
        edges = sorted(set([max(1, int(x)) for x in gs] + cands))
        while len(edges) < (k-1):
            edges.append(edges[-1]+1)
        edges = edges[:k-1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1
    return edges + [np.inf]

def build_bins(values, mode="Tableau-like (Equal Interval)", k=5):
    vals = np.asarray(values, dtype=float)
    pos = vals[vals > 0]
    if len(pos) == 0:
        return [1, 2, 3, 4, np.inf]
    if mode.startswith("Tableau"):
        return bins_equal_interval(pos, k)
    if mode == "Quantiles":
        return bins_quantiles(pos, k)
    if mode.startswith("Natural"):
        return bins_fisher_jenks(pos, k)
    if mode.startswith("Pretty"):
        return bins_pretty_125(pos, k)
    return bins_equal_interval(pos, k)

pos_bins = build_bins(g["Company_Count"].values, mode=bin_mode, k=5)
cls = mapclassify.UserDefined(g["Company_Count"].values, bins=pos_bins)
g["bin"] = cls.yb

# --------------------------- Plot ---------------------------
palette = ["#B5E7F4", "#90DBEF", "#74D1EA", "#4BB5CF", "#2B8EAA"]
fig, ax = plt.subplots(figsize=(7.5, 8.5))

for i, r in g.iterrows():
    cnt = int(r["Company_Count"])
    if cnt == 0:
        face = "#F0F0F0"
    else:
        idx = int(r["bin"])
        idx = 0 if idx is None or np.isnan(idx) else max(0, min(idx, len(palette)-1))
        face = palette[idx]
    g.iloc[[i]].plot(ax=ax, color=face, edgecolor="#4D4D4D", linewidth=0.5)

bounds = g.total_bounds

# Labels & callouts
label_pos = {
    "North East": ("right", 650000), "North West": ("left", 400000),
    "Yorkshire and The Humber": ("right", 480000), "East Midlands": ("right", 380000),
    "West Midlands": ("left", 320000), "East of England": ("right", 280000),
    "London": ("right", 180000), "South East": ("right", 80000),
    "South West": ("left", 120000), "Wales": ("left", 220000),
    "Scotland": ("left", 750000), "Northern Ireland": ("left", 500000),
}
for _, r in g.iterrows():
    cx, cy = r.geometry.centroid.x, r.geometry.centroid.y
    name = r["nuts118nm"].replace(" (England)", "")
    cnt = int(r["Company_Count"])
    if name not in label_pos: continue
    side, ty = label_pos[name]
    if side == "left":
        lx, tx, ha = bounds[0]-30000, bounds[0]-35000, "right"
    else:
        lx, tx, ha = bounds[2]+30000, bounds[2]+35000, "left"
    circ = Circle((cx, cy), 5000, facecolor="#FFD40E", edgecolor="black", linewidth=0.5, zorder=10)
    circ.set_path_effects([Stroke(linewidth=1.2, foreground="black"), Normal()])
    ax.add_patch(circ)
    ax.add_line(Line2D([cx, cx], [cy, ty], color="black", linewidth=0.8))
    ax.add_line(Line2D([cx, lx], [ty, ty], color="black", linewidth=0.8))
    ax.text(tx, ty, name, fontsize=11, va="bottom", ha=ha)
    ax.text(tx, ty-8000, f"{cnt}", fontsize=11, va="top", ha=ha, fontweight="bold")

# Legend (min/max only)
pos_vals = g.loc[g["Company_Count"] > 0, "Company_Count"]
min_pos, max_pos = (0, 0) if len(pos_vals) == 0 else (int(pos_vals.min()), int(pos_vals.max()))
box_w, start_x, start_y = 0.025, 0.04, 0.90
for i, col in enumerate(palette):
    rect = Rectangle((start_x + i*box_w, start_y), box_w, box_w,
                     transform=fig.transFigure, fc=col, ec="none")
    fig.patches.append(rect)
ax.text(start_x - 0.005, start_y + box_w/2, f"{min_pos}",
       transform=fig.transFigure, fontsize=13, va="center", ha="right")
ax.text(start_x + len(palette)*box_w + 0.005, start_y + box_w/2, f"{max_pos}",
       transform=fig.transFigure, fontsize=13, va="center", ha="left")

ax.set_title(map_title, fontsize=15, fontweight="bold", pad=10)
ax.axis("off")
plt.tight_layout()

# --------------------------- Show & export ---------------------------
st.pyplot(fig, use_container_width=True)

st.markdown("### Export Map")
st.markdown("""
<style>
div[data-testid="column"] { flex: 1 1 45% !important; }
div[data-testid="stMarkdownContainer"] h3 { color: #000 !important; margin-bottom: 0.3rem !important; }
</style>
""", unsafe_allow_html=True)

svg, png = io.BytesIO(), io.BytesIO()
fig.savefig(svg, format="svg", bbox_inches="tight"); svg.seek(0)
fig.savefig(png, format="png", bbox_inches="tight", dpi=300); png.seek(0)

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("### For Adobe 🧑🏼‍🎨")
    st.download_button("Download SVG", data=svg, file_name="uk_company_map.svg",
                       mime="image/svg+xml", use_container_width=True)
with c2:
    st.markdown("### For Google Slides 📈")
    st.download_button("Download PNG (300 dpi)", data=png, file_name="uk_company_map.png",
                       mime="image/png", use_container_width=True)

# --------------------------- Footer image ---------------------------
st.markdown("---")
st.image(
    "you_are_welcome.png",
    caption="Last updated:24/10/25 -JT",
    use_container_width=False,
    width=175
)
