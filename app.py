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

# --------------------------- Cached: download + load + simplify shapefile ---------------------------
@st.cache_resource
def load_regions_gdf():
    """
    Downloads the shapefile from Google Drive (if needed),
    reads into a GeoDataFrame and simplifies geometry.
    Cached for the whole Streamlit session.
    """
    os.makedirs(SHAPEFILE_DIR, exist_ok=True)
    shp_path = os.path.join(SHAPEFILE_DIR, SHAPEFILE_NAME)

    # Download once if missing
    if not os.path.exists(shp_path):
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

    # Load & simplify
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(shp_path)

    # Light simplification for faster plotting (tweak tolerance as needed)
    try:
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=500, preserve_topology=True)
    except Exception:
        # If simplify fails for any reason, just return original
        pass

    return gdf

# --------------------------- Helpers: formatting ---------------------------
def format_pct_3sf(n, total):
    """Return n/total as a percentage string to 3 significant figures."""
    if total <= 0:
        return "0%"
    pct = 100 * (float(n) / float(total))
    s = f"{pct:.3g}"  # three significant figures; auto-scales
    return f"{s}%"

def format_money_3sf(x):
    """
    Format a numeric value as money with £ and units (k, m, b),
    to 3 significant figures.
    Examples: 1234 -> £1.23k, 1_200_000 -> £1.2m, 532 -> £532
    """
    x = float(x)
    if x == 0:
        return "£0"
    neg = x < 0
    x_abs = abs(x)

    if x_abs >= 1e9:
        unit = "b"
        divisor = 1e9
    elif x_abs >= 1e6:
        unit = "m"
        divisor = 1e6
    elif x_abs >= 1e3:
        unit = "k"
        divisor = 1e3
    else:
        unit = ""
        divisor = 1.0

    scaled = x_abs / divisor
    s = f"{scaled:.3g}"  # 3 significant figures on the scaled number
    sign = "-" if neg else ""
    return f"{sign}£{s}{unit}"

# --------------------------- UI ---------------------------
st.title("Mapphew 🗺️")
st.write(
    "Upload a CSV or Excel with **Head Office Address - Region** / "
    "**(Company) Head Office Address - Region** and "
    "**Registered Address - Region** / **(Company) Registered Address - Region**. "
    "The app will merge these two columns and create the map."
)

gdf_regions = load_regions_gdf()
if gdf_regions is None:
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

# Display mode (labels)
display_mode = st.radio(
    "Display values as:",
    ["Raw value (count/sum)", "Percentage of total (3 s.f.)"],
    horizontal=True,
    index=0
)

# --------------------------- Load file ---------------------------
ext = uploaded.name.split(".")[-1].lower()

if ext == "csv":
    df = pd.read_csv(uploaded)
else:
    # Excel: let user choose which sheet to use
    xls = pd.ExcelFile(uploaded, engine="openpyxl")
    sheet_name = st.selectbox("Choose a sheet", options=xls.sheet_names, index=0)
    df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")

# --------------------------- Resolve region columns (supports "(Company) ..." variants) ---------------------------
region_col_aliases = {
    "Head Office Address - Region": [
        "Head Office Address - Region",
        "(Company) Head Office Address - Region",
    ],
    "Registered Address - Region": [
        "Registered Address - Region",
        "(Company) Registered Address - Region",
    ],
}

resolved_cols = {}
missing_canonical = []

for canonical, aliases in region_col_aliases.items():
    found = None
    for a in aliases:
        if a in df.columns:
            found = a
            break
    if found is None:
        missing_canonical.append(canonical)
    else:
        resolved_cols[canonical] = found

if missing_canonical:
    # Build a helpful error message listing accepted alternatives
    details = []
    for canonical, aliases in region_col_aliases.items():
        alias_list = ", ".join(f"`{a}`" for a in aliases)
        details.append(f"- **{canonical}**: one of {alias_list}")
    st.error(
        "Missing required region columns.\n\n"
        "Please ensure your file contains at least one column for each of the following:\n\n"
        + "\n".join(details)
    )
    st.stop()

head_col = resolved_cols["Head Office Address - Region"]
reg_col = resolved_cols["Registered Address - Region"]

# --------------------------- Aggregation mode (count vs sum) ---------------------------
agg_mode = st.radio(
    "What metric should the map show?",
    ["Number of companies (row count)", "Sum a numeric column"],
    index=0
)

sum_col = None
sum_is_money = False

if agg_mode == "Sum a numeric column":
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns available to sum. Please upload a file with at least one numeric column.")
        st.stop()
    sum_col = st.selectbox("Numeric column to sum per region:", options=numeric_cols)
    # Tick box for money formatting
    sum_is_money = st.checkbox(
        "Treat summed values as money (£ with k / m / b units, 3 s.f.)",
        value=False
    )

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

# --------------------------- Clean & merge regions ---------------------------
for c in [head_col, reg_col]:
    df[c] = (
        df[c]
        .astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "(no value)": np.nan, "": np.nan})
    )

df["Region (merged)"] = df[head_col].fillna(df[reg_col])

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

# --------------------------- Aggregate per region (COUNT or SUM) ---------------------------
valid = df[df["Region_Mapped"] != "Unknown"]

if agg_mode == "Number of companies (row count)":
    agg_series = valid.groupby("Region_Mapped").size()
else:
    agg_series = valid.groupby("Region_Mapped")[sum_col].sum()

counts = agg_series.reset_index(name="Region_Value")

# --------------------------- Join shapes ---------------------------
g = gdf_regions.merge(counts, left_on="nuts118nm", right_on="Region_Mapped", how="left")
g["Region_Value"] = g["Region_Value"].fillna(0)

# --------------------------- Totals ---------------------------
_total_value = float(g["Region_Value"].sum())

# --------------------------- Binning helpers ---------------------------
def bins_equal_interval(pos_vals, k=5):
    lo, hi = float(np.min(pos_vals)), float(np.max(pos_vals))
    if lo == hi:
        edges = [hi] * (k - 1)
    else:
        step = (hi - lo) / k
        edges = [lo + step * i for i in range(1, k)]
    return [*edges, np.inf]

def bins_quantiles(pos_vals, k=5):
    qs = np.quantile(pos_vals, np.linspace(0, 1, k + 1))[1:-1].tolist()
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
