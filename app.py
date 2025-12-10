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
# ... (format_pct_3sf and format_money_3sf functions remain unchanged) ...

def format_pct_3sf(n, total):
    """Return n/total as a percentage string to 3 significant figures."""
    if total <= 0:
        return "0%"
    pct = 100 * (float(n) / float(total))
    s = f"{pct:.3g}"  # three significant figures; auto-scales
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
    s = f"{scaled:.3g}"  # 3 significant figures on the scaled number
    sign = "-" if neg else ""
    return f"{sign}£{s}{unit}"

# --------------------------- UI START ---------------------------

# 1. Main Area Headers (Always visible)
st.header("UK Regional Company Map Generator 🗺️")
st.subheader("Interactive Choropleth Map (NUTS-1 Regions)")

# 2. Load Geographical Data (Dependency Check)
gdf_regions = load_regions_gdf()

if gdf_regions is None:
    # IMPROVED ERROR FEEDBACK
    st.error("🛑 **Fatal Error: Map Dependencies Missing**")
    st.markdown("""
        The application failed to load the required geographical boundary data 
        (NUTS Level 1 UK regions) from the external source. The app cannot proceed.""")
    st.stop()


# 3. Sidebar: Step 1 - Upload Data File
with st.sidebar:
    st.header("1. Upload Data File 📂")

    # Use st.markdown for clear, scannable column requirements
    st.markdown("""
        To map your data, your file must contain at least **one** column
        from each of the following required groups:
        
        * **Primary Region (Preferred)**: 
            `Head Office Address - Region` OR `(Company) Head Office Address - Region`
        * **Secondary Region (Fallback)**: 
            `Registered Address - Region` OR `(Company) Registered Address - Region`
    """)

    # Use st.info to explain the critical merge logic
    st.info("""
        💡 **Mapping Logic:**
        The app uses the **Primary Region** first, and falls back to 
        the **Secondary Region** if the primary field is blank.
    """)

    # Place the uploader prominently at the end of the instructions
    uploaded = st.file_uploader("Drag and drop your file below:", type=["csv", "xlsx", "xls"])

# Stop execution if no file is uploaded yet
if not uploaded:
    st.stop()


# --------------------------- Load file (MUST BE HERE, AFTER st.stop()) ---------------------------
ext = uploaded.name.split(".")[-1].lower()

if ext == "csv":
    df = pd.read_csv(uploaded)
else:
    # Excel: let user choose which sheet to use (Sheet selection must be in the main area OR sidebar)
    xls = pd.ExcelFile(uploaded, engine="openpyxl")
    # Placing sheet selection in the main area here is fine, as it's the first step post-upload
    sheet_name = st.selectbox("Choose a sheet", options=xls.sheet_names, index=0)
    df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")


# --------------------------- Secondary UI (DATA-DEPENDENT: AGGREGATION & FILTERING) ---------------------------
# All controls that use `df` or rely on the data being loaded must be inside this conditional block.
with st.sidebar:
    st.markdown("---")
    st.header("2. Configure Metrics & Filters 🔢")

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
            st.error("No numeric columns available to sum. Stopping.")
            st.stop()
        sum_col = st.selectbox("Numeric column to sum per region:", options=numeric_cols)
        # Tick box for money formatting
        sum_is_money = st.checkbox(
            "Treat summed values as money (£ with k / m / b units, 3 s.f.)",
            value=False
        )

    # --------------------------- Optional filtering (Using Expander for better UX) ---------------------------
    st.markdown("---")
    with st.expander("3. Optional Data Filter 🔎"):
        st.caption("Filter data before aggregation.")
        # Note: filter_col, unique_vals, selected_vals, filter_mode now defined here
        filter_col = st.selectbox("Select a column to filter:", options=df.columns, index=0)
        unique_vals = df[filter_col].dropna().unique()
        if len(unique_vals) > 100:
            st.warning("Too many unique values — showing only the first 100 distinct values.")
            unique_vals = unique_vals[:100]
        selected_vals = st.multiselect("Select values:", options=sorted(unique_vals, key=lambda x: str(x)))
        filter_mode = st.radio("Filter mode:", ["Include", "Exclude"], horizontal=True)

# NOTE: The filtering execution must happen after the controls are defined, but before processing starts.
# Applying the filter here:
if selected_vals:
    original_row_count = len(df) # Storing original count for better feedback
    if filter_mode == "Include":
        df = df[df[filter_col].isin(selected_vals)]
    else:
        df = df[~df[filter_col].isin(selected_vals)]
    st.success(f"Filtered to {len(df)} rows (from {original_row_count} total) based on **{filter_col}** ({filter_mode}).")


# --------------------------- Map Configuration (MOVED TO SIDEBAR) ---------------------------
with st.sidebar:
    st.markdown("---")
    st.header("4. Map Style & Labels 🎨")

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
        horizontal=False # Vertical display is better in the sidebar
    )


# --------------------------- Resolve region columns (supports "(Company) ..." variants) ---------------------------
# This logic is correctly placed here after `df` is defined.
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
# ... (region_mapping remains unchanged) ...
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
# ... (All binning helper functions remain unchanged) ...

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
    lo_e, hi_e = int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1
    cands = sorted({
        m * (10 ** e)
        for e in range(lo_e, hi_e + 1)
        for m in (1, 2, 5)
        if lo <= m * (10 ** e) <= hi
    })
    if len(cands) >= (k - 1):
        step = len(cands) / (k - 1)
        edges = [cands[int(round((i + 1) * step)) - 1] for i in range(k - 1)]
    else:
        gs = np.geomspace(lo, hi, num=k).tolist()[1:-1]
        edges = sorted(set([max(1, int(x)) for x in gs] + cands))
        while len(edges) < (k - 1):
            edges.append(edges[-1] + 1)
        edges = edges[:k - 1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
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

# --------------------------- Build bins & assign colours (vectorised) ---------------------------
# Light → dark violet gradient
palette = ["#E0DEE9", "#B4B1CE", "#8884B3", "#5C5799", "#302A7E"]

pos_bins = build_bins(g["Region_Value"].values, mode=bin_mode, k=len(palette))
cls = mapclassify.UserDefined(g["Region_Value"].values, bins=pos_bins)
g["bin"] = cls.yb

def pick_colour(row):
    val = float(row["Region_Value"])
    if val == 0:
        return "#F0F0F0"
    idx = row["bin"]
    if idx is None or (isinstance(idx, float) and np.isnan(idx)):
        idx = 0
    idx = max(0, min(int(idx), len(palette) - 1))
    return palette[idx]

g["face_color"] = g.apply(pick_colour, axis=1)

# --------------------------- Plot ---------------------------
# ... (Plotting code remains unchanged) ...

fig, ax = plt.subplots(figsize=(7.5, 8.5))

# Single vectorised plot call for all polygons
g.plot(ax=ax, color=g["face_color"], edgecolor="#4D4D4D", linewidth=0.5)

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
    val = float(r["Region_Value"])
    if name not in label_pos:
        continue

    side, ty = label_pos[name]
    if side == "left":
        lx, tx, ha = bounds[0] - 30000, bounds[0] - 35000, "right"
    else:
        lx, tx, ha = bounds[2] + 30000, bounds[2] + 35000, "left"

    circ = Circle((cx, cy), 5000, facecolor="#FFD40E", edgecolor="black", linewidth=0.5, zorder=10)
    circ.set_path_effects([Stroke(linewidth=1.2, foreground="black"), Normal()])
    ax.add_patch(circ)
    ax.add_line(Line2D([cx, cx], [cy, ty], color="black", linewidth=0.8))
    ax.add_line(Line2D([cx, lx], [ty, ty], color="black", linewidth=0.8))
    ax.text(tx, ty, name, fontsize=11, va="bottom", ha=ha)

    # Label value: raw or %
    if display_mode == "Percentage of total (3 s.f.)":
        label_val = format_pct_3sf(val, _total_value)
    else:
        if agg_mode == "Sum a numeric column" and sum_is_money:
            label_val = format_money_3sf(val)
        else:
            label_val = f"{int(round(val)):,}"
    ax.text(tx, ty - 8000, label_val, fontsize=11, va="top", ha=ha, fontweight="bold")

# Legend (min/max only) – raw values (count or sum)
pos_vals = g.loc[g["Region_Value"] > 0, "Region_Value"]
if len(pos_vals) == 0:
    min_label = "0"
    max_label = "0"
else:
    min_raw, max_raw = float(pos_vals.min()), float(pos_vals.max())
    if agg_mode == "Sum a numeric column" and sum_is_money:
        min_label = format_money_3sf(min_raw)
        max_label = format_money_3sf(max_raw)
    else:
        min_label = f"{int(round(min_raw))}"
        max_label = f"{int(round(max_raw))}"

box_w, start_x, start_y = 0.025, 0.04, 0.90
for i, col in enumerate(palette):
    rect = Rectangle(
        (start_x + i * box_w, start_y),
        box_w,
        box_w,
        transform=fig.transFigure,
        fc=col,
        ec="none",
    )
    fig.patches.append(rect)

ax.text(
    start_x - 0.005,
    start_y + box_w / 2,
    min_label,
    transform=fig.transFigure,
    fontsize=13,
    va="center",
    ha="right",
)
ax.text(
    start_x + len(palette) * box_w + 0.005,
    start_y + box_w / 2,
    max_label,
    transform=fig.transFigure,
    fontsize=13,
    va="center",
    ha="left",
)

ax.set_title(map_title, fontsize=15, fontweight="bold", pad=10)
ax.axis("off")
plt.tight_layout()

# --------------------------- Show & export ---------------------------
st.pyplot(fig, use_container_width=True)

st.markdown("### Export Map")
st.markdown(
    """
<style>
div[data-testid="column"] { flex: 1 1 45% !important; }
div[data-testid="stMarkdownContainer"] h3 { color: #000 !important; margin-bottom: 0.3rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

svg, png = io.BytesIO(), io.BytesIO()
fig.savefig(svg, format="svg", bbox_inches="tight")
svg.seek(0)
fig.savefig(png, format="png", bbox_inches="tight", dpi=300)
png.seek(0)

c1, c2 = st.columns([1, 1])
with c1:
    # IMPROVED EXPORT MICROCOPY
    st.markdown("### Vector Graphic (SVG)")
    st.download_button(
        "Download for Editing (.svg)",
        data=svg,
        file_name="uk_company_map.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )
with c2:
    # IMPROVED EXPORT MICROCOPY
    st.markdown("### High-Resolution Image (PNG)")
    st.download_button(
        "Download for Presentation (.png)",
        data=png,
        file_name="uk_company_map.png",
        mime="image/png",
        use_container_width=True,
    )

# --------------------------- Footer image ---------------------------
st.markdown("---")
st.image(
    "you_are_welcome.png",
    caption="Last updated:24/10/25 -JT",
    use_container_width=False,
    width=175,
)
