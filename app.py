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

# Define the NUTS Level 1 Regions for manual input (Used only in Manual Mode)
NUTS1_REGIONS = [
    "East Midlands (England)", "East of England", "London", "North East (England)",
    "North West (England)", "Northern Ireland", "Scotland", "South East (England)",
    "South West (England)", "Wales", "West Midlands (England)", "Yorkshire and The Humber"
]

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

    # Increased tolerance for faster plotting and loading
    try:
        gdf["geometry"] = gdf["geometry"].simplify(tolerance=2000, preserve_topology=True)
    except Exception:
        pass

    return gdf

# --------------------------- Caching for unique values ---------------------------
@st.cache_data
def get_unique_values(df, col):
    """Caches unique values for a column to speed up filter selector."""
    if col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 100:
            return unique_vals[:100]
        return unique_vals
    return np.array([])

# --------------------------- Caching for Main Processing Pipeline (File Mode) ---------------------------
@st.cache_data
def get_processed_data(df_input, agg_mode, sum_col, region_cols_tuple):
    # This is the original function logic for file processing
    df = df_input.copy()
    
    gdf_regions = load_regions_gdf()
    if gdf_regions is None:
        st.error("Geo-data resource failed to load during processing.")
        st.stop()

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
    
    _total_value = float(g["Region_Value"].sum())
    
    return g, _total_value

# --------------------------- New: Process Manual Data into GeoDataFrame (Used only in Manual Mode) ---------------------------
@st.cache_data
def get_processed_manual_data(input_dict, is_money):
    """Creates the aggregated GeoDataFrame directly from manual input."""
    gdf_regions = load_regions_gdf()
    if gdf_regions is None:
        return None, 0, is_money

    counts = pd.DataFrame(
        list(input_dict.items()),
        columns=["Region_Mapped", "Region_Value"]
    )
    
    counts["Region_Value"] = pd.to_numeric(counts["Region_Value"], errors='coerce').fillna(0)

    g = gdf_regions.merge(counts, left_on="nuts118nm", right_on="Region_Mapped", how="left")
    g["Region_Value"] = g["Region_Value"].fillna(0)
    
    _total_value = float(g["Region_Value"].sum())
    
    return g, _total_value, is_money

# --------------------------- Helpers: formatting ---------------------------
def format_pct_3sf(n, total):
    """Return n/total as a percentage string to 3 significant figures."""
    if total <= 0:
        return "0%"
    pct = 100 * (float(n) / float(total))
    s = f"{pct:.3g}"
    return f"{s}%"

def format_money_3sf(x):
    """
    Format a numeric value as money with £ and units (k, m, b),
    to 3 significant figures.
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
    s = f"{scaled:.3g}"
    sign = "-" if neg else ""
    return f"{sign}£{s}{unit}"

# --------------------------- UI START ---------------------------

# 1. Main Area Headers (Always visible)
st.header("UK Regional Company Generator")

# 2. Load Geographical Data (Dependency Check)
gdf_regions = load_regions_gdf()

if gdf_regions is None:
    st.error("Fatal Error: Map Dependencies Missing")
    st.markdown("The application failed to load the required geographical boundary data (NUTS Level 1 UK regions) from the external source. The app cannot proceed.")
    st.stop()

# Initialize variables to be used by the plotter
g = None
_total_value = 0
agg_mode = "Number of companies (row count)" 
display_mode = "Raw value"
sum_col = None
sum_is_money_tracker = False # Holds the final state of sum_is_money

# --- START SIDEBAR ---
with st.sidebar:
    st.header("1. Choose Data Input Method")
    data_mode = st.radio(
        "Select your input method:",
        ["File Upload (Full App)", "Manual Data Entry (Fast Map)"],
        index=0
    )

    if data_mode == "File Upload (Full App)":
        # --- ORIGINAL FILE UPLOAD UI ---
        
        st.markdown("---")
        st.header("1. Upload Data File")

        st.markdown("""
            To map your data, your file must contain at least **one** column
            from each of the following required groups:
            
            * **Primary Region (Preferred)**: 
                `Head Office Address - Region` OR `(Company) Head Office Address - Region`
            * **Secondary Region (Fallback)**: 
                `Registered Address - Region` OR `(Company) Registered Address - Region`
        """)

        uploaded = st.file_uploader("Drag and drop your file below:", type=["csv", "xlsx", "xls"])

        # Stop execution if no file is uploaded yet
        if not uploaded:
            st.stop()
            
        # --------------------------- Load file (CONDITIONAL LOGIC) ---------------------------
        ext = uploaded.name.split(".")[-1].lower()

        if ext == "csv":
            df = pd.read_csv(uploaded)
        else:
            # Excel Sheet Selection in Sidebar (Step 1b)
            st.markdown("---")
            st.subheader("1b. Choose Sheet")
            xls = pd.ExcelFile(uploaded, engine="openpyxl")
            sheet_name = st.selectbox("Choose a sheet to load:", options=xls.sheet_names, index=0)
            
            df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")

        # --------------------------- Resolve region columns (Prerequisite for Mapping) ---------------------------
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

        # --------------------------- Clean & merge regions (MOVED OUT OF CACHE) ---------------------------
        for c in [head_col, reg_col]:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan, "None": np.nan, "(no value)": np.nan, "": np.nan})
            )

        df["Region (merged)"] = df[head_col].fillna(df[reg_col])

        # --------------------------- Region mapping (MOVED OUT OF CACHE) ---------------------------
        region_mapping = {
            "East Midlands": "East Midlands (England)", "East of England": "East of England",
            "London": "London", "North East": "North East (England)",
            "North West": "North West (England)", "Northern Ireland": "Northern Ireland",
            "Scotland": "Scotland", "South East": "South East (England)",
            "South West": "South West (England)", "Wales": "Wales",
            "West Midlands": "West Midlands (England)", "Yorkshire and The Humber": "Yorkshire and The Humber",
            # Scotland subregions -> Scotland
            "West of Scotland": "Scotland", "East of Scotland": "Scotland", 
            "South of Scotland": "Scotland", "Highlands and Islands": "Scotland", 
            "Tayside": "Scotland", "Aberdeen": "Scotland",
        }
        df["Region_Mapped"] = df["Region (merged)"].map(region_mapping).fillna("Unknown")
        
        # Apply filter outside sidebar, but define controls inside
        
        # --------------------------- Secondary UI (DATA-DEPENDENT: AGGREGATION & FILTERING) ---------------------------
        st.markdown("---")
        st.header("2. Configure Metrics & Filters")

        # --------------------------- Aggregation mode (count vs sum) ---------------------------
        agg_mode = st.radio(
            "What metric should the map show?",
            ["Number of companies (row count)", "Sum a numeric column"],
            index=0
        )

        sum_col = None

        if agg_mode == "Sum a numeric column":
            # Note: numeric_cols is safe to calculate here as df is now loaded and mapped
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns available to sum. Stopping.")
                st.stop()
            sum_col = st.selectbox("Numeric column to sum per region:", options=numeric_cols)
            
            sum_is_money_tracker = st.checkbox(
                "Treat summed values as money (£ with k / m / b units, 3 s.f.)",
                value=False
            )

        # --------------------------- Optional filtering (Using Expander for better UX) ---------------------------
        st.markdown("---")
        with st.expander("3. Optional Data Filter"):
            st.caption("Filter data before aggregation.")
            # Note: filter_col is now based on the original df, which is fine
            filter_col = st.selectbox("Select a column to filter:", options=df.columns, index=0)
            
            unique_vals = get_unique_values(df, filter_col)
            
            selected_vals = st.multiselect("Select values:", options=sorted(unique_vals, key=lambda x: str(x)))
            filter_mode = st.radio("Filter mode:", ["Include", "Exclude"], horizontal=True)

        # Map Configuration
        st.markdown("---")
        st.header("4. Map Style & Labels")

        st.info("Color scheme set to **Natural Breaks** (Fisher-Jenks) for optimal visualization.")

        map_title = st.text_input("Enter your custom map title:", "UK Company Distribution by NUTS Level 1 Region")

        display_mode = st.radio(
            "Display values as:",
            ["Raw value", "Percentage of total"],
            horizontal=False
        )
        
        st.markdown("---")

        # NOTE: Applying the filter here (after controls are defined, before region processing)
        if selected_vals:
            original_row_count = len(df)    
            # Create a copy of df (df_filtered) and apply the filter
            df_filtered = df[df[filter_col].isin(selected_vals)].copy() if filter_mode == "Include" else df[~df[filter_col].isin(selected_vals)].copy()
            st.success(f"Filtered to {len(df_filtered)} rows (from {original_row_count} total) based on **{filter_col}** ({filter_mode}).")
        else:
            df_filtered = df.copy() # Use a copy of the full DataFrame if no filter is applied

        # CALL CACHED PROCESSING FUNCTION for file mode
        region_cols_tuple = tuple(sorted(resolved_cols.items()))

        with st.spinner("Processing data, mapping regions, and aggregating values..."):
            g, _total_value = get_processed_data(
                df_filtered,  
                agg_mode,  
                sum_col,  
                region_cols_tuple
            )


    else: # Manual Data Entry (Fast Map)
        # --- MANUAL ENTRY UI ---
        
        st.markdown("---")
        st.subheader("1a. Enter Regional Values")
        st.markdown("**Enter a number for each of the 12 UK NUTS 1 Regions.**")

        manual_input_dict = {}
        cols = st.columns(2)
        
        # --- CHANGE 1: CLEARLY LABELED INPUT BOXES ---
        for i, region in enumerate(NUTS1_REGIONS):
            # Display name without (England) suffix for cleaner labels
            display_name = region.replace(" (England)", "")
            with cols[i % 2]:
                # Use the region name as the label
                manual_input_dict[region] = st.text_input(display_name, value="0", key=region)
            
        st.markdown("---")
        sum_is_money_manual = st.checkbox(
            "Treat values as money (£ with k / m / b units, 3 s.f.)",
            value=False
        )
        
        # Process the manual data
        with st.spinner("Processing manual input..."):
            g, _total_value, sum_is_money_tracker = get_processed_manual_data(manual_input_dict, sum_is_money_manual)
            
        if g is None:
            st.error("Failed to process manual data.")
            st.stop()

        # Manual Map Configuration (Simplified)
        st.markdown("---")
        st.header("2. Map Style & Labels")
        
        # --- CHANGE 2: REMOVED INFO BOXES ---
        # The info boxes (Metric is Raw Value, Color Scheme) are now gone.
        
        map_title = st.text_input("Enter your custom map title:", "UK Data Distribution by NUTS Level 1 Region")

        display_mode = st.radio(
            "Display values as:",
            ["Raw value", "Percentage of total"],
            horizontal=False
        )
        st.markdown("---")


# --------------------------- PLOTTING LOGIC (COMMON FOR BOTH MODES) ---------------------------

# Binning helpers (remain outside cache)
def bins_equal_interval(pos_vals, k=5):
    lo, hi = float(np.min(pos_vals)), float(np.max(pos_vals))
    if lo == hi:
        edges = [hi] * (k - 1)
    else:
        step = (hi - lo) / k
        edges = [lo + step * i for i in range(1, k)]
    return [*edges, np.inf]

def bins_fisher_jenks(pos_vals, k=5):
    u = np.unique(pos_vals)
    k_eff = int(min(k, max(2, len(u))))
    try:
        fj = mapclassify.FisherJenks(pos_vals, k=k_eff)
        bins = list(fj.bins[:-1]) + [np.inf]
        return bins
    except Exception:
        return bins_equal_interval(pos_vals, k)

def build_bins(values, mode="Natural Breaks (Fisher-Jenks)", k=5):
    vals = np.asarray(values, dtype=float)
    pos = vals[vals > 0]
    if len(pos) == 0:
        return [1, 2, 3, 4, np.inf]
    # For this application, we only use Fisher-Jenks
    return bins_fisher_jenks(pos, k)

bin_mode = "Natural Breaks (Fisher-Jenks)"

# Build bins & assign colours (vectorised)
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
fig, ax = plt.subplots(figsize=(7.5, 8.5))

# Single vectorised plot call for all polygons
g.plot(ax=ax, color=g["face_color"], edgecolor="#4D4D4D", linewidth=0.5)

bounds = g.total_bounds

# Labels & callouts (RESTORED TO ORIGINAL: NAME + VALUE)
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

    # Label value: raw or % (RESTORED VALUE DISPLAY)
    if display_mode == "Percentage of total":
        label_val = format_pct_3sf(val, _total_value)
    else:
        # Use the unified tracker for formatting
        if sum_is_money_tracker:
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
    if sum_is_money_tracker:
        min_label = format_money_3sf(min_raw)
        max_label = format_money_3sf(max_raw)
    else:
        min_label = f"{int(round(min_raw)):,}"
        max_label = f"{int(round(max_raw)):,}"

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
    st.markdown("### Editable Source File (.svg)")
    st.download_button(
        "Download SVG",
        data=svg,
        file_name="uk_company_map.svg",
        mime="image/svg+xml",
        use_container_width=True,
    )
with c2:
    st.markdown("### Image for Presentation (.png)")
    st.download_button(
        "Download PNG",
        data=png,
        file_name="uk_company_map.png",
        mime="image/png",
        use_container_width=True,
    )

# --------------------------- Footer image ---------------------------
st.markdown("---")
st.caption("Last updated:24/10/25 -JT")
