# --- Unified Data Refresh Function ---
def refresh_all_data():
    """Clear all cached data and force reload from files."""
    try:
        # Clear transaction data cache
        load_all_transactions.clear()
        
        # Clear listings data cache
        load_all_listings.clear()
        
        # Clear monthly data cache
        get_monthly_df.clear()
        
        # Clear similarity model cache
        get_similarity_model.clear()
        
        # Clear Prophet model cache
        load_prophet_model.clear()
        
        # Clear any other cached data
        st.cache_data.clear()
        st.cache_resource.clear()
        
        st.success("✅ All cached data cleared! Data will reload on next interaction.")
        
    except Exception as e:
        st.error(f"❌ Error clearing cache: {str(e)}")
        st.info("💡 Try refreshing the page manually.")

import streamlit as st
import os
from streamlit import cache_data
import pandas as pd
all_listings = pd.DataFrame()
import re
import math
# --- Required for Similarity Matching ---
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

def get_location_str(row):
    comm = row.get('Community') or row.get('Community/Building')
    subcomm = row.get('Subcommunity') or row.get('Sub Community / Building')
    comm = str(comm) if pd.notnull(comm) else ''
    subcomm = str(subcomm) if pd.notnull(subcomm) else ''
    if comm and subcomm:
        return f"{comm}, {subcomm}"
    elif comm:
        return comm
    elif subcomm:
        return subcomm
    else:
        return ''
def tune_prophet_hyperparameters(monthly_df):
    # Your implementation or stub
    return None, None

# --- Similarity Model Caching ---
@st.cache_resource(ttl=3600)  # 1 hour TTL
def get_similarity_model(df, numeric_features, categorical_features, cache_version=1):
    """Preprocess and fit KNN on given DataFrame df for similarity matching."""
    sim_df = df.copy()
    # Coerce numeric features
    for col in numeric_features:
        if col in sim_df.columns:
            sim_df[col] = pd.to_numeric(sim_df[col], errors='coerce')
    sim_df = sim_df.dropna(subset=[col for col in numeric_features if col in sim_df.columns])
    # Fill missing categoricals
    for col in categorical_features:
        if col in sim_df.columns:
            sim_df[col] = sim_df[col].fillna("UNKNOWN")
    # Build pipeline
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))
    preprocessor = ColumnTransformer(transformers)
    X = preprocessor.fit_transform(sim_df)
    knn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    knn.fit(X)
    # Return the processed sim_df, preprocessor, and knn model
    return sim_df, preprocessor, knn

@st.cache_resource
def load_prophet_model(
    growth, n_changepoints, changepoint_range, changepoint_prior_scale,
    yearly_seasonality, weekly_seasonality, daily_seasonality,
    seasonality_prior_scale, interval_width, cap, monthly_df, uncertainty_samples=1000
):
    """Cache and return a fitted Prophet model."""
    # Ensure seasonality arguments are booleans
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)
   
    # Build Prophet parameters dict, only including seasonality if enabled
    prophet_params = {
        'growth': growth,
        'n_changepoints': n_changepoints,
        'changepoint_range': changepoint_range,
        'changepoint_prior_scale': changepoint_prior_scale,
        'seasonality_prior_scale': seasonality_prior_scale,
        'interval_width': interval_width,
        'uncertainty_samples': uncertainty_samples
    }
    
    # Only add seasonality parameters if they are enabled
    if to_bool(yearly_seasonality):
        prophet_params['yearly_seasonality'] = "auto"
    if to_bool(weekly_seasonality):
        prophet_params['weekly_seasonality'] = "auto"
    if to_bool(daily_seasonality):
        prophet_params['daily_seasonality'] = "auto"
    
    m_temp = Prophet(**prophet_params)  # type: ignore
    
    if growth == "logistic" and cap is not None:
        monthly_df['cap'] = cap
    m_temp.fit(monthly_df)
    return m_temp


import streamlit.components.v1 as components
# AgGrid for interactive tables in tab2
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import numpy as np
import plotly.graph_objects as go


from sklearn.linear_model import LinearRegression
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
 
import numpy as np
from datetime import datetime, timedelta
import time
import json

def prepare_prophet_df(df):
    df2 = df.dropna(subset=['Evidence Date', 'Price (AED/sq ft)']).copy()
    df2['ds'] = pd.to_datetime(df2['Evidence Date'])
    df2['y'] = pd.to_numeric(df2['Price (AED/sq ft)'], errors='coerce')
    # Use median instead of mean for monthly aggregation
    monthly = df2.set_index('ds')['y'].resample('ME').median().reset_index()
    monthly['y_avg'] = df2['y'].median()
    return monthly

# --- Cache monthly DataFrame preparation for Prophet ---
@st.cache_data
def get_monthly_df(df, last_n_days):
    """
    Filter df by Evidence Date >= today - last_n_days and return monthly aggregated DataFrame for Prophet.
    """
    # Copy to avoid mutating original
    df_filt = df.copy()
    if 'Evidence Date' in df_filt.columns:
        df_filt['Evidence Date'] = pd.to_datetime(df_filt['Evidence Date'], errors='coerce')
        cutoff = datetime.now() - timedelta(days=last_n_days)
        df_filt = df_filt[df_filt['Evidence Date'] >= cutoff]
    # Use prepare_prophet_df to aggregate
    return prepare_prophet_df(df_filt)



# --- Cached loader for all transaction files ---
@cache_data
def load_all_transactions(transactions_dir):
    """Load and concatenate all Excel files in the transactions directory, with caching."""
    files = [f for f in os.listdir(transactions_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    dfs = []
    for file in files:
        file_path = os.path.join(transactions_dir, file)
        try:
            df = pd.read_excel(file_path)
            # Normalize column names
            df.columns = df.columns.str.replace('\xa0', '', regex=False).str.strip()
            # Rename misnamed EvidenceDate to Evidence Date, if not already present
            if 'EvidenceDate' in df.columns and 'Evidence Date' not in df.columns:
                df = df.rename(columns={'EvidenceDate': 'Evidence Date'})
            # Drop any duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            df['Source File'] = file
            dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to load {file}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _reset_filters():
    # Preserve only sales_recurrence
    saved = st.session_state.get("sales_recurrence", "All")
    # Clear all session state
    st.session_state.clear()
    # Restore sales_recurrence
    st.session_state["sales_recurrence"] = saved

def _on_development_change():
    """Callback function for development selection changes"""
    # Use state coordination to prevent race conditions
    if not should_process_callback("development", 500):
        return
    
    # Set flag to prevent other callbacks from interfering
    set_filter_update_flag()
    
    # Clear dependent filters
    if "community" in st.session_state:
        st.session_state.pop("community")
    if "subcommunity" in st.session_state:
        st.session_state.pop("subcommunity")
    if "layout_type" in st.session_state:
        st.session_state.pop("layout_type")
    if "unit_type" in st.session_state:
        st.session_state.pop("unit_type")
    
    # Clear flag
    clear_filter_update_flag()

def _on_community_change():
    """Callback function for community selection changes"""
    # Use state coordination to prevent race conditions
    if not should_process_callback("community", 500):
        return
    
    # Set flag to prevent other callbacks from interfering
    set_filter_update_flag()
    
    # Clear dependent filters
    if "subcommunity" in st.session_state:
        st.session_state.pop("subcommunity")
    if "layout_type" in st.session_state:
        st.session_state.pop("layout_type")
    if "unit_type" in st.session_state:
        st.session_state.pop("unit_type")
    
    # Clear flag
    clear_filter_update_flag()

def _on_subcommunity_change():
    """Callback function for subcommunity selection changes"""
    # Use state coordination to prevent race conditions
    if not should_process_callback("subcommunity", 500):
        return
    
    # Set flag to prevent other callbacks from interfering
    set_filter_update_flag()
    
    # Clear dependent filters
    if "layout_type" in st.session_state:
        st.session_state.pop("layout_type")
    if "unit_type" in st.session_state:
        st.session_state.pop("unit_type")
    
    # Clear flag
    clear_filter_update_flag()

def _on_bedrooms_change():
    """Callback function for bedrooms selection changes"""
    # Use state coordination to prevent race conditions
    if not should_process_callback("bedrooms", 500):
        return
    
    # Set flag to prevent other callbacks from interfering
    set_filter_update_flag()
    
    # Clear dependent filters
    if "layout_type" in st.session_state:
        st.session_state.pop("layout_type")
    if "unit_type" in st.session_state:
        st.session_state.pop("unit_type")
    
    # Clear flag
    clear_filter_update_flag()

# --- Page Config ---
st.set_page_config(page_title="Valuation App V2", layout="wide")

# --- State Coordination System ---
def initialize_state_coordination():
    """Initialize state coordination to prevent race conditions"""
    if "state_coordination" not in st.session_state:
        st.session_state["state_coordination"] = {
            "last_update": time.time(),
            "updating_filters": False,
            "filter_update_queue": [],
            "debounce_timers": {}
        }

def should_process_callback(callback_name, debounce_ms=500):
    """Check if callback should be processed based on debouncing"""
    coord = st.session_state.get("state_coordination", {})
    timers = coord.get("debounce_timers", {})
    
    current_time = time.time()
    last_time = timers.get(callback_name, 0)
    
    if current_time - last_time < (debounce_ms / 1000):
        return False
    
    # Update timer
    timers[callback_name] = current_time
    coord["debounce_timers"] = timers
    st.session_state["state_coordination"] = coord
    
    return True

def set_filter_update_flag():
    """Set flag to indicate filter update in progress"""
    coord = st.session_state.get("state_coordination", {})
    coord["updating_filters"] = True
    coord["last_update"] = time.time()
    st.session_state["state_coordination"] = coord

def clear_filter_update_flag():
    """Clear flag when filter update is complete"""
    coord = st.session_state.get("state_coordination", {})
    coord["updating_filters"] = False
    st.session_state["state_coordination"] = coord
    
    # Clear filter caches to force recalculation
    cache_keys_to_clear = [key for key in st.session_state.keys() if isinstance(key, str) and key.startswith(('filters_', 'no_time_filters_'))]
    for key in cache_keys_to_clear:
        st.session_state.pop(key, None)

def is_filter_update_in_progress():
    """Check if a filter update is currently in progress"""
    coord = st.session_state.get("state_coordination", {})
    return coord.get("updating_filters", False)

# Initialize state coordination
initialize_state_coordination()

# --- Load Transaction Data from Data/Transactions (cached) ---
transactions_dir = os.path.join(os.path.dirname(__file__), 'Data', 'Transactions')
# Initialize transaction file selection
txn_files = [f for f in os.listdir(transactions_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
if "included_txn_files" not in st.session_state:
    st.session_state["included_txn_files"] = txn_files

# Check if we need to reload data (only if file selection changed)
data_cache_key = f"data_{hash(tuple(sorted(st.session_state.get('included_txn_files', []))))}"
if data_cache_key not in st.session_state:
    with st.spinner("Loading transaction files..."):
        # Load all, then filter by user selection
        all_transactions = load_all_transactions(transactions_dir)
        all_transactions = all_transactions[all_transactions["Source File"].isin(st.session_state["included_txn_files"])]
        st.session_state[data_cache_key] = all_transactions
else:
    all_transactions = st.session_state[data_cache_key]
# --- After loading transactions, load rental data and extract contract dates ---
if not all_transactions.empty:
    # Parse Evidence Date once for filtering
    if 'Evidence Date' in all_transactions.columns:
        all_transactions['Evidence Date'] = pd.to_datetime(all_transactions['Evidence Date'], errors='coerce')
    # Count unique source files actually loaded
    import pandas as pd
    source_file_col = all_transactions['Source File']
    if not isinstance(source_file_col, pd.Series):
        source_file_col = pd.Series(source_file_col)
    file_count = int(source_file_col.nunique())
    st.success(f"Loaded {file_count} transaction file(s).")
else:
    st.warning("No transaction data loaded.")

# --- Load rental data and extract contract dates ---
rentals_dir = os.path.join(os.path.dirname(__file__), "Data", "Rentals")
rental_files = [f for f in os.listdir(rentals_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
rental_dfs = []
for file in rental_files:
    file_path = os.path.join(rentals_dir, file)
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.replace('\xa0', '', regex=False).str.strip()
        # Normalize unit numbers for matching
        df['Unit No.'] = df['Unit No.'].astype(str).str.strip().str.upper()
        # Split Evidence Date into start/end if present
        if 'Evidence Date' in df.columns:
            df[['Contract Start', 'Contract End']] = (
                df['Evidence Date']
                .astype(str)
                .str.replace('\u00A0', '', regex=False)
                .str.split('/', expand=True)
            )
            df['Contract Start'] = pd.to_datetime(
                df['Contract Start'].str.strip(), errors='coerce', dayfirst=True
            )
            df['Contract End'] = pd.to_datetime(
                df['Contract End'].str.strip(), errors='coerce', dayfirst=True
            )
        rental_dfs.append(df)
    except Exception as e:
        st.warning(f"Failed to load rental data from {file}: {e}")
if rental_dfs:
    rental_df = pd.concat(rental_dfs, ignore_index=True)
else:
    import pandas as pd
    rental_df = pd.DataFrame(columns=pd.Index(['Unit No.', 'Contract Start', 'Contract End']))

# --- Cached loader for all transaction files ---
@cache_data
def load_all_listings(listings_dir):
    """Load and concatenate all Excel listing files, with caching."""
    # Check if directory exists, if not return empty DataFrame
    if not os.path.exists(listings_dir):
        return pd.DataFrame()
    
    try:
        files = [f for f in os.listdir(listings_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    except Exception as e:
        st.warning(f"Failed to access listings directory: {e}")
        return pd.DataFrame()
    
    dfs = []
    for file in files:
        path = os.path.join(listings_dir, file)
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.replace('\xa0', '', regex=False).str.strip()
            df['Source File'] = file
            dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to load listing {file}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# --- Cached loader for all rent listing files ---
@cache_data
def load_all_rent_listings(rent_listings_dir):
    """Load and concatenate all Excel rent listing files, with caching."""
    # Check if directory exists, if not return empty DataFrame
    if not os.path.exists(rent_listings_dir):
        return pd.DataFrame()
    
    try:
        files = [f for f in os.listdir(rent_listings_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    except Exception as e:
        st.warning(f"Failed to access rent listings directory: {e}")
        return pd.DataFrame()
    
    dfs = []
    for file in files:
        path = os.path.join(rent_listings_dir, file)
        try:
            df = pd.read_excel(path)
            df.columns = df.columns.str.replace('\xa0', '', regex=False).str.strip()
            df['Source File'] = file
            dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to load rent listing {file}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

 # --- Load Layout Types Mapping ---
layout_dir = os.path.join(os.path.dirname(__file__), 'Data', 'Layout Types')
layout_files = [f for f in os.listdir(layout_dir) if f.endswith('.xlsx')]
layout_dfs = []
details_dfs = []
for file in layout_files:
    file_path = os.path.join(layout_dir, file)
    try:
        # Load Types sheet (for legacy mapping)
        df_l = pd.read_excel(file_path, sheet_name='Types')
        df_l.columns = df_l.columns.str.replace('\xa0', '', regex=False).str.strip()
        project_name = os.path.splitext(file)[0]
        if project_name.lower().endswith('_layouts'):
            project_name = project_name[:-len('_layouts')]
        df_l['Project'] = project_name
        layout_dfs.append(df_l)
    except Exception:
        continue
    try:
        # Load Details sheet (for new context-aware mapping)
        df_d = pd.read_excel(file_path, sheet_name='Details')
        df_d.columns = df_d.columns.str.replace('\xa0', '', regex=False).str.strip()
        df_d['Project'] = project_name
        details_dfs.append(df_d)
    except Exception:
        continue
if layout_dfs:
    layout_map_df = pd.concat(layout_dfs, ignore_index=True).drop_duplicates(subset=['Unit No.'])
    layout_map = dict(zip(layout_map_df['Unit No.'], layout_map_df['Layout Type']))
else:
    layout_map_df = pd.DataFrame(columns=pd.Index(['Unit No.', 'Layout Type', 'Project']))
    layout_map = {}
if details_dfs:
    details_df = pd.concat(details_dfs, ignore_index=True)
else:
    details_df = pd.DataFrame(columns=pd.Index(['Development', 'Community', 'Subcommunity', 'Layout Type', 'Beds', 'BUA', 'Type', 'Project']))

def get_unit_details(dev, comm, subcomm, layout_type):
    """Return dict of details for a given context (dev, comm, subcomm, layout_type) from details_df."""
    if details_df.empty:
        return {}
    
    # Map expected column names to actual column names in the layout files
    dev_col = 'Development' if 'Development' in details_df.columns else 'All Developments'
    comm_col = 'Community' if 'Community' in details_df.columns else 'Community/Building'
    subcomm_col = 'Subcommunity' if 'Subcommunity' in details_df.columns else 'Sub Community / Building'
    
    mask = (
        (details_df[dev_col] == dev) &
        (details_df[comm_col] == comm) &
        (details_df[subcomm_col] == subcomm) &
        (details_df['Layout Type'] == layout_type)
    )
    row = details_df[mask]
    if not row.empty:
        r = row.iloc[0]
        return dict(Beds=r.get('Beds'), BUA=r.get('BUA'), Type=r.get('Type'))
    return {}

# Add Layout Type to transactions
if not all_transactions.empty:
    # Clean unit numbers for strict, case-insensitive match
    import pandas as pd
    unit_no_col = all_transactions['Unit No.']
    if not isinstance(unit_no_col, pd.Series):
        unit_no_col = pd.Series(unit_no_col)
    all_transactions['Unit No.'] = unit_no_col.astype(str).str.strip().str.upper()
    layout_unit_no_col = layout_map_df['Unit No.']
    if not isinstance(layout_unit_no_col, pd.Series):
        layout_unit_no_col = pd.Series(layout_unit_no_col)
    layout_map_df['Unit No.'] = layout_unit_no_col.astype(str).str.strip().str.upper()
    layout_map = dict(zip(layout_map_df['Unit No.'], layout_map_df['Layout Type']))
    unit_no_series = all_transactions['Unit No.']
    if not isinstance(unit_no_series, pd.Series):
        unit_no_series = pd.Series(unit_no_series)
    all_transactions['Layout Type'] = unit_no_series.map(layout_map).fillna('')
else:
    all_transactions['Layout Type'] = ''

# --- Sidebar ---

with st.sidebar:
    st.title("Valuation Controls")

    if st.button("🔄 Reset Filters", key="reset_filters"):
        _reset_filters()

    # Unified refresh button for all data
    if st.button("🔄 Refresh All Data", key="refresh_all"):
        refresh_all_data()

    with st.expander("📂 Select Transaction Files", expanded=False):
        all_txn_files = [f for f in os.listdir(transactions_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
        st.multiselect(
            "Include transaction files:",
            options=all_txn_files,
            key="included_txn_files"
        )

    # --- Property Filters ---
    # Get values from session state (auto-filled from unit selection)
        property_type = st.session_state.get("property_type", "")
        bedrooms = st.session_state.get("bedrooms", "")
        bua = st.session_state.get("bua", "")
        plot_size = st.session_state.get("plot_size", "")

    # --- Location Filters ---
    st.subheader("Location")
    if not all_transactions.empty:
        dev_options = sorted(pd.Series(all_transactions['All Developments']).dropna().unique())
        com_options = sorted(pd.Series(all_transactions['Community/Building']).dropna().unique())
        subcom_options = sorted(pd.Series(all_transactions['Sub Community / Building']).dropna().unique())
    else:
        dev_options = com_options = subcom_options = []

    # Get current values from session state
    current_development = st.session_state.get("development", "")
    current_community = st.session_state.get("community", [])
    current_subcommunity = st.session_state.get("subcommunity", [])

    # Get options for property and sales recurrence from data
    if not all_transactions.empty:
        prop_type_col = all_transactions['Unit Type']
        if not isinstance(prop_type_col, pd.Series):
            prop_type_col = pd.Series(prop_type_col)
        prop_type_options = sorted(prop_type_col.dropna().unique())
        sales_rec_col = all_transactions['Sales Recurrence']
        if not isinstance(sales_rec_col, pd.Series):
            sales_rec_col = pd.Series(sales_rec_col)
        sales_rec_options = ['All'] + sorted(sales_rec_col.dropna().unique())
    else:
        prop_type_options = []
        sales_rec_options = ['All']

    # --- Determine selected values for autofill ---
    # (Removed all logic involving unit_number)

    # --- Location selectors (always enabled) ---
    development = st.session_state.get("development", "")
    community = st.session_state.get("community", [])
    subcommunity = st.session_state.get("subcommunity", [])

    development = st.selectbox(
        "Development",
        options=[""] + dev_options,
        index=([""] + dev_options).index(development) if development in dev_options else 0,
        key="development",
        on_change=_on_development_change,
        placeholder=""
    )
    # --- Community Filter ---
    if not all_transactions.empty:
        community_col = all_transactions['Community/Building']
        if not isinstance(community_col, pd.Series):
            community_col = pd.Series(community_col)
        community_options = sorted(community_col.dropna().unique())
    else:
        community_options = []
    current_community = st.session_state.get("community", [])
    community = st.multiselect(
        "Community",
        options=community_options,
        default=current_community if current_community and all(c in community_options for c in current_community) else [],
        key="community",
        on_change=_on_community_change
    )

    # --- Subcommunity Filter (context-aware) ---
    layout_df_filtered = layout_map_df.copy()
    if community:
        if not isinstance(community, list):
            if isinstance(community, (np.ndarray, pd.Series)):
                community = list(map(str, community.tolist()))
            else:
                community = [str(community)]
        if not isinstance(layout_df_filtered, pd.DataFrame):
            layout_df_filtered = pd.DataFrame(layout_df_filtered)
        comm_col = layout_df_filtered['Community/Building'] if 'Community/Building' in layout_df_filtered.columns else pd.Series([])
        if not isinstance(comm_col, pd.Series):
            if isinstance(comm_col, np.ndarray):
                comm_col = pd.Series(comm_col)
            else:
                comm_col = pd.Series([comm_col])
        layout_df_filtered = layout_df_filtered[comm_col.isin(community)]
    # --- Subcommunity Filter (context-aware) ---
    subcom_col = layout_df_filtered['Sub Community / Building'] if 'Sub Community / Building' in layout_df_filtered.columns else pd.Series([])
    if not isinstance(subcom_col, pd.Series):
        subcom_col = pd.Series(subcom_col)
    subcom_options = sorted(subcom_col.dropna().unique())
    current_subcommunity = st.session_state.get("subcommunity", [])
    subcommunity = st.multiselect(
        "Sub community / Building",
        options=subcom_options,
        default=current_subcommunity if current_subcommunity and all(s in subcom_options for s in current_subcommunity) else [],
        key="subcommunity"
    )

    # --- Bedrooms Filter (context-aware) ---
    bedrooms_df = all_transactions.copy()
    if not isinstance(bedrooms_df, pd.DataFrame):
        bedrooms_df = pd.DataFrame(bedrooms_df)
    if development:
        filter_mask = bedrooms_df['All Developments'] == development
        if not isinstance(filter_mask, pd.Series):
            filter_mask = pd.Series(filter_mask)
        bedrooms_df = bedrooms_df[filter_mask]
        if not isinstance(bedrooms_df, pd.DataFrame):
            bedrooms_df = pd.DataFrame(bedrooms_df)
    if community:
        community_col = bedrooms_df['Community/Building']
        if not isinstance(community_col, pd.Series):
            community_col = pd.Series(community_col)
        filter_mask = community_col.isin(community)
        if not isinstance(filter_mask, pd.Series):
            filter_mask = pd.Series(filter_mask)
        bedrooms_df = bedrooms_df[filter_mask]
        if not isinstance(bedrooms_df, pd.DataFrame):
            bedrooms_df = pd.DataFrame(bedrooms_df)
    if subcommunity:
        subcommunity_col = bedrooms_df['Sub Community / Building']
        if not isinstance(subcommunity_col, pd.Series):
            subcommunity_col = pd.Series(subcommunity_col)
        filter_mask = subcommunity_col.isin(subcommunity)
        if not isinstance(filter_mask, pd.Series):
            filter_mask = pd.Series(filter_mask)
        bedrooms_df = bedrooms_df[filter_mask]
        if not isinstance(bedrooms_df, pd.DataFrame):
            bedrooms_df = pd.DataFrame(bedrooms_df)
    # Get beds column with error handling
    try:
        if 'Beds' in bedrooms_df.columns:
            beds_col = bedrooms_df['Beds']
            # Convert to pandas Series and handle any data type issues
            if not isinstance(beds_col, pd.Series):
                beds_col = pd.Series(beds_col)
            # Convert to string and filter out problematic values
            beds_col_str = beds_col.astype(str)
            beds_col_clean = beds_col_str[beds_col_str != 'nan']
            beds_col_clean = beds_col_clean[beds_col_clean != 'None']
            beds_col_clean = beds_col_clean[beds_col_clean != '']
            # Ensure we have a pandas Series before calling unique() and empty
            if not isinstance(beds_col_clean, pd.Series):
                beds_col_clean = pd.Series(beds_col_clean)
            beds_options = sorted(beds_col_clean.unique()) if not beds_col_clean.empty else []
        else:
            beds_options = []
    except Exception:
        # Fallback to empty list if anything goes wrong
        beds_options = []
    current_bedrooms = st.session_state.get("bedrooms", "")
    bedrooms = st.selectbox(
        "Bedrooms",
        options=[""] + beds_options,
        index=([""] + beds_options).index(current_bedrooms) if current_bedrooms in beds_options else 0,
        key="bedrooms",
        on_change=_on_bedrooms_change,
        placeholder=""
    )

    # --- Layout Type Filter (context-aware) ---
    layout_df_filtered = layout_map_df.copy()
    if community:
        if not isinstance(community, list):
            if isinstance(community, (np.ndarray, pd.Series)):
                community = list(map(str, community.tolist()))
            else:
                community = [str(community)]
        if not isinstance(layout_df_filtered, pd.DataFrame):
            layout_df_filtered = pd.DataFrame(layout_df_filtered)
        comm_col = layout_df_filtered['Community/Building'] if 'Community/Building' in layout_df_filtered.columns else pd.Series([])
        if not isinstance(comm_col, pd.Series):
            if isinstance(comm_col, np.ndarray):
                comm_col = pd.Series(comm_col)
            else:
                comm_col = pd.Series([comm_col])
        layout_df_filtered = layout_df_filtered[comm_col.isin(community)]
    if subcommunity:
        if not isinstance(subcommunity, list):
            if isinstance(subcommunity, (np.ndarray, pd.Series)):
                subcommunity = list(map(str, subcommunity.tolist()))
            else:
                subcommunity = [str(subcommunity)]
        if not isinstance(layout_df_filtered, pd.DataFrame):
            layout_df_filtered = pd.DataFrame(layout_df_filtered)
        subcom_col = layout_df_filtered['Sub Community / Building'] if 'Sub Community / Building' in layout_df_filtered.columns else pd.Series([])
        if not isinstance(subcom_col, pd.Series):
            if isinstance(subcom_col, np.ndarray):
                subcom_col = pd.Series(subcom_col)
            else:
                subcom_col = pd.Series([subcom_col])
        layout_df_filtered = layout_df_filtered[subcom_col.isin(subcommunity)]
    if bedrooms:
        if not isinstance(layout_df_filtered, pd.DataFrame):
            layout_df_filtered = pd.DataFrame(layout_df_filtered)
        if 'Beds' in layout_df_filtered.columns:
            layout_df_filtered = layout_df_filtered[layout_df_filtered['Beds'].astype(str) == str(bedrooms)]
    layout_type_col = layout_df_filtered['Layout Type'] if 'Layout Type' in layout_df_filtered.columns else pd.Series([])
    layout_type_col = pd.Series(layout_type_col)
    layout_options = sorted(layout_type_col.dropna().unique())
    current_layout_type = st.session_state.get("layout_type", [])
    layout_type = st.multiselect(
        "Layout Type",
        options=layout_options,
        default=current_layout_type if current_layout_type and all(l in layout_options for l in current_layout_type) else [],
        key="layout_type"
    )

    # --- Unit Type Filter (context-aware) ---
    unit_type_df = layout_df_filtered.copy()
    if layout_type:
        if not isinstance(layout_type, list):
            if isinstance(layout_type, (np.ndarray, pd.Series)):
                layout_type = list(map(str, layout_type.tolist()))
            else:
                layout_type = [str(layout_type)]
        if not isinstance(unit_type_df, pd.DataFrame):
            unit_type_df = pd.DataFrame(unit_type_df)
        layout_type_col = unit_type_df['Layout Type'] if 'Layout Type' in unit_type_df.columns else pd.Series([])
        if not isinstance(layout_type_col, pd.Series):
            if isinstance(layout_type_col, np.ndarray):
                layout_type_col = pd.Series(layout_type_col)
            else:
                layout_type_col = pd.Series([layout_type_col])
        unit_type_df = unit_type_df[layout_type_col.isin(layout_type)]
    unit_type_col = unit_type_df['Type'] if 'Type' in unit_type_df.columns else pd.Series([])
    if not isinstance(unit_type_col, pd.Series):
        unit_type_col = pd.Series(unit_type_col)
    unit_type_options = sorted(unit_type_col.dropna().unique())
    current_unit_type = st.session_state.get("unit_type", [])
    unit_type = st.multiselect(
        "Unit Type",
        options=unit_type_options,
        default=current_unit_type if current_unit_type and all(u in unit_type_options for u in current_unit_type) else [],
        key="unit_type"
    )

    # --- Time Period Filter ---
    st.subheader("Time Period")
    time_filter_mode = st.selectbox("Time Filter Mode", ["", "Last N Days", "After Date", "From Date to Date"], 
                                   index=["", "Last N Days", "After Date", "From Date to Date"].index(st.session_state.get("time_filter_mode", "Last N Days")), 
                                   key="time_filter_mode", placeholder="")
    
    last_n_days = None
    after_date = None
    date_range = (None, None)
    
    if time_filter_mode == "Last N Days":
        last_n_days = st.number_input("Enter number of days", min_value=1, step=1, 
                                     value=st.session_state.get("last_n_days", 365), key="last_n_days")
    elif time_filter_mode == "After Date":
        after_date = st.date_input("Select start date", 
                                  value=st.session_state.get("after_date", None), key="after_date")
    elif time_filter_mode == "From Date to Date":
        date_range = st.date_input("Select date range", 
                                  value=st.session_state.get("date_range", (None, None)), key="date_range")
    st.markdown("---")

    # Floor Tolerance filter toggle
    enable_floor_tol = st.checkbox("Enable floor tolerance", 
                                  value=st.session_state.get("enable_floor_tol", False), key="enable_floor_tol")
    if enable_floor_tol and property_type == "Apartment":
        floor_tolerance = st.number_input(
            "Floor tolerance (± floors)", min_value=0, step=1, 
            value=st.session_state.get("floor_tolerance", 0), key="floor_tolerance"
        )
    else:
        floor_tolerance = 0
        
    # BUA Tolerance filter toggle
    enable_bua_tol = st.checkbox("Enable BUA tolerance", 
                                value=st.session_state.get("enable_bua_tol", False), key="enable_bua_tol")
    if enable_bua_tol:
        bua_tolerance = st.number_input("BUA tolerance (± sq ft)", min_value=0, step=1, 
                                       value=st.session_state.get("bua_tolerance", 0), key="bua_tolerance")
    else:
        bua_tolerance = 0
        
    # Plot Size Tolerance filter toggle
    enable_plot_tol = st.checkbox("Enable Plot Size tolerance", 
                                 value=st.session_state.get("enable_plot_tol", False), key="enable_plot_tol")
    if enable_plot_tol:
        plot_tolerance = st.number_input("Plot Size tolerance (± sq ft)", min_value=0, step=1, 
                                        value=st.session_state.get("plot_tolerance", 0), key="plot_tolerance")
    else:
        plot_tolerance = 0

    sales_recurrence = st.selectbox("Sales Recurrence", 
                                   options=sales_rec_options, 
                                   index=sales_rec_options.index(st.session_state.get("sales_recurrence", "All")), 
                                   key="sales_recurrence", placeholder="")

# --- Filter Transactions based on Time Period ---
filtered_transactions = all_transactions.copy()

# Apply Time Period Filter
if not filtered_transactions.empty:
    if 'Evidence Date' in filtered_transactions.columns:
        filtered_transactions['Evidence Date'] = pd.to_datetime(filtered_transactions['Evidence Date'], errors='coerce')

        today = datetime.now()

        if time_filter_mode == "Last N Days" and last_n_days:
            date_threshold = today - timedelta(days=last_n_days)
            filtered_transactions = filtered_transactions[filtered_transactions['Evidence Date'] >= date_threshold]

        elif time_filter_mode == "After Date" and after_date:
            filtered_transactions = filtered_transactions[filtered_transactions['Evidence Date'] >= pd.to_datetime(after_date)]

        elif time_filter_mode == "From Date to Date" and isinstance(date_range, (tuple, list)) and len(date_range) == 2 and date_range[0] is not None and date_range[1] is not None:
            start_date = pd.to_datetime(str(date_range[0]))
            end_date = pd.to_datetime(str(date_range[1]))
            filtered_transactions = filtered_transactions[
                (filtered_transactions['Evidence Date'] >= start_date) &
                (filtered_transactions['Evidence Date'] <= end_date)
            ]

# --- Apply sidebar filters ---
    if development:
        filtered_transactions = filtered_transactions[filtered_transactions['All Developments'] == development]
    if community:
        community_col = filtered_transactions['Community/Building']
        if not isinstance(community_col, pd.Series):
            community_col = pd.Series(community_col)
        filtered_transactions = filtered_transactions[community_col.isin(community)]
    if subcommunity:
        subcommunity_col = filtered_transactions['Sub Community / Building']
        if not isinstance(subcommunity_col, pd.Series):
            subcommunity_col = pd.Series(subcommunity_col)
        filtered_transactions = filtered_transactions[subcommunity_col.isin(subcommunity)]
    if property_type:
        filtered_transactions = filtered_transactions[filtered_transactions['Unit Type'] == property_type]
    if bedrooms:
        filtered_transactions = filtered_transactions[filtered_transactions['Beds'].astype(str) == bedrooms]
    if layout_type:
        filtered_transactions = filtered_transactions[filtered_transactions['Layout Type'].isin(layout_type)]  # type: ignore
# --- Unit Type filter ---
unit_type = st.session_state.get("unit_type", [])
if unit_type:
    unit_type_col = filtered_transactions['Unit Type']
    if not isinstance(unit_type_col, pd.Series):
        unit_type_col = pd.Series(unit_type_col)
    filtered_transactions = filtered_transactions[unit_type_col.isin(unit_type)]
    # Floor tolerance filter for apartments if enabled
    if layout_type:
        layout_type_col = filtered_transactions['Layout Type']
        if not isinstance(layout_type_col, pd.Series):
            layout_type_col = pd.Series(layout_type_col)
        filtered_transactions = filtered_transactions[layout_type_col.isin(layout_type)]  # type: ignore
    # BUA tolerance filter if enabled
    # Plot Size tolerance filter if enabled
    if sales_recurrence != "All":
        sales_rec_col = filtered_transactions['Sales Recurrence']
        if not isinstance(sales_rec_col, pd.Series):
            sales_rec_col = pd.Series(sales_rec_col)
        filtered_transactions = filtered_transactions[sales_rec_col == sales_recurrence]

 # --- Load Live Listings Data from Data/Listings ---
listings_dir = os.path.join(os.path.dirname(__file__), 'Data', 'Listings')
with st.spinner("Loading live listings..."):
    all_listings = load_all_listings(listings_dir)
# Pre-convert numeric columns for listings
if not all_listings.empty:
    if 'Beds' in all_listings.columns:
        all_listings['Beds'] = pd.to_numeric(all_listings['Beds'], errors='coerce')
    if 'Floor Level' in all_listings.columns:
        all_listings['Floor Level'] = pd.to_numeric(all_listings['Floor Level'], errors='coerce')
    if 'Price (AED)' in all_listings.columns:
        all_listings['Price (AED)'] = pd.to_numeric(all_listings['Price (AED)'], errors='coerce')
    if 'BUA' in all_listings.columns:
        all_listings['BUA'] = pd.to_numeric(all_listings['BUA'], errors='coerce')
    if 'Days Listed' in all_listings.columns:
        all_listings['Days Listed'] = pd.to_numeric(all_listings['Days Listed'], errors='coerce')

 # --- Load Rent Listings Data from Data/Listings Rent ---
rent_listings_dir = os.path.join(os.path.dirname(__file__), 'Data', 'Listings Rent')
with st.spinner("Loading rent listings..."):
    all_rent_listings = load_all_rent_listings(rent_listings_dir)
# Pre-convert numeric columns for rent listings
if not all_rent_listings.empty:
    if 'Beds' in all_rent_listings.columns:
        all_rent_listings['Beds'] = pd.to_numeric(all_rent_listings['Beds'], errors='coerce')
    if 'Floor Level' in all_rent_listings.columns:
        all_rent_listings['Floor Level'] = pd.to_numeric(all_rent_listings['Floor Level'], errors='coerce')
    if 'Price (AED)' in all_rent_listings.columns:
        all_rent_listings['Price (AED)'] = pd.to_numeric(all_rent_listings['Price (AED)'], errors='coerce')
    if 'BUA' in all_rent_listings.columns:
        all_rent_listings['BUA'] = pd.to_numeric(all_rent_listings['BUA'], errors='coerce')
    if 'Days Listed' in all_rent_listings.columns:
        all_rent_listings['Days Listed'] = pd.to_numeric(all_rent_listings['Days Listed'], errors='coerce')

# --- Apply sidebar filters to listings and rental data ---
def apply_sidebar_filters_to_listings(listings_df, filtered_transactions):
    """Apply sidebar filters to listings data based on filtered transactions."""
    if listings_df.empty:
        return listings_df.copy()
    
    filtered_listings = listings_df.copy()
    
    # Get current filter values from session state
    current_development = st.session_state.get("development", "")
    current_community = st.session_state.get("community", [])
    current_subcommunity = st.session_state.get("subcommunity", "")
    current_property_type = st.session_state.get("property_type", "")
    current_bedrooms = st.session_state.get("bedrooms", "")
    current_layout_type = st.session_state.get("layout_type", [])
    current_unit_type = st.session_state.get("unit_type", [])
    
    # Apply development filter
    if current_development:
        if 'Development' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Development'] == current_development]
        elif 'All Developments' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['All Developments'] == current_development]
    
    # Apply community filter
    if current_community:
        if 'Community' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Community'].isin(current_community)]
        elif 'Community/Building' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Community/Building'].isin(current_community)]
    
    # Apply subcommunity filter
    if current_subcommunity:
        if 'Subcommunity' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Subcommunity'].isin(current_subcommunity)]
        elif 'Sub Community / Building' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Sub Community / Building'].isin(current_subcommunity)]
    
    # Apply property type filter
    if current_property_type:
        if 'Property Type' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Property Type'] == current_property_type]
        elif 'Unit Type' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Unit Type'] == current_property_type]
    
    # Apply bedroom filter
    if current_bedrooms:
        if 'Beds' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Beds'].astype(str) == current_bedrooms]
    
    # Apply layout type filter
    if current_layout_type:
        if 'Layout Type' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Layout Type'].isin(current_layout_type)]
    # Apply unit type filter
    if current_unit_type:
        if 'Unit Type' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Unit Type'].isin(current_unit_type)]
    
    return filtered_listings

def apply_sidebar_filters_to_rentals(rental_df, filtered_transactions):
    """Apply sidebar filters to rental data based on filtered transactions."""
    if rental_df.empty:
        return rental_df.copy()
    
    filtered_rentals = rental_df.copy()
    
    # Get current filter values from session state
    current_development = st.session_state.get("development", "")
    current_community = st.session_state.get("community", [])
    current_subcommunity = st.session_state.get("subcommunity", "")
    current_property_type = st.session_state.get("property_type", "")
    current_bedrooms = st.session_state.get("bedrooms", "")
    current_layout_type = st.session_state.get("layout_type", [])
    current_unit_type = st.session_state.get("unit_type", [])
    
    # Apply development filter
    if current_development:
        if 'All Developments' in filtered_rentals.columns:
            filtered_rentals = filtered_rentals[filtered_rentals['All Developments'] == current_development]
    
    # Apply community filter
    if current_community:
        if 'Community/Building' in filtered_rentals.columns:
            filtered_rentals = filtered_rentals[filtered_rentals['Community/Building'].isin(current_community)]
    
    # Apply subcommunity filter
    if current_subcommunity:
        if 'Sub Community/Building' in filtered_rentals.columns:
            filtered_rentals = filtered_rentals[filtered_rentals['Sub Community/Building'].isin(current_subcommunity)]
    
    # Apply property type filter
    if current_property_type:
        if 'Unit Type' in filtered_rentals.columns:
            filtered_rentals = filtered_rentals[filtered_rentals['Unit Type'] == current_property_type]
    
    # Apply bedroom filter
    if current_bedrooms:
        if 'Beds' in filtered_rentals.columns:
            filtered_rentals = filtered_rentals[filtered_rentals['Beds'].astype(str) == current_bedrooms]
    
    # Apply layout type filter
    if current_layout_type:
        if 'Layout Type' in filtered_rentals.columns:
            filtered_rentals = filtered_rentals[filtered_rentals['Layout Type'].isin(current_layout_type)]
    # Apply unit type filter
    if current_unit_type:
        if 'Unit Type' in filtered_rentals.columns:
            filtered_rentals = filtered_rentals[filtered_rentals['Unit Type'].isin(current_unit_type)]
    
    return filtered_rentals

# Apply filters to create filtered datasets for all tabs
# Cache filtered data to prevent unnecessary recalculations
filter_cache_key = f"filters_{hash(str(sorted(st.session_state.items())))}"
if filter_cache_key not in st.session_state or is_filter_update_in_progress():
    filtered_listings = apply_sidebar_filters_to_listings(all_listings, filtered_transactions)
    filtered_rent_listings = apply_sidebar_filters_to_listings(all_rent_listings, filtered_transactions)
    filtered_rental_data = apply_sidebar_filters_to_rentals(rental_df, filtered_transactions)
    
    # Cache the results
    st.session_state[filter_cache_key] = {
        'filtered_listings': filtered_listings,
        'filtered_rent_listings': filtered_rent_listings,
        'filtered_rental_data': filtered_rental_data
    }
else:
    cached_data = st.session_state[filter_cache_key]
    filtered_listings = cached_data['filtered_listings']
    filtered_rent_listings = cached_data['filtered_rent_listings']
    filtered_rental_data = cached_data['filtered_rental_data']

 # --- Apply sidebar filters except time period (for Search Unit) ---
# Cache no-time filtered data to prevent unnecessary recalculations
no_time_filter_cache_key = f"no_time_filters_{hash(str(sorted(st.session_state.items())))}"
if no_time_filter_cache_key not in st.session_state or is_filter_update_in_progress():
    # Use the same filter logic as the main filtering but without time period
    filtered_transactions_no_time = all_transactions.copy()

    # Apply the same filters as the main filtering logic
    if development:
        filtered_transactions_no_time = filtered_transactions_no_time[filtered_transactions_no_time['All Developments'] == development]
    if community:
        community_col = filtered_transactions_no_time['Community/Building']
        if not isinstance(community_col, pd.Series):
            community_col = pd.Series(community_col)
        filtered_transactions_no_time = filtered_transactions_no_time[community_col.isin(community)]
    if subcommunity:
        subcommunity_col = filtered_transactions_no_time['Sub Community / Building']
        if not isinstance(subcommunity_col, pd.Series):
            subcommunity_col = pd.Series(subcommunity_col)
        filtered_transactions_no_time = filtered_transactions_no_time[subcommunity_col.isin(subcommunity)]
    if property_type:
        filtered_transactions_no_time = filtered_transactions_no_time[filtered_transactions_no_time['Unit Type'] == property_type]
    if bedrooms:
        filtered_transactions_no_time = filtered_transactions_no_time[filtered_transactions_no_time['Beds'].astype(str) == bedrooms]
    if layout_type:
        layout_type_col = filtered_transactions_no_time['Layout Type']
        if not isinstance(layout_type_col, pd.Series):
            layout_type_col = pd.Series(layout_type_col)
        filtered_transactions_no_time = filtered_transactions_no_time[layout_type_col.isin(layout_type)]
    # Get unit_type from session state
    unit_type = st.session_state.get("unit_type", [])
    if unit_type:
        unit_type_col = filtered_transactions_no_time['Unit Type']
        if not isinstance(unit_type_col, pd.Series):
            unit_type_col = pd.Series(unit_type_col)
        filtered_transactions_no_time = filtered_transactions_no_time[unit_type_col.isin(unit_type)]
    if sales_recurrence != "All":
        sales_rec_col = filtered_transactions_no_time['Sales Recurrence']
        if not isinstance(sales_rec_col, pd.Series):
            sales_rec_col = pd.Series(sales_rec_col)
        filtered_transactions_no_time = filtered_transactions_no_time[sales_rec_col == sales_recurrence]

    # Apply same filters to rental data (without time period)
    filtered_rental_data_no_time = apply_sidebar_filters_to_rentals(rental_df, filtered_transactions_no_time)
    
    # Cache the results
    st.session_state[no_time_filter_cache_key] = {
        'filtered_transactions_no_time': filtered_transactions_no_time,
        'filtered_rental_data_no_time': filtered_rental_data_no_time
    }
else:
    cached_data = st.session_state[no_time_filter_cache_key]
    filtered_transactions_no_time = cached_data['filtered_transactions_no_time']
    filtered_rental_data_no_time = cached_data['filtered_rental_data_no_time']

 # --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Sales", "Listings: Sale", "Listings: Rent", "Tracker"])

with tab1:
    st.title("Real Estate Valuation Sales")

    # --- Search Unit Box ---
    # Get all unique units from filtered_transactions_no_time (or another relevant DataFrame)
    unit_col_candidates = [
        'Unit No.', 'Unit Number', 'Unit', 'UnitNo', 'Unit_No', 'Unit_Number'
    ]
    # Ensure DataFrame for filtered_transactions_no_time and filtered_rental_data_no_time
    if not isinstance(filtered_transactions_no_time, pd.DataFrame):
        filtered_transactions_no_time = pd.DataFrame(filtered_transactions_no_time)
    if not isinstance(filtered_rental_data_no_time, pd.DataFrame):
        filtered_rental_data_no_time = pd.DataFrame(filtered_rental_data_no_time)
    unit_col = None
    for col in unit_col_candidates:
        if col in filtered_transactions_no_time.columns:
            unit_col = col
            break
    unit_options = []
    if unit_col:
        unit_col_data = filtered_transactions_no_time[unit_col]
        if not isinstance(unit_col_data, pd.Series):
            unit_col_data = pd.Series(unit_col_data)
        unit_options = sorted(unit_col_data.dropna().astype(str).unique())
    selected_unit = st.selectbox(
        "Search Unit",
        options=[""] + unit_options,
        index=0,
        key="search_unit",
        help="Select a unit to view its details, sales, and rent history."
    )

    if selected_unit:
        # Only show sales and rent transactions, remove status info
        st.markdown(f"#### Sales Transactions for {selected_unit}")
        sales_col = filtered_transactions_no_time[unit_col]
        if not isinstance(sales_col, pd.Series):
            sales_col = pd.Series(sales_col)
        sales_tx = filtered_transactions_no_time[sales_col.astype(str) == selected_unit]
        if not sales_tx.empty:
            st.dataframe(sales_tx)
        else:
            st.info("No sales transactions found for this unit.")
        rent_tx = pd.DataFrame()
        rent_unit_col = None
        for col in unit_col_candidates:
            if col in filtered_rental_data_no_time.columns:
                rent_unit_col = col
                break
        if rent_unit_col:
            rent_col = filtered_rental_data_no_time[rent_unit_col]
            if not isinstance(rent_col, pd.Series):
                rent_col = pd.Series(rent_col)
            rent_tx = filtered_rental_data_no_time[rent_col.astype(str) == selected_unit]
        st.markdown(f"#### Rent Transactions for {selected_unit}")
        if not rent_tx.empty:
            st.dataframe(rent_tx)
        else:
            st.info("No rent transactions found for this unit.")
    else:
            st.info("No rent transactions found for this unit.")
    # --- End Search Unit Box ---

    # Transaction History
    st.subheader("Transaction History")
    if isinstance(filtered_transactions, pd.DataFrame) and filtered_transactions.shape[0] > 0:
        columns_to_hide = ["Select Data Points", "Maid", "Study", "Balcony", "Developer Name", "Source", "Comments", "Source File", "View"]
        visible_columns = [col for col in filtered_transactions.columns if col not in columns_to_hide]
        # Hide Sub Community column if no values present
        if 'Sub Community / Building' in filtered_transactions.columns and pd.Series(filtered_transactions['Sub Community / Building']).dropna().empty:
            if 'Sub Community / Building' in visible_columns:
                visible_columns.remove('Sub Community / Building')
        # Format Evidence Date to YYYY-MM-DD (remove time)
        if 'Evidence Date' in filtered_transactions.columns:
            filtered_transactions = filtered_transactions.copy()
            filtered_transactions['Evidence Date'] = filtered_transactions['Evidence Date'].dt.strftime('%Y-%m-%d')
        # Show count of transactions
        st.markdown(f"**Showing {filtered_transactions.shape[0]} transactions**")
        st.dataframe(filtered_transactions[visible_columns])
    else:
        st.info("No transactions match the current filters.")

    # --- METRICS AND HISTOGRAM ---
    # ... removed metrics and histogram code ...
    # --- END METRICS AND HISTOGRAM ---

with tab2:
    if isinstance(filtered_listings, pd.DataFrame) and filtered_listings.shape[0] > 0:
        st.subheader("Sale Listings (Filtered)")
        
        # Use filtered listings based on sidebar filters
        
        # Add verified filter
        verified_filter = st.radio(
            "Verified listings:",
            ["All listings", "Verified only"],
            index=0,
            horizontal=True,
            key="sale_verified_filter"
        )
        
        # Apply verified filter
        if verified_filter == "Verified only" and 'Verified' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Verified'].str.lower() == 'yes']
        
        # Calculate unique listings count (after verified filter)
        total_listings = filtered_listings.shape[0]
        if isinstance(filtered_listings, pd.DataFrame) and 'DLD Permit Number' in filtered_listings.columns:
            dld_col = filtered_listings['DLD Permit Number']
            if not isinstance(dld_col, pd.Series):
                dld_col = pd.Series(dld_col)
            unique_dlds = dld_col.dropna().nunique()
        else:
            unique_dlds = total_listings
        
        # Show total count
        st.markdown(f"**Total: {total_listings} listings / {unique_dlds} unique listings**")
        
        # Add listing type selector
        listing_type = st.radio(
            "Show listings:",
            ["All listings", "Unique listings", "Duplicate listings"],
            index=0,
            horizontal=True,
            key="sale_listing_type"
        )
        
        # Filter listings based on selection
        if listing_type == "Unique listings":
            # Select unique listings (prioritizing verified and most recent)
            if 'DLD Permit Number' in filtered_listings.columns:
                def select_unique_listings(df):
                    groups = []
                    for dld, group in df.groupby("DLD Permit Number"):
                        if pd.isna(dld) or str(dld).strip() == "":
                            continue
                        # Prioritize verified listings
                        verified_group = group[group["Verified"].str.lower() == "yes"] if "Verified" in group.columns else pd.DataFrame()
                        if not verified_group.empty:
                            if "Listed When" in verified_group.columns:
                                idx = verified_group["Listed When"].idxmax()
                                groups.append(verified_group.loc[[idx]])
                            else:
                                groups.append(verified_group.iloc[[0]])
                        else:
                            if "Listed When" in group.columns:
                                idx = group["Listed When"].idxmax()
                                groups.append(group.loc[[idx]])
                            else:
                                groups.append(group.iloc[[0]])
                    if groups:
                        return pd.concat(groups, ignore_index=True)
                    else:
                        return pd.DataFrame(columns=df.columns)
                
                filtered_listings = select_unique_listings(filtered_listings)
                st.markdown(f"**Showing {filtered_listings.shape[0]} unique listings**")
                
        elif listing_type == "Duplicate listings":
            # Show only listings that have duplicates
            if isinstance(filtered_listings, pd.DataFrame) and 'DLD Permit Number' in filtered_listings.columns:
                dld_col = filtered_listings['DLD Permit Number']
                if not isinstance(dld_col, pd.Series):
                    dld_col = pd.Series(dld_col)
                dld_counts = dld_col.value_counts()
                duplicate_dlds = [dld for dld in dld_counts.index if dld_counts[dld] > 1 and str(dld).strip() != ""]
                filtered_listings = filtered_listings[dld_col.isin(duplicate_dlds)]
                st.markdown(f"**Showing {filtered_listings.shape[0]} duplicate listings**")
        
        # --- Price comparison feature (MOVED HERE, ENSURED NUMERIC) ---
        if isinstance(filtered_listings, pd.DataFrame) and 'Price (AED)' in filtered_listings.columns:
            asking_price = st.number_input(
                "Enter asking price (AED):",
                min_value=0,
                value=st.session_state.get("sale_asking_price", 0),
                key="sale_asking_price"
            )
            price_col = filtered_listings['Price (AED)']
            if not isinstance(price_col, pd.Series):
                if isinstance(price_col, (list, np.ndarray)):
                    price_col = pd.Series(price_col)
                else:
                    price_col = pd.Series([price_col])
            prices_numeric = pd.to_numeric(price_col, errors='coerce')
            if not isinstance(prices_numeric, pd.Series):
                prices_numeric = pd.Series(prices_numeric)
            prices = prices_numeric.dropna()
            above_count = below_count = same_count = 0
            if asking_price > 0 and not prices.empty:
                above_count = (prices > asking_price).sum()
                below_count = (prices < asking_price).sum()
                same_count = (prices == asking_price).sum()
            st.markdown(f"<div style='margin-top: -0.5em; margin-bottom: 1em;'><b>Price Analysis:</b> 🔺 {above_count} above | 🔻 {below_count} below | ⚖️ {same_count} same</div>", unsafe_allow_html=True)
        
        # Hide certain columns but keep them in the DataFrame
        columns_to_hide = ["Reference Number", "URL", "Source File", "Unit No.", "Unit Number", "Listed When", "Listed when", "DLD Permit Number", "Description"]
        if isinstance(filtered_listings, pd.DataFrame):
            visible_columns = [c for c in filtered_listings.columns if c not in columns_to_hide] + ["URL"]
        else:
            visible_columns = []

        # Use AgGrid for clickable selection
        gb = GridOptionsBuilder.from_dataframe(filtered_listings[visible_columns])
        gb.configure_selection('single', use_checkbox=False, rowMultiSelectWithClick=False)
        grid_options = gb.build()

        # Before passing to AgGrid, ensure DataFrame
        data_for_aggrid = filtered_listings[visible_columns]
        if not isinstance(data_for_aggrid, pd.DataFrame):
            data_for_aggrid = pd.DataFrame(data_for_aggrid)
        grid_response = AgGrid(
            data_for_aggrid,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            theme='alpine',
            key="sale_listings_grid"
        )

        # Handle selection from AgGrid (can be list of dicts or DataFrame)
        sel = grid_response['selected_rows']
        selected_url = None
        if isinstance(sel, list):
            if len(sel) > 0:
                selected_url = sel[0].get("URL")
        elif isinstance(sel, pd.DataFrame):
            if sel.shape[0] > 0 and "URL" in sel.columns:
                selected_url = sel.iloc[0]["URL"]
        if selected_url:
            # Foldable Listing Preview
            with st.expander("Listing Preview", expanded=False):
                components.html(
                    f'''
                    <iframe
                        src="{selected_url}"
                        width="100%"
                        height="1600px"
                        style="border:none;"
                        scrolling="yes"
                    ></iframe>
                    ''',
                    height=1600
                )
    else:
        st.info("No sale listings data found.")
with tab3:
    if isinstance(filtered_rent_listings, pd.DataFrame) and filtered_rent_listings.shape[0] > 0:
        st.subheader("Rent Listings (Filtered)")
        
        # Use filtered rent listings based on sidebar filters
        
        # Add verified filter
        verified_filter = st.radio(
            "Verified listings:",
            ["All listings", "Verified only"],
            index=0,
                horizontal=True,
            key="rent_verified_filter"
        )
        
        # Apply verified filter
        if verified_filter == "Verified only" and 'Verified' in filtered_rent_listings.columns:
            filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['Verified'].str.lower() == 'yes']
        
        # Calculate unique listings count (after verified filter)
        total_rent_listings = filtered_rent_listings.shape[0]
        if isinstance(filtered_rent_listings, pd.DataFrame) and 'DLD Permit Number' in filtered_rent_listings.columns:
            dld_col = filtered_rent_listings['DLD Permit Number']
            if not isinstance(dld_col, pd.Series):
                dld_col = pd.Series(dld_col)
            unique_dlds = dld_col.dropna().nunique()
        else:
            unique_dlds = total_rent_listings
        
        # Show total count
        st.markdown(f"**Total: {total_rent_listings} listings / {unique_dlds} unique listings**")
        
        # Add listing type selector
        listing_type = st.radio(
            "Show listings:",
            ["All listings", "Unique listings", "Duplicate listings"],
            index=0,
            horizontal=True,
            key="rent_listing_type"
        )
        
        # Filter listings based on selection
        if listing_type == "Unique listings":
            # Select unique listings (prioritizing verified and most recent)
            if 'DLD Permit Number' in filtered_rent_listings.columns:
                def select_unique_listings(df):
                    groups = []
                    for dld, group in df.groupby("DLD Permit Number"):
                        if pd.isna(dld) or str(dld).strip() == "":
                            continue
                        # Prioritize verified listings
                        verified_group = group[group["Verified"].str.lower() == "yes"] if "Verified" in group.columns else pd.DataFrame()
                        if not verified_group.empty:
                            if "Listed When" in verified_group.columns:
                                idx = verified_group["Listed When"].idxmax()
                                groups.append(verified_group.loc[[idx]])
                            else:
                                groups.append(verified_group.iloc[[0]])
                        else:
                            if "Listed When" in group.columns:
                                idx = group["Listed When"].idxmax()
                                groups.append(group.loc[[idx]])
                            else:
                                groups.append(group.iloc[[0]])
                    if groups:
                        return pd.concat(groups, ignore_index=True)
                    else:
                        return pd.DataFrame(columns=df.columns)
                
                filtered_rent_listings = select_unique_listings(filtered_rent_listings)
                st.markdown(f"**Showing {filtered_rent_listings.shape[0]} unique listings**")
                
        elif listing_type == "Duplicate listings":
            # Show only listings that have duplicates
            if isinstance(filtered_rent_listings, pd.DataFrame) and 'DLD Permit Number' in filtered_rent_listings.columns:
                dld_col = filtered_rent_listings['DLD Permit Number']
                if not isinstance(dld_col, pd.Series):
                    dld_col = pd.Series(dld_col)
                dld_counts = dld_col.value_counts()
                duplicate_dlds = [dld for dld in dld_counts.index if dld_counts[dld] > 1 and str(dld).strip() != ""]
                filtered_rent_listings = filtered_rent_listings[dld_col.isin(duplicate_dlds)]
                st.markdown(f"**Showing {filtered_rent_listings.shape[0]} duplicate listings**")
        
        # --- Price comparison feature (MOVED HERE, ENSURED NUMERIC) ---
        if isinstance(filtered_rent_listings, pd.DataFrame) and 'Price (AED)' in filtered_rent_listings.columns:
            asking_price = st.number_input(
                "Enter asking price (AED):",
                min_value=0,
                value=st.session_state.get("rent_asking_price", 0),
                key="rent_asking_price"
            )
            price_col = filtered_rent_listings['Price (AED)']
            if not isinstance(price_col, pd.Series):
                if isinstance(price_col, (list, np.ndarray)):
                    price_col = pd.Series(price_col)
                else:
                    price_col = pd.Series([price_col])
            prices_numeric = pd.to_numeric(price_col, errors='coerce')
            if not isinstance(prices_numeric, pd.Series):
                prices_numeric = pd.Series(prices_numeric)
            prices = prices_numeric.dropna()
            above_count = below_count = same_count = 0
            if asking_price > 0 and not prices.empty:
                above_count = (prices > asking_price).sum()
                below_count = (prices < asking_price).sum()
                same_count = (prices == asking_price).sum()
            st.markdown(f"<div style='margin-top: -0.5em; margin-bottom: 1em;'><b>Price Analysis:</b> 🔺 {above_count} above | 🔻 {below_count} below | ⚖️ {same_count} same</div>", unsafe_allow_html=True)
        
        # Hide certain columns but keep them in the DataFrame
        columns_to_hide = ["Reference Number", "URL", "Source File", "Unit No.", "Unit Number", "Listed When", "Listed when", "DLD Permit Number", "Description"]
        visible_columns = [c for c in filtered_rent_listings.columns if c not in columns_to_hide] + ["URL"]

        # Use AgGrid for clickable selection
        gb = GridOptionsBuilder.from_dataframe(filtered_rent_listings[visible_columns])
        gb.configure_selection('single', use_checkbox=False, rowMultiSelectWithClick=False)
        grid_options = gb.build()

        # Before passing to AgGrid, ensure DataFrame
        data_for_aggrid = filtered_rent_listings[visible_columns]
        if not isinstance(data_for_aggrid, pd.DataFrame):
            data_for_aggrid = pd.DataFrame(data_for_aggrid)
        grid_response = AgGrid(
            data_for_aggrid,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            theme='alpine',
            key="rent_listings_grid"
        )

        # Handle selection from AgGrid (can be list of dicts or DataFrame)
        sel = grid_response['selected_rows']
        selected_url = None
        if isinstance(sel, list):
            if len(sel) > 0:
                selected_url = sel[0].get("URL")
        elif isinstance(sel, pd.DataFrame):
            if sel.shape[0] > 0 and "URL" in sel.columns:
                selected_url = sel.iloc[0]["URL"]
        if selected_url:
            # Foldable Listing Preview
            with st.expander("Listing Preview", expanded=False):
                components.html(
                    f'''
                    <iframe
                        src="{selected_url}"
                        width="100%"
                        height="1600px"
                        style="border:none;"
                        scrolling="yes"
                    ></iframe>
                    ''',
                    height=1600
                )
    else:
        st.info("No rent listings data found.")
with tab4:
    st.title("Rental Tracker (Filtered)")
    
    # Load all rental data from Data/Rentals directory
    rentals_dir = os.path.join(os.path.dirname(__file__), "Data", "Rentals")
    rental_files = [f for f in os.listdir(rentals_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
    rental_dfs = []
    
    for file in rental_files:
        file_path = os.path.join(rentals_dir, file)
        try:
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.replace('\xa0', '', regex=False).str.strip()
            # Normalize unit numbers for matching
            df['Unit No.'] = df['Unit No.'].astype(str).str.strip().str.upper()
            # Split Evidence Date into Contract Start/End if present
            if 'Evidence Date' in df.columns:
                def split_contract_dates(val):
                    if pd.isnull(val):
                        return pd.NaT, pd.NaT
                    val_str = str(val).replace('\u00A0', '').replace('\xa0', '').strip()
                    if '/' in val_str:
                        parts = val_str.split('/')
                        if len(parts) == 2:
                            start = pd.to_datetime(parts[0].strip(), errors='coerce', dayfirst=True)
                            end = pd.to_datetime(parts[1].strip(), errors='coerce', dayfirst=True)
                            return start, end
                    return pd.NaT, pd.NaT
                
                contract_dates = df['Evidence Date'].apply(split_contract_dates)
                df['Contract Start'] = [dates[0] for dates in contract_dates]
                df['Contract End'] = [dates[1] for dates in contract_dates]
            
            rental_dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to load rental data from {file}: {e}")
    
    if rental_dfs:
        rental_df = pd.concat(rental_dfs, ignore_index=True)
    else:
        rental_df = pd.DataFrame()
    
    # Use filtered rental data based on sidebar filters
    
    # Map Layout Type using layout_map if available
    if 'Unit No.' in filtered_rental_data.columns and layout_map:
        filtered_rental_data['Layout Type'] = filtered_rental_data['Unit No.'].map(layout_map).fillna(filtered_rental_data.get('Layout Type', ''))
        filtered_rental_data['Layout Type'] = filtered_rental_data['Layout Type'].replace('', 'N/A')
    elif 'Layout Type' in filtered_rental_data.columns:
        filtered_rental_data['Layout Type'] = filtered_rental_data['Layout Type'].replace('', 'N/A')
    else:
        filtered_rental_data['Layout Type'] = 'N/A'
    
    # Calculate status for all records
    today = pd.Timestamp.now().normalize()
    if 'Evidence Date' in filtered_rental_data.columns:
        def split_evidence_date(val):
            if pd.isnull(val):
                return pd.NaT, pd.NaT
            val_str = str(val).replace('\u00A0', '').replace('\xa0', '').strip()
            if '/' in val_str:
                parts = val_str.split('/')
                if len(parts) == 2:
                    start = pd.to_datetime(parts[0].strip(), errors='coerce', dayfirst=True)
                    end = pd.to_datetime(parts[1].strip(), errors='coerce', dayfirst=True)
                    return start, end
            return pd.NaT, pd.NaT
        start_end = filtered_rental_data['Evidence Date'].apply(split_evidence_date)
        filtered_rental_data['Start Date_dt'] = [d[0] for d in start_end]
        filtered_rental_data['End Date_dt'] = [d[1] for d in start_end]
    else:
        filtered_rental_data['Start Date_dt'] = pd.NaT
        filtered_rental_data['End Date_dt'] = pd.NaT
    
    # Calculate status using Start/End dates
    days_left = (pd.to_datetime(filtered_rental_data['End Date_dt'], errors='coerce') - today).dt.days
    days_since_end = (today - pd.to_datetime(filtered_rental_data['End Date_dt'], errors='coerce')).dt.days
    start_dates = pd.to_datetime(filtered_rental_data['Start Date_dt'], errors='coerce')
    end_dates = pd.to_datetime(filtered_rental_data['End Date_dt'], errors='coerce')
    status = []
    for i in range(len(filtered_rental_data)):
        start = start_dates.iloc[i]
        end = end_dates.iloc[i]
        left = days_left.iloc[i]
        since_end = days_since_end.iloc[i]
        # New: check for 'Rented Recently (last 90 days)' (active contract started in last 90 days)
        if pd.notnull(start) and pd.notnull(end):
            if start <= today <= end:
                days_since_start = (today - start).days
                if 0 <= days_since_start <= 90:
                    status.append('🟠')  # Rented Recently (last 90 days)
                elif left < 31:
                    status.append('🟣')  # Expiring <30 days
                elif left <= 90:
                    status.append('🟡')  # Expiring Soon
                else:
                    status.append('🔴')  # Rented
            elif 0 < since_end <= 60:
                status.append('🔵')  # Recently Vacant
            else:
                status.append('🟢')  # Available
        else:
            status.append('🟢')  # Available
    filtered_rental_data['Status'] = status
    
    # --- Context-aware unique unit count ---
    # Try to use layout_map_df for unique unit count if possible
    layout_unit_count = None
    rentals_unit_count = None
    layout_note = ""
    # Build filter mask for layout_map_df
    if not layout_map_df.empty:
        layout_filtered = layout_map_df.copy()
        dev = st.session_state.get("development", "")
        comm = st.session_state.get("community", [])
        subcomm = st.session_state.get("subcommunity", [])
        beds = st.session_state.get("bedrooms", "")
        layout_type = st.session_state.get("layout_type", [])
        unit_type = st.session_state.get("unit_type", [])
        # Debug: show unique values in layout file and current filter values
        # Use 'Development' if present and populated, else 'All Developments'
        dev_col = None
        if 'Development' in layout_filtered.columns and bool(layout_filtered['Development'].notna().any()):
            dev_col = 'Development'
        elif 'All Developments' in layout_filtered.columns and bool(layout_filtered['All Developments'].notna().any()):
            dev_col = 'All Developments'
        # Use 'Community' if present and populated, else 'Community/Building'
        comm_col = None
        if 'Community' in layout_filtered.columns and bool(layout_filtered['Community'].notna().any()):
            comm_col = 'Community'
        elif 'Community/Building' in layout_filtered.columns and bool(layout_filtered['Community/Building'].notna().any()):
            comm_col = 'Community/Building'
        # Apply filters
        import pandas as pd
        if dev and dev_col:
            if not isinstance(layout_filtered, pd.DataFrame):
                layout_filtered = pd.DataFrame(layout_filtered)
            if dev_col in layout_filtered.columns and bool(layout_filtered[dev_col].notna().any()):
                layout_filtered = layout_filtered[layout_filtered[dev_col] == dev]
        if not isinstance(comm, list):
            comm = [comm] if comm else []
        if comm and comm_col:
            if not isinstance(layout_filtered, pd.DataFrame):
                layout_filtered = pd.DataFrame(layout_filtered)
            if comm_col in layout_filtered.columns and bool(layout_filtered[comm_col].notna().any()):
                layout_filtered = layout_filtered[layout_filtered[comm_col].isin(comm)]
        if not isinstance(subcomm, list):
            subcomm = [subcomm] if subcomm else []
        if subcomm:
            if not isinstance(layout_filtered, pd.DataFrame):
                layout_filtered = pd.DataFrame(layout_filtered)
            if 'Sub Community / Building' in layout_filtered.columns and bool(layout_filtered['Sub Community / Building'].notna().any()):
                layout_filtered = layout_filtered[layout_filtered['Sub Community / Building'].isin(subcomm)]
        if beds:
            if not isinstance(layout_filtered, pd.DataFrame):
                layout_filtered = pd.DataFrame(layout_filtered)
            if 'Beds' in layout_filtered.columns and bool(layout_filtered['Beds'].notna().any()):
                layout_filtered = layout_filtered[layout_filtered['Beds'].astype(str) == str(beds)]
        if not isinstance(layout_type, list):
            layout_type = [layout_type] if layout_type else []
        if layout_type:
            if not isinstance(layout_filtered, pd.DataFrame):
                layout_filtered = pd.DataFrame(layout_filtered)
            if 'Layout Type' in layout_filtered.columns and bool(layout_filtered['Layout Type'].notna().any()):
                layout_filtered = layout_filtered[layout_filtered['Layout Type'].isin(layout_type)]
        if not isinstance(unit_type, list):
            unit_type = [unit_type] if unit_type else []
        if unit_type:
            if not isinstance(layout_filtered, pd.DataFrame):
                layout_filtered = pd.DataFrame(layout_filtered)
            if 'Type' in layout_filtered.columns and bool(layout_filtered['Type'].notna().any()):
                layout_filtered = layout_filtered[layout_filtered['Type'].isin(unit_type)]
        if 'Unit No.' in layout_filtered.columns:
            unit_no_col = layout_filtered['Unit No.']
            if not isinstance(unit_no_col, pd.Series):
                unit_no_col = pd.Series(unit_no_col)
            layout_unit_count = unit_no_col.nunique()
    # Always get unique units from rentals data as fallback
    if 'Unit No.' in filtered_rental_data.columns:
        rentals_unit_count = filtered_rental_data['Unit No.'].nunique()
    # Decide which to show
    if layout_unit_count is not None and layout_unit_count > 0:
        total_units = layout_unit_count
        layout_note = "(from layout file)"
    else:
        total_units = rentals_unit_count if rentals_unit_count is not None else 0
        layout_note = "(from rentals data)"
    # ...
    # Show data count and unique unit count
    st.info(f"📊 Showing {len(filtered_rental_data)} rental records")
    st.info(f"🏢 Unique Units: {total_units} {layout_note}")
    
    # --- Tracker Metrics Persistence ---
    TRACKER_METRICS_FILE = "tracker_metrics.json"
    def load_last_tracker_metrics():
        try:
            with open(TRACKER_METRICS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def save_tracker_metrics(metrics):
        try:
            with open(TRACKER_METRICS_FILE, "w") as f:
                json.dump(metrics, f)
        except Exception:
            pass

    # ... inside the Tracker tab metrics display ...
    if filtered_rental_data is not None and not filtered_rental_data.empty:
        st.subheader("�� Rental Overview")
        # Status counts from rental data
        status_counts = filtered_rental_data['Status'].value_counts()
        rented_units = status_counts.get('🔴', 0)
        expiring_soon = status_counts.get('🟡', 0)
        expiring_30_days = status_counts.get('🟣', 0)
        recently_vacant = status_counts.get('🔵', 0)
        available_units = status_counts.get('🟢', 0)

        # Load last session's metrics
        last_metrics = load_last_tracker_metrics() or {}
        # Prepare current metrics
        current_metrics = {
            "Rented Units": rented_units,
            "Available Units": available_units,
            "Expiring Soon": expiring_soon,
            "Expiring <30 days": expiring_30_days,
            "Recently Vacant": recently_vacant
        }

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Use context-aware unique unit count for Total Units
            st.metric("Total Units", total_units)
            delta = current_metrics["Available Units"] - last_metrics.get("Available Units", current_metrics["Available Units"])
            st.metric("Available Units", available_units, delta if last_metrics else None)
        with col2:
            delta = current_metrics["Rented Units"] - last_metrics.get("Rented Units", current_metrics["Rented Units"])
            st.metric("Rented Units", rented_units, delta if last_metrics else None)
            delta = current_metrics["Expiring Soon"] - last_metrics.get("Expiring Soon", current_metrics["Expiring Soon"])
            st.metric("Expiring Soon", expiring_soon, delta if last_metrics else None)
        with col3:
            delta = current_metrics["Expiring <30 days"] - last_metrics.get("Expiring <30 days", current_metrics["Expiring <30 days"])
            st.metric("Expiring <30 days", expiring_30_days, delta if last_metrics else None)
            delta = current_metrics["Recently Vacant"] - last_metrics.get("Recently Vacant", current_metrics["Recently Vacant"])
            st.metric("Recently Vacant", recently_vacant, delta if last_metrics else None)
        with col4:
            # Calculate average rent if available
            rent_col_candidates = [
                'Annualised Rental Price(AED)',
                'Annualised Rental Price (AED)',
                'Rent (AED)', 'Annual Rent', 'Rent AED', 'Rent'
            ]
            rent_col = None
            for col in rent_col_candidates:
                if col in filtered_rental_data.columns:
                    rent_col = col
                    break
            if rent_col:
                valid_rents = filtered_rental_data[rent_col].dropna()
                if not valid_rents.empty:
                    avg_rent = valid_rents.mean()
                    st.metric("Avg Annual Rent", f"AED {avg_rent:,.0f}")
                else:
                    st.metric("Avg Annual Rent", "N/A")
            else:
                st.metric("Avg Annual Rent", "N/A")
        # Save current metrics for next session
        save_tracker_metrics(current_metrics)
    
    # Status filter (single-select, with 'All' option and text labels)
    status_label_map = {
        '🔴': 'Rented',
        '🟢': 'Available',
        '🟡': 'Expiring Soon',
        '🟣': 'Expiring <30 days',
        '��': 'Recently Vacant',
        '🟠': 'Rented Recently (last 90 days)'
    }
    status_options = []
    if 'Status' in filtered_rental_data.columns:
        status_options = sorted(filtered_rental_data['Status'].dropna().unique())
    status_select_options = ['All'] + status_options
    def status_format_func(val):
        if val == 'All':
            return 'All'
        return f"{val} {status_label_map.get(val, '')}".strip()
    selected_status = st.selectbox(
        "Status",
        options=status_select_options,
        index=0,
        key="tracker_status_filter",
        format_func=status_format_func
    )
    # Apply the Status filter BEFORE the table
    filtered_table_data = filtered_rental_data.copy()
    if selected_status != 'All' and 'Status' in filtered_table_data.columns:
        filtered_table_data = filtered_table_data[filtered_table_data['Status'] == selected_status]

    # --- Table Display ---
    # --- NEW: Merge layout inventory with rental data ---
    display_data = None
    if not layout_map_df.empty and layout_unit_count is not None and layout_unit_count > 0:
        # Use filtered layout as inventory base
        inventory_df = layout_filtered.copy()
        # Normalize unit numbers for matching
        if 'Unit No.' in inventory_df.columns:
            unit_col = inventory_df['Unit No.']
            if not isinstance(unit_col, pd.Series):
                unit_col = pd.Series(unit_col)
            inventory_df['Unit No.'] = unit_col.astype(str).str.strip().str.upper()
        if 'Unit No.' in filtered_rental_data.columns:
            unit_col = filtered_rental_data['Unit No.']
            if not isinstance(unit_col, pd.Series):
                unit_col = pd.Series(unit_col)
            filtered_rental_data['Unit No.'] = unit_col.astype(str).str.strip().str.upper()
        # Merge: left join inventory with rental data on Unit No.
        merged = pd.merge(
            inventory_df,
            filtered_rental_data,
            on='Unit No.',
            how='left',
            suffixes=('', '_rental')
        )
        # Status: if no rental contract, set to '🟢' (Available)
        merged['Status'] = merged['Status'].fillna('🟢')
        # For contract fields, fill N/A if missing
        for col in ['Start Date_dt', 'End Date_dt']:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), 'N/A')
        # For rent columns, fill N/A if missing
        rent_col_candidates = [
            'Annualised Rental Price(AED)',
            'Annualised Rental Price (AED)',
            'Rent (AED)', 'Annual Rent', 'Rent AED', 'Rent'
        ]
        for col in rent_col_candidates:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), 'N/A')
        display_data = merged
    else:
        # Fallback: use filtered_rental_data as before
        display_data = filtered_rental_data.copy()
        # Deduplicate: keep only the most recent rental record per unit
        if 'Unit No.' in display_data.columns:
            display_data = display_data.sort_values(['Unit No.'], ascending=[True])
            display_data = display_data.drop_duplicates(subset=['Unit No.'], keep='first')
    # --- END NEW ---

    # Select columns to display
    columns_to_show = [
        'Status', 'Unit No.', 'All Developments', 'Community/Building', 'Sub Community/Building',
        'Layout Type', 'Beds', 'Unit Size (sq ft)', 'Start Date_dt', 'End Date_dt'
    ]
    # Add rent amount column if available
    rent_col_candidates = [
        'Annualised Rental Price(AED)',
        'Annualised Rental Price (AED)',
        'Rent (AED)', 'Annual Rent', 'Rent AED', 'Rent'
    ]
    rent_col = None
    for col in rent_col_candidates:
        if col in display_data.columns:
            rent_col = col
            break
    if rent_col:
        columns_to_show.append(rent_col)
    # Filter to only show available columns
    available_columns = [col for col in columns_to_show if col in display_data.columns]
    display_data = display_data[available_columns]
    # Rename columns for better display
    column_mapping = {
        'All Developments': 'Development',
        'Community/Building': 'Community',
        'Sub Community/Building': 'Sub Community',
        'Unit Size (sq ft)': 'BUA (sq ft)',
        'Start Date_dt': 'Contract Start',
        'End Date_dt': 'Contract End',
        'Annualised Rental Price(AED)': 'Annual Rent (AED)',
        'Annualised Rental Price (AED)': 'Annual Rent (AED)',
        'Rent (AED)': 'Rent (AED)',
        'Annual Rent': 'Annual Rent (AED)',
        'Rent AED': 'Rent (AED)',
        'Rent': 'Rent (AED)'
    }
    if not isinstance(display_data, pd.DataFrame):
        display_data = pd.DataFrame(display_data)
    display_data = display_data.rename(columns=column_mapping)
    # Format Contract Start and End columns to show only date in DD/MM/YY format (after renaming)
    if 'Contract Start' in display_data.columns:
        display_data['Contract Start'] = pd.to_datetime(display_data['Contract Start'], errors='coerce').dt.strftime('%d/%m/%y').fillna('N/A')
    if 'Contract End' in display_data.columns:
        display_data['Contract End'] = pd.to_datetime(display_data['Contract End'], errors='coerce').dt.strftime('%d/%m/%y').fillna('N/A')

    # --- Apply Status filter to display_data (fix) ---
    if selected_status != 'All' and 'Status' in display_data.columns:
        display_data = display_data[display_data['Status'] == selected_status]
    # --- END Status filter fix ---

    # Ensure display_data is a DataFrame before passing to AgGrid
    if not isinstance(display_data, pd.DataFrame):
        display_data = pd.DataFrame(display_data)

    # Use AgGrid for interactive table
    gb = GridOptionsBuilder.from_dataframe(display_data)
    gb.configure_selection('single', use_checkbox=False, rowMultiSelectWithClick=False)
    gb.configure_grid_options(domLayout='normal')
    # Configure column properties for better layout
    for col in display_data.columns:
        if 'Rent' in col or 'AED' in col:
            gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], valueFormatter="value.toLocaleString()", flex=1)
        elif col == 'Status':
            gb.configure_column(col, width=100)
        elif col == 'Unit No.':
            gb.configure_column(col, width=100)
        else:
            gb.configure_column(col, flex=1)
    grid_options = gb.build()
    # Display the table
    grid_response = AgGrid(
        display_data,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme='alpine',
        key="rentals_grid",
        height=400
    )
    # ... existing code ...

