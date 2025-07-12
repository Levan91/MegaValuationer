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

def _on_unit_number_change():
    """Callback function for unit number selection changes"""
    # Use state coordination to prevent race conditions
    if not should_process_callback("unit_number", 500):
        return
    
    # Set flag to prevent other callbacks from interfering
    set_filter_update_flag()
    
    # Get the selected unit number
    unit_number = st.session_state.get("unit_number", "")
    if unit_number:
        # Get the unit's data
        unit_data = all_transactions[all_transactions['Unit No.'] == unit_number]
        # Ensure unit_data is a DataFrame
        if not isinstance(unit_data, pd.DataFrame):
            unit_data = pd.DataFrame(unit_data)
        if not unit_data.empty:
            unit_row = unit_data.iloc[0]
            
            # Check if current filters are compatible with the selected unit
            current_development = st.session_state.get("development", "")
            current_community = st.session_state.get("community", [])
            current_subcommunity = st.session_state.get("subcommunity", [])
            
            # Only update filters if they conflict with the unit or are not set
            if not st.session_state.get("lock_location_filters", False):
                # Development: only update if not set or conflicts
                if not current_development:
                    st.session_state["development"] = unit_row['All Developments']
                elif current_development != unit_row['All Developments']:
                    # Conflict detected - update to unit's development
                    st.session_state["development"] = unit_row['All Developments']
                
                # Community: only update if not set or conflicts
                unit_community = unit_row['Community/Building']
                if not current_community:
                    st.session_state["community"] = [unit_community] if pd.notna(unit_community) else []
                elif unit_community not in current_community:
                    # Conflict detected - update to unit's community
                    st.session_state["community"] = [unit_community] if pd.notna(unit_community) else []
                
                # Subcommunity: only update if not set or conflicts
                unit_subcommunity = unit_row['Sub Community / Building']
                if not current_subcommunity:
                    st.session_state["subcommunity"] = unit_subcommunity if pd.notna(unit_subcommunity) else ""
                elif unit_subcommunity not in (current_subcommunity if isinstance(current_subcommunity, list) else [current_subcommunity]):
                    # Conflict detected - update to unit's subcommunity
                    st.session_state["subcommunity"] = unit_subcommunity if pd.notna(unit_subcommunity) else ""
            
            # Property filters: always update to match the unit
            st.session_state["property_type"] = unit_row['Unit Type'] if 'Unit Type' in unit_row else ""
            st.session_state["bedrooms"] = str(unit_row['Beds']) if 'Beds' in unit_row else ""
            st.session_state["bua"] = str(unit_row['Unit Size (sq ft)']) if 'Unit Size (sq ft)' in unit_row else ""
            st.session_state["plot_size"] = str(unit_row['Plot Size (sq ft)']) if 'Plot Size (sq ft)' in unit_row else ""
            
            # Layout Type: update to match the unit
            if 'Layout Type' in unit_row:
                st.session_state["layout_type"] = [unit_row['Layout Type']] if pd.notna(unit_row['Layout Type']) else []
    
    # Clear flag
    clear_filter_update_flag()

def _on_development_change():
    """Callback function for development selection changes"""
    # Use state coordination to prevent race conditions
    if not should_process_callback("development", 300):
        return
    
    # Set flag to prevent other callbacks from interfering
    set_filter_update_flag()
    
    # Only clear unit_number if it no longer matches the selected development
    unit_number = st.session_state.get("unit_number", "")
    development = st.session_state.get("development", "")

    if unit_number and development:
        unit_no_series = all_transactions[
            all_transactions['All Developments'] == development
        ]['Unit No.']
        import pandas as pd
        if not isinstance(unit_no_series, pd.Series):
            unit_no_series = pd.Series(unit_no_series)
        valid_units = unit_no_series.dropna().unique()
        if unit_number not in valid_units:
            st.session_state.pop("unit_number", None)

    # Clear dependent filters
    st.session_state.pop("community", None)
    st.session_state.pop("subcommunity", None)
    
    # Clear flag
    clear_filter_update_flag()

def _on_community_change():
    """Callback function for community selection changes"""
    # Use state coordination to prevent race conditions
    if not should_process_callback("community", 300):
        return
    
    # Set flag to prevent other callbacks from interfering
    set_filter_update_flag()
    
    # Only clear unit_number if it no longer matches the selected communities
    unit_number = st.session_state.get("unit_number", "")
    community = st.session_state.get("community", [])

    if unit_number and community:
        import pandas as pd
        community_col = all_transactions['Community/Building']
        if not isinstance(community_col, pd.Series):
            community_col = pd.Series(community_col)
        valid_units = all_transactions[
            community_col.isin(community)
        ]['Unit No.']
        if not isinstance(valid_units, pd.Series):
            valid_units = pd.Series(valid_units)
        valid_units = valid_units.dropna().unique()
        if unit_number not in valid_units:
            st.session_state.pop("unit_number", None)

    # Clear dependent filters
    st.session_state.pop("subcommunity", None)
    
    # Clear flag
    clear_filter_update_flag()

def _on_subcommunity_change():
    """Callback function for subcommunity selection changes"""
    # Use state coordination to prevent race conditions
    if not should_process_callback("subcommunity", 300):
        return
    
    # Set flag to prevent other callbacks from interfering
    set_filter_update_flag()
    
    # Only clear unit_number if it no longer matches the selected subcommunities
    unit_number = st.session_state.get("unit_number", "")
    subcommunity = st.session_state.get("subcommunity", [])

    if unit_number and subcommunity:
        import pandas as pd
        subcommunity_col = all_transactions['Sub Community / Building']
        if not isinstance(subcommunity_col, pd.Series):
            subcommunity_col = pd.Series(subcommunity_col)
        mask = subcommunity_col.isin(subcommunity)
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask)
        unit_nos = all_transactions[mask]['Unit No.']
        if not isinstance(unit_nos, pd.Series):
            unit_nos = pd.Series(unit_nos)
        unit_nos = unit_nos.reset_index(drop=True)
        valid_units = unit_nos.dropna().unique()
        if unit_number not in valid_units:
            st.session_state.pop("unit_number", None)
    
    # Clear flag
    clear_filter_update_flag()

def _on_bedrooms_change():
    """Callback function for bedrooms selection changes"""
    # Use state coordination to prevent race conditions
    if not should_process_callback("bedrooms", 300):
        return
    
    # Set flag to prevent other callbacks from interfering
    set_filter_update_flag()
    
    # Clear only the unit number selection when bedrooms changes
    st.session_state.pop("unit_number", None)
    
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

def should_process_callback(callback_name, debounce_ms=300):
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
with st.spinner("Loading transaction files..."):
    # Load all, then filter by user selection
    all_transactions = load_all_transactions(transactions_dir)
    all_transactions = all_transactions[all_transactions["Source File"].isin(st.session_state["included_txn_files"])]
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
        st.rerun()

    # Unified refresh button for all data
    if st.button("🔄 Refresh All Data", key="refresh_all"):
        refresh_all_data()
        st.rerun()

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

    # --- Unit Number Filter ---
    st.subheader("Unit Number")
    if not all_transactions.empty:
        # Check if Development is selected - if not, show message and empty options
        current_development = st.session_state.get("development", "")
        if not current_development:
            st.info("Please select a Development to choose a Unit Number.")
            unit_number_options = []
        else:
            unit_df = all_transactions.copy()
            
            # Get current filter values from session state
            current_community = st.session_state.get("community", [])
            current_subcommunity = st.session_state.get("subcommunity", "")
            
            # Apply filters based on current session state values
            unit_df = unit_df[unit_df['All Developments'] == current_development]
            if current_community:
                community_col = unit_df['Community/Building']
                if not isinstance(community_col, pd.Series):
                    community_col = pd.Series(community_col)
                unit_df = unit_df[community_col.isin(current_community)]
            if current_subcommunity:
                subcommunity_col = unit_df['Sub Community / Building']
                if not isinstance(subcommunity_col, pd.Series):
                    subcommunity_col = pd.Series(subcommunity_col)
                unit_df = unit_df[subcommunity_col.isin([current_subcommunity] if isinstance(current_subcommunity, str) else current_subcommunity)]
            
            unit_no_col = unit_df['Unit No.']
            if not isinstance(unit_no_col, pd.Series):
                unit_no_col = pd.Series(unit_no_col)
            unit_number_options = sorted(unit_no_col.dropna().unique())
    else:
        unit_number_options = []
    
    current_unit = st.session_state.get("unit_number", "")
    options = [""] + unit_number_options
    # Safer index calculation
    try:
        default_idx = options.index(current_unit) if current_unit in options else 0
    except (ValueError, IndexError):
        default_idx = 0
    
    unit_number = st.selectbox(
        "Unit Number",
        options=options,
        index=default_idx,
        key="unit_number",
        on_change=_on_unit_number_change,
        placeholder=""
    )



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
    if unit_number:
        unit_row = all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]  # type: ignore

        development = unit_row['All Developments']
        community = [unit_row['Community/Building']] if pd.notna(unit_row['Community/Building']) else []
        subcommunity = unit_row['Sub Community / Building']

        layout_type_val = unit_row['Layout Type'] if 'Layout Type' in unit_row else ''
        # Use context-aware lookup for Beds, BUA, Type
        details = get_unit_details(development, community[0] if community else '', subcommunity, layout_type_val)
        property_type = details.get('Type', unit_row['Unit Type'] if 'Unit Type' in unit_row else '')
        bedrooms = str(details.get('Beds', unit_row['Beds'] if 'Beds' in unit_row else ''))
        bua = details.get('BUA', unit_row['Unit Size (sq ft)'] if 'Unit Size (sq ft)' in unit_row else '')
        floor = str(unit_row['Floor Level']) if pd.notna(unit_row['Floor Level']) else ""
        plot_size = unit_row['Plot Size (sq ft)'] if pd.notna(unit_row['Plot Size (sq ft)']) else ""
    else:
        development = current_development
        community = current_community
        subcommunity = current_subcommunity
        property_type = st.session_state.get("property_type", "")
        bedrooms = st.session_state.get("bedrooms", "")
        bua = st.session_state.get("bua", "")
        plot_size = st.session_state.get("plot_size", "")

    # --- Location selectors (always enabled) ---
    development = st.selectbox(
        "Development",
        options=[""] + dev_options,
        index=([""] + dev_options).index(development) if development in dev_options else 0,
        key="development",
        on_change=_on_development_change,
        placeholder=""
    )
    community = st.multiselect(
        "Community",
        options=com_options,
        default=community if community else [],
        key="community",
        on_change=_on_community_change
    )
    # Filter subcom_options by selected community
    if community:
        community_col = all_transactions['Community/Building']
        if not isinstance(community_col, pd.Series):
            community_col = pd.Series(community_col)
        subcom_df = all_transactions[community_col.isin(community)]
        subcom_col = subcom_df['Sub Community / Building']
        if not isinstance(subcom_col, pd.Series):
            subcom_col = pd.Series(subcom_col)
        subcom_options = sorted(subcom_col.dropna().unique())
    else:
        subcom_options = []
    
    # Handle subcommunity as list or string properly
    if isinstance(subcommunity, list):
        subcommunity_default = subcommunity
    else:
        subcommunity_default = [subcommunity] if subcommunity else []
    
    subcommunity = st.multiselect(
        "Sub community / Building",
        options=subcom_options,
        default=subcommunity_default,
        key="subcommunity",
        on_change=_on_subcommunity_change
    )

    # --- Bedroom Filter ---
    if not all_transactions.empty:
        beds_col = all_transactions['Beds']
        if not isinstance(beds_col, pd.Series):
            beds_col = pd.Series(beds_col)
        bedroom_options = sorted(beds_col.dropna().astype(str).unique())
    else:
        bedroom_options = []
    
    bedrooms = st.selectbox(
        "Bedrooms",
        options=[""] + bedroom_options,
        index=([""] + bedroom_options).index(bedrooms) if bedrooms in bedroom_options else 0,
        key="bedrooms",
        on_change=_on_bedrooms_change
    )

    # --- Layout Type Filter ---
    layout_df_filtered = layout_map_df.copy()
    filtered_unit_nos = set()

    # Get current filter values from session state
    current_development = st.session_state.get("development", "")
    current_community = st.session_state.get("community", [])
    current_subcommunity = st.session_state.get("subcommunity", "")

    if current_community:
        community_col = all_transactions['Community/Building']
        if not isinstance(community_col, pd.Series):
            community_col = pd.Series(community_col)
        unit_nos = all_transactions[community_col.isin(current_community)]['Unit No.']
        if not isinstance(unit_nos, pd.Series):
            unit_nos = pd.Series(unit_nos)
        filtered_unit_nos.update(unit_nos.dropna().unique())
    if current_subcommunity:
        subcommunity_col = all_transactions['Sub Community / Building']
        if not isinstance(subcommunity_col, pd.Series):
            subcommunity_col = pd.Series(subcommunity_col)
        mask = subcommunity_col.isin([current_subcommunity] if isinstance(current_subcommunity, str) else current_subcommunity)
        unit_nos = all_transactions[mask]['Unit No.']
        if not isinstance(unit_nos, pd.Series):
            unit_nos = pd.Series(unit_nos)
        unit_nos = unit_nos.reset_index(drop=True)
        valid_units = unit_nos.dropna().unique()
        if unit_number not in valid_units:
            st.session_state.pop("unit_number", None)
        filtered_unit_nos.update(unit_nos.dropna().unique())

    if filtered_unit_nos:
        unit_no_col = layout_df_filtered['Unit No.']
        if not isinstance(unit_no_col, pd.Series):
            unit_no_col = pd.Series(unit_no_col)
        layout_df_filtered = layout_df_filtered[unit_no_col.isin(list(filtered_unit_nos))]
    else:
        unit_no_col = layout_df_filtered['Unit No.']
        if not isinstance(unit_no_col, pd.Series):
            unit_no_col = pd.Series(unit_no_col)
        layout_df_filtered = layout_df_filtered[unit_no_col.isin([])]  # no match, disable options

    # Get current layout type from session state
    current_layout_type = st.session_state.get("layout_type", [])
    
    # Get layout options from filtered layout data
    layout_type_col = layout_df_filtered['Layout Type']
    if not isinstance(layout_type_col, pd.Series):
        layout_type_col = pd.Series(layout_type_col)
    layout_options = sorted(layout_type_col.dropna().unique())
    
    normalized_unit_number = unit_number.strip().upper() if unit_number else ""
    mapped_layout = layout_map.get(normalized_unit_number, "") if normalized_unit_number else ""

    # If a unit is selected, show only its mapped layout as the option
    if normalized_unit_number and mapped_layout:
        layout_options = [mapped_layout]

    # Use session state for layout type with proper default handling
    layout_type = st.multiselect(
        "Layout Type",
        options=layout_options,
        default=current_layout_type if current_layout_type and all(l in layout_options for l in current_layout_type) else ([mapped_layout] if mapped_layout in layout_options else []),
        key="layout_type"
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
# Floor tolerance filter for apartments if enabled
if property_type == "Apartment" and unit_number and st.session_state.get("enable_floor_tol", False):
    tol = st.session_state.get("floor_tolerance", 0)
    try:
        selected_floor = int(
            all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Floor Level']  # type: ignore
        )
        if tol > 0:
            low = selected_floor - tol
            high = selected_floor + tol
            filtered_transactions = filtered_transactions[
                (filtered_transactions['Floor Level'] >= low) &
                (filtered_transactions['Floor Level'] <= high)
            ]
        else:
            filtered_transactions = filtered_transactions[
                filtered_transactions['Floor Level'] == selected_floor
            ]
    except Exception:
        pass
if layout_type:
    filtered_transactions = filtered_transactions[filtered_transactions['Layout Type'].isin(layout_type)]  # type: ignore
# BUA tolerance filter if enabled
if property_type == "Apartment" and unit_number and st.session_state.get("enable_bua_tol", False):
    tol = st.session_state.get("bua_tolerance", 0)
    if tol > 0:
        try:
            selected_bua = float(
                all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Unit Size (sq ft)']  # type: ignore
            )
            low = selected_bua - tol
            high = selected_bua + tol
            filtered_transactions = filtered_transactions[
                (filtered_transactions['Unit Size (sq ft)'] >= low) &
                (filtered_transactions['Unit Size (sq ft)'] <= high)
            ]
        except Exception:
            pass
# Plot Size tolerance filter if enabled
if unit_number and st.session_state.get("enable_plot_tol", False):
    tol = st.session_state.get("plot_tolerance", 0)
    if tol > 0:
        try:
            selected_plot = float(
                all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Plot Size (sq ft)']  # type: ignore
            )
            low = selected_plot - tol
            high = selected_plot + tol
            filtered_transactions = filtered_transactions[
                (filtered_transactions['Plot Size (sq ft)'] >= low) &
                (filtered_transactions['Plot Size (sq ft)'] <= high)
            ]
        except Exception:
            pass
if sales_recurrence != "All":
    filtered_transactions = filtered_transactions[filtered_transactions['Sales Recurrence'] == sales_recurrence]

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
    if listings_df.empty or filtered_transactions.empty:
        return listings_df.copy()
    
    filtered_listings = listings_df.copy()
    
    # Get unique values from filtered transactions for matching
    if 'All Developments' in filtered_transactions.columns:
        valid_developments = filtered_transactions['All Developments'].dropna().unique()
        if 'Development' in filtered_listings.columns and len(valid_developments) > 0:
            filtered_listings = filtered_listings[filtered_listings['Development'].isin(valid_developments)]
    
    if 'Community/Building' in filtered_transactions.columns:
        valid_communities = filtered_transactions['Community/Building'].dropna().unique()
        if 'Community' in filtered_listings.columns and len(valid_communities) > 0:
            filtered_listings = filtered_listings[filtered_listings['Community'].isin(valid_communities)]
    
    if 'Sub Community / Building' in filtered_transactions.columns:
        valid_subcommunities = filtered_transactions['Sub Community / Building'].dropna().unique()
        if 'Subcommunity' in filtered_listings.columns and len(valid_subcommunities) > 0:
            filtered_listings = filtered_listings[filtered_listings['Subcommunity'].isin(valid_subcommunities)]
    
    if 'Unit Type' in filtered_transactions.columns:
        valid_property_types = filtered_transactions['Unit Type'].dropna().unique()
        if 'Property Type' in filtered_listings.columns and len(valid_property_types) > 0:
            filtered_listings = filtered_listings[filtered_listings['Property Type'].isin(valid_property_types)]
    
    if 'Beds' in filtered_transactions.columns:
        valid_beds = filtered_transactions['Beds'].dropna().unique()
        if 'Beds' in filtered_listings.columns and len(valid_beds) > 0:
            filtered_listings = filtered_listings[filtered_listings['Beds'].isin(valid_beds)]
    
    if 'Layout Type' in filtered_transactions.columns:
        valid_layouts = filtered_transactions['Layout Type'].dropna().unique()
        if 'Layout Type' in filtered_listings.columns and len(valid_layouts) > 0:
            filtered_listings = filtered_listings[filtered_listings['Layout Type'].isin(valid_layouts)]
    
    return filtered_listings

def apply_sidebar_filters_to_rentals(rental_df, filtered_transactions):
    """Apply sidebar filters to rental data based on filtered transactions."""
    if rental_df.empty or filtered_transactions.empty:
        return rental_df.copy()
    
    filtered_rentals = rental_df.copy()
    
    # Get unique values from filtered transactions for matching
    if 'All Developments' in filtered_transactions.columns:
        valid_developments = filtered_transactions['All Developments'].dropna().unique()
        if 'All Developments' in filtered_rentals.columns and len(valid_developments) > 0:
            filtered_rentals = filtered_rentals[filtered_rentals['All Developments'].isin(valid_developments)]
    
    if 'Community/Building' in filtered_transactions.columns:
        valid_communities = filtered_transactions['Community/Building'].dropna().unique()
        if 'Community/Building' in filtered_rentals.columns and len(valid_communities) > 0:
            filtered_rentals = filtered_rentals[filtered_rentals['Community/Building'].isin(valid_communities)]
    
    if 'Sub Community / Building' in filtered_transactions.columns:
        valid_subcommunities = filtered_transactions['Sub Community / Building'].dropna().unique()
        if 'Sub Community/Building' in filtered_rentals.columns and len(valid_subcommunities) > 0:
            filtered_rentals = filtered_rentals[filtered_rentals['Sub Community/Building'].isin(valid_subcommunities)]
    
    if 'Unit Type' in filtered_transactions.columns:
        valid_property_types = filtered_transactions['Unit Type'].dropna().unique()
        if 'Unit Type' in filtered_rentals.columns and len(valid_property_types) > 0:
            filtered_rentals = filtered_rentals[filtered_rentals['Unit Type'].isin(valid_property_types)]
    
    if 'Beds' in filtered_transactions.columns:
        valid_beds = filtered_transactions['Beds'].dropna().unique()
        if 'Beds' in filtered_rentals.columns and len(valid_beds) > 0:
            filtered_rentals = filtered_rentals[filtered_rentals['Beds'].isin(valid_beds)]
    
    if 'Layout Type' in filtered_transactions.columns:
        valid_layouts = filtered_transactions['Layout Type'].dropna().unique()
        if 'Layout Type' in filtered_rentals.columns and len(valid_layouts) > 0:
            filtered_rentals = filtered_rentals[filtered_rentals['Layout Type'].isin(valid_layouts)]
    
    return filtered_rentals

# Apply filters to create filtered datasets for all tabs
filtered_listings = apply_sidebar_filters_to_listings(all_listings, filtered_transactions)
filtered_rent_listings = apply_sidebar_filters_to_listings(all_rent_listings, filtered_transactions)
filtered_rental_data = apply_sidebar_filters_to_rentals(rental_df, filtered_transactions)

 # --- Main Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Listings: Sale", "Listings: Rent", "Trend & Valuation", "Tracker"])

with tab1:
    # Remove unnecessary and error-prone variable deletion
    # st.warning("DASHBOARD TEST MARKER - If you see this, you are in the Dashboard tab!")
    st.markdown("<!-- DASHBOARD TAB START -->")
    st.title("Real Estate Valuation Dashboard")

    # Selected Unit Info box
    st.markdown("### Selected Unit Info")
    info_parts = []
    if unit_number:
        info_parts.append(f"📦 {unit_number}")
    if subcommunity:
        info_parts.append(f"🏙️ {subcommunity}")
    if "Layout Type" in all_transactions.columns:
        layout_val = all_transactions.loc[all_transactions["Unit No."] == unit_number, "Layout Type"].values
        if layout_val.size > 0:
            info_parts.append(f"🗂️ {layout_val[0]}")
    # Add last transaction date for this unit
    last_txn_date = None
    if unit_number and "Evidence Date" in all_transactions.columns:
        unit_txns = all_transactions[all_transactions["Unit No."] == unit_number]
        # Ensure unit_txns is a DataFrame before checking .empty
        import pandas as pd
        if not isinstance(unit_txns, pd.DataFrame):
            unit_txns = pd.DataFrame(unit_txns)
        if not unit_txns.empty:
            # Ensure Evidence Date is datetime
            unit_txns = unit_txns.copy()
            unit_txns["Evidence Date"] = pd.to_datetime(unit_txns["Evidence Date"], errors="coerce")
            last_txn = unit_txns["Evidence Date"].max()
            # Ensure last_txn is a scalar Timestamp before formatting
            if pd.notnull(last_txn) and isinstance(last_txn, pd.Timestamp):
                last_txn_date = last_txn.strftime("%Y-%m-%d")
    if last_txn_date:
        # Make it bold and orange for visibility
        info_parts.append(f"<b style='color:orange;'>🕒 Last Transaction: {last_txn_date}</b>")
    rental_info_html = None
    # Add rental contract start/end dates and amount for this unit in the requested format
    if unit_number and not rental_df.empty:
        normalized_unit_number = str(unit_number).strip().upper()
        unit_rentals = rental_df[rental_df['Unit No.'] == normalized_unit_number]
        import pandas as pd
        if not isinstance(unit_rentals, pd.DataFrame):
            unit_rentals = pd.DataFrame(unit_rentals)
        # Ensure 'Contract Start' is datetime
        if 'Contract Start' in unit_rentals.columns:
            unit_rentals = unit_rentals.copy()
            unit_rentals['Contract Start'] = pd.to_datetime(unit_rentals['Contract Start'], errors='coerce')
        if not unit_rentals.empty and 'Contract Start' in unit_rentals.columns:
            latest_rental = unit_rentals.sort_values(by='Contract Start', ascending=False).iloc[0]
            start = latest_rental['Contract Start']
            end = latest_rental['Contract End']
            # Try to get the rent amount from several possible column names
            rent_col_candidates = [
                'Annualised Rental Price(AED)',  # no space (actual column in your file)
                'Annualised Rental Price (AED)', # with space
                'Rent (AED)', 'Annual Rent', 'Rent AED', 'Rent'
            ]
            amount = None
            for col in rent_col_candidates:
                if col in latest_rental and pd.notnull(latest_rental[col]):
                    amount = latest_rental[col]
                    break
            # Debug: print rent columns and value
            st.write('DEBUG: Rental columns:', list(latest_rental.index))
            st.write('DEBUG: Rental amount:', amount)
            if pd.notnull(start) and pd.notnull(end):
                rental_info_html = f"<b style='color:#007bff;'>Rented: {start.strftime('%d-%b-%Y')} / {end.strftime('%d-%b-%Y')}"
                if amount is not None and pd.notnull(amount):
                    rental_info_html += f" | Amount: AED {amount:,.0f}"
                rental_info_html += "</b>"
    if info_parts:
        st.markdown(" | ".join(info_parts), unsafe_allow_html=True)
    if rental_info_html:
        st.markdown(rental_info_html, unsafe_allow_html=True)

    # Unit details
    if unit_number:
        selected_unit_data = all_transactions[all_transactions['Unit No.'] == unit_number].copy()
        if not isinstance(selected_unit_data, pd.DataFrame):
            selected_unit_data = pd.DataFrame(selected_unit_data)
        if not selected_unit_data.empty:
            unit_info = selected_unit_data.iloc[0]
            st.markdown(f"**Development:** {unit_info['All Developments']}")
            st.markdown(f"**Community:** {unit_info['Community/Building']}")
            st.markdown(f"**Property Type:** {unit_info['Unit Type']}")
            st.markdown(f"**Bedrooms:** {unit_info['Beds']}")
            st.markdown(f"**BUA:** {unit_info['Unit Size (sq ft)']}")
            st.markdown(f"**Plot Size:** {unit_info['Plot Size (sq ft)']}")
            st.markdown(f"**Floor Level:** {unit_info['Floor Level']}")
        else:
            st.markdown(f"**Development:** {development}")
            st.markdown(f"**Community:** {community}")
            st.markdown(f"**Property Type:** {property_type}")
            st.markdown(f"**Bedrooms:** {bedrooms}")
            st.markdown(f"**BUA:** {bua}")
            st.markdown(f"**Plot Size:** {plot_size}")
            st.markdown(f"**Floor Level:**")
    else:
        st.markdown(f"**Development:** {development}")
        st.markdown(f"**Community:** {community}")
        st.markdown(f"**Property Type:** {property_type}")
        st.markdown(f"**Bedrooms:** {bedrooms}")
        st.markdown(f"**BUA:** {bua}")
        st.markdown(f"**Plot Size:** {plot_size}")
        st.markdown(f"**Floor Level:**")

    # Transaction history for selected unit
    if unit_number:
        selected_unit_data = all_transactions[all_transactions['Unit No.'] == unit_number].copy()
        if not isinstance(selected_unit_data, pd.DataFrame):
            selected_unit_data = pd.DataFrame(selected_unit_data)
        if not selected_unit_data.empty:
            st.markdown("---")
            st.markdown("**Transaction History for Selected Unit:**")
            unit_txn_columns_to_hide = ["Unit No.", "Unit Number", "Select Data Points", "Maid", "Study", "Balcony", "Developer Name", "Source", "Comments", "Source File", "View"]
            unit_txn_visible_columns = [col for col in selected_unit_data.columns if col not in unit_txn_columns_to_hide]
            if 'Sub Community / Building' in selected_unit_data.columns and selected_unit_data['Sub Community / Building'].dropna().empty:
                if 'Sub Community / Building' in unit_txn_visible_columns:
                    unit_txn_visible_columns.remove('Sub Community / Building')
            if 'Evidence Date' in selected_unit_data.columns:
                selected_unit_data = selected_unit_data.copy()
                selected_unit_data['Evidence Date'] = selected_unit_data['Evidence Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(selected_unit_data[unit_txn_visible_columns])
        else:
            st.info("No transaction data found for selected unit.")

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

    st.markdown("<!-- DASHBOARD TAB END -->")

with tab2:
    st.markdown("<!-- LISTINGS: SALE TAB START -->")
    if isinstance(filtered_listings, pd.DataFrame) and filtered_listings.shape[0] > 0:
        st.subheader("Sale Listings (Filtered)")
        
        # Use filtered listings based on sidebar filters
        
        # Calculate unique listings count
        total_listings = filtered_listings.shape[0]
        if 'DLD Permit Number' in filtered_listings.columns:
            unique_dlds = filtered_listings['DLD Permit Number'].dropna().nunique()
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
            if 'DLD Permit Number' in filtered_listings.columns:
                dld_counts = filtered_listings['DLD Permit Number'].value_counts()
                duplicate_dlds = [dld for dld in dld_counts.index if dld_counts[dld] > 1 and str(dld).strip() != ""]
                filtered_listings = filtered_listings[filtered_listings['DLD Permit Number'].isin(duplicate_dlds)]
                st.markdown(f"**Showing {filtered_listings.shape[0]} duplicate listings**")
        
        # Hide certain columns but keep them in the DataFrame
        columns_to_hide = ["Reference Number", "URL", "Source File", "Unit No.", "Unit Number", "Listed When", "Listed when", "DLD Permit Number", "Description"]
        visible_columns = [c for c in filtered_listings.columns if c not in columns_to_hide] + ["URL"]

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
    st.markdown("<!-- LISTINGS: SALE TAB END -->")

with tab3:
    st.markdown("<!-- LISTINGS: RENT TAB START -->")
    if isinstance(filtered_rent_listings, pd.DataFrame) and filtered_rent_listings.shape[0] > 0:
        st.subheader("Rent Listings (Filtered)")
        
        # Use filtered rent listings based on sidebar filters
        
        # Calculate unique listings count
        total_rent_listings = filtered_rent_listings.shape[0]
        if 'DLD Permit Number' in filtered_rent_listings.columns:
            unique_dlds = filtered_rent_listings['DLD Permit Number'].dropna().nunique()
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
            if 'DLD Permit Number' in filtered_rent_listings.columns:
                dld_counts = filtered_rent_listings['DLD Permit Number'].value_counts()
                duplicate_dlds = [dld for dld in dld_counts.index if dld_counts[dld] > 1 and str(dld).strip() != ""]
                filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['DLD Permit Number'].isin(duplicate_dlds)]
                st.markdown(f"**Showing {filtered_rent_listings.shape[0]} duplicate listings**")
        
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
    st.markdown("<!-- LISTINGS: RENT TAB END -->")

# All Prophet/chart/forecast code is strictly inside tab4 below
with tab4:
    st.markdown("<!-- TREND & VALUATION TAB START -->")
    # ... (all Prophet/chart/forecast code as in previous block) ...
    # (No Prophet/chart/forecast code outside this block)
    # (No monthly_df, forecast, fig, st.plotly_chart, st.table, st.download_button, Prophet parameter UI, etc. outside this block)
    # (Only function definitions and imports may remain outside)

    # ALL Prophet/chart/forecast code goes here!
    with st.expander("Forecast & Listings Filters", expanded=False):
        prophet_last_n_days = st.number_input(
            "",
            min_value=30, max_value=3650, step=30, value=1095, key="prophet_last_n_days",
            help="Prophet: Last N Days"
        )
        # Live Listings Filters
        verified_only = st.checkbox(
            "Use Only Verified Listings",
            value=True,
            help="Filter to listings marked as verified."
        )
        listings_days = st.number_input(
            "Recent Listings: Last N Days",
            min_value=1, max_value=365, step=1, value=90,
            help="Include only listings not older than N days."
        )
    # Immediately after the expander, before any use of monthly_df
    monthly_df = get_monthly_df(all_transactions, prophet_last_n_days)
    # ... move all Prophet/chart/forecast code here ...

    # Ensure listing_df is defined for this tab
    listing_df = all_listings.copy() if 'all_listings' in locals() else pd.DataFrame()
    # Determine BUA for info and actual values
    unit_bua = None
    if unit_number:
        try:
            unit_bua = float(all_transactions.loc[all_transactions['Unit No.'] == unit_number, 'Unit Size (sq ft)'].iloc[0])
        except:
            unit_bua = None
    else:
        try:
            unit_bua = float(bua)
        except:
            unit_bua = None

    # --- Rental Status Indicator for Trend & Valuation ---
    today = pd.Timestamp.now().normalize()
    rental_circle = "🟢"
    if unit_number and 'Contract Start' in rental_df.columns:
        unit_rentals = rental_df[rental_df['Unit No.'].astype(str) == unit_number]
        if not unit_rentals.empty:
            latest = unit_rentals.sort_values(by='Contract Start', ascending=False).iloc[0]  # type: ignore
            start, end = latest['Contract Start'], latest['Contract End']
            if pd.notnull(start) and pd.notnull(end):
                days_left = (end - today).days
                if start <= today <= end:
                    rental_circle = "🟡" if days_left <= 90 else "🔴"

    # --- Selected Unit Info ---
    info_parts = [rental_circle]
    if unit_number:
        info_parts.append(f"📦 {unit_number}")
    if development:
        info_parts.append(f"🏢 {development}")
    if community:
        # community may be a list
        comm = ", ".join(community) if isinstance(community, list) else community
        info_parts.append(f"🏘️ {comm}")
    if subcommunity:
        info_parts.append(f"🏙️ {subcommunity}")
    if "Layout Type" in all_transactions.columns:
        layout_val = all_transactions.loc[all_transactions["Unit No."] == unit_number, "Layout Type"].values
        if layout_val.size > 0:
            info_parts.append(f"🗂️ {layout_val[0]}")
    # If layout type exists in listing_df and only one unique type, show it
    if 'Layout Type' in listing_df.columns:
        unique_layouts = listing_df['Layout Type'].dropna().unique()
        if len(unique_layouts) == 1:
            info_parts.append(f"📐 Layout: {unique_layouts[0]}")
        elif len(unique_layouts) > 1:
            info_parts.append("📐 Layout: Multiple")
    # Floor Level
    if unit_number and "Floor Level" in all_transactions.columns:
        try:
            fl = all_transactions.loc[all_transactions["Unit No."] == unit_number, "Floor Level"].iloc[0]
            if pd.notna(fl):
                info_parts.append(f"🛗 Floor {int(fl)}")
        except:
            pass
    # BUA (sqft)
    if unit_bua:
        info_parts.append(f"📐 {unit_bua:.0f} sqft")
    # Plot Size (sqft)
    try:
        ps = float(plot_size) if plot_size else float(all_transactions.loc[all_transactions["Unit No."] == unit_number, "Plot Size (sq ft)"].iloc[0])
        info_parts.append(f"🌳 {ps:.0f} sqft")
    except:
        pass
    if info_parts:
        st.markdown(" | ".join(info_parts))
    st.header("Trend & Valuation")

    # Prepare filtered listing DataFrame
    listing_df = all_listings.copy() if 'all_listings' in locals() else pd.DataFrame()
    # Add a switch to filter Prophet/chart listings: all, only unique, only duplicates
    prophet_filter_mode = st.radio(
        "Listings to use for Prophet and chart:",
        ["All listings", "Only unique listings", "Only duplicate listings"],
        index=1,  # Default to unique listings
        horizontal=True,
        key="prophet_listings_filter_mode"
    )
    import pandas as pd
    dld_col = listing_df["DLD Permit Number"] if "DLD Permit Number" in listing_df.columns else pd.Series([])
    if not isinstance(dld_col, pd.Series):
        dld_col = pd.Series(dld_col)
    dld_counts = dld_col.value_counts()
    unique_dlds = [dld for dld in dld_counts.index if dld_counts[dld] == 1 and str(dld).strip() != ""]
    duplicate_dlds = [dld for dld in dld_counts.index if dld_counts[dld] > 1 and str(dld).strip() != ""]
    if prophet_filter_mode == "Only unique listings":
        # Ensure 'Listed When' is datetime for sorting
        if "Listed When" in listing_df.columns:
            listing_df["Listed When"] = pd.to_datetime(listing_df["Listed When"], errors="coerce")
        def select_unique(df):
            groups = []
            for dld, group in df.groupby("DLD Permit Number"):
                if dld is None or str(dld).strip() == "":
                    continue
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
        listing_df = select_unique(listing_df)
    elif prophet_filter_mode == "Only duplicate listings":
        listing_df = listing_df[listing_df["DLD Permit Number"].isin(duplicate_dlds)]
    # (rest of Prophet/chart code unchanged)

    if verified_only and 'Verified' in listing_df.columns:
        listing_df = listing_df[listing_df['Verified'].str.lower() == 'yes']
    if 'Days Listed' in listing_df.columns:
        # Ensure Days Listed is numeric for pd.to_timedelta
        listing_df['Days Listed'] = pd.to_numeric(listing_df['Days Listed'], errors='coerce')
        # Ensure Days Listed is a pandas Series before calling pd.to_timedelta
        days_listed_col = listing_df['Days Listed']
        if not isinstance(days_listed_col, pd.Series):
            days_listed_col = pd.Series(days_listed_col)
        listing_df['Listing Date'] = pd.Timestamp.now() - pd.to_timedelta(days_listed_col, unit='D')
        cutoff_listings = pd.Timestamp.now() - pd.Timedelta(days=listings_days)
        listing_df = listing_df[listing_df['Listing Date'] >= cutoff_listings]

    listing_df = pd.DataFrame(listing_df)
    # Sync listings context to selected unit
    mask = pd.Series(True, index=listing_df.index)
    # Development match
    if 'Development' in listing_df.columns and development:
        mask &= listing_df['Development'] == development
        # Community match
        if 'Community' in listing_df.columns and community:
            comm_list = community if isinstance(community, list) else [community]
            community_col = listing_df['Community']
            if not isinstance(community_col, pd.Series):
                community_col = pd.Series(community_col)
            mask &= community_col.isin(comm_list)
        # Subcommunity match
        subcol = 'Subcommunity' if 'Subcommunity' in listing_df.columns else 'Sub Community / Building'
        if subcol in listing_df.columns and subcommunity:
            subcommunity_col = listing_df[subcol]
            if not isinstance(subcommunity_col, pd.Series):
                subcommunity_col = pd.Series(subcommunity_col)
            if isinstance(subcommunity, (list, tuple)):
                mask &= subcommunity_col.isin(subcommunity)
            else:
                mask &= subcommunity_col == subcommunity
        listing_df = listing_df[mask]
        if layout_type:
            layout_col = listing_df['Layout Type'] if 'Layout Type' in listing_df.columns else pd.Series([])
            if not isinstance(layout_col, pd.Series):
                layout_col = pd.Series(layout_col)
            listing_df = listing_df[layout_col.isin(layout_type)]

    # Compute median listing price per sqft and AED
    listing_df = pd.DataFrame(listing_df)
    if not listing_df.empty and 'Price (AED)' in listing_df.columns and 'BUA' in listing_df.columns:
        listing_df['Price_per_sqft'] = listing_df['Price (AED)'] / listing_df['BUA']
        live_med_sqft = listing_df['Price_per_sqft'].median()
        median_price = live_med_sqft * unit_bua if unit_bua else None
    else:
        median_price = None

    # Display total number of live listings
    total_listings = listing_df.shape[0]
    st.markdown(f"**Total Live Listings:** {total_listings}")
    if len(monthly_df) < 6:
        st.error(f"❌ Insufficient data for forecasting. Need at least 6 months, got {len(monthly_df)}.")
        st.info(" Try increasing the time period or relaxing filters.")
    elif len(monthly_df) < 12:
        st.warning(f"⚠️ Limited data ({len(monthly_df)} months). Using simple median forecast.")
    elif len(monthly_df) < 18:
        # Short-history fallback: flat median of recent data
        # Convert to actual values
        if unit_bua:
            monthly_df['actual'] = monthly_df['y'] * unit_bua
        else:
            monthly_df['actual'] = monthly_df['y']
        # Use last up to 4 months
        n = min(4, len(monthly_df))
        recent = monthly_df['actual'].iloc[-n:]
        median_val = recent.median()
        # Build flat forecast
        last_date = monthly_df['ds'].iloc[-1]
        future = pd.date_range(last_date + pd.offsets.MonthEnd(1), periods=6, freq='M')
        flat_df = pd.DataFrame({'ds': future})
        flat_df['Forecast Value'] = median_val
        flat_df['Lower CI'] = median_val * 0.95
        flat_df['Upper CI'] = median_val * 1.05
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_df['ds'], y=monthly_df['actual'],
            mode='lines+markers', name='Historical',
            line=dict(color='blue'),
            marker=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=flat_df['ds'], y=flat_df['Forecast Value'],
            mode='lines+markers', name='Flat Median Forecast'
        ))
        fig.update_layout(
            title='Flat-Median Forecast (Actual Value)',
            xaxis_title='Month', yaxis_title='AED'
        )
        st.plotly_chart(fig, use_container_width=True)
        # Table
        display_df = flat_df.rename(columns={'ds':'Month'})[['Month','Forecast Value','Lower CI','Upper CI']]
        # Ensure Arrow compatibility for st.table
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
        st.table(display_df)
    else:
        # Full-history Prophet forecast
        # Get or fit cached Prophet model with spinner
        try:
            # Initialize Prophet parameters with default values
            growth = 'linear'
            n_changepoints = 25
            changepoint_range = 0.8
            changepoint_prior_scale = 0.05
            yearly_seasonality = True
            weekly_seasonality = False
            daily_seasonality = False
            seasonality_prior_scale = 10.0
            interval_width = 0.8
            cap = None
            scenario_pct = 0
            
            with st.spinner("Fitting Prophet model..."):
                m = load_prophet_model(
                    growth, n_changepoints, changepoint_range, changepoint_prior_scale,
                    yearly_seasonality, weekly_seasonality, daily_seasonality,
                    seasonality_prior_scale, interval_width, cap, monthly_df
                )
            
            # Create future dataframe
            future = m.make_future_dataframe(periods=6, freq='M')
            
            # Make predictions
            forecast = m.predict(future)
            
            # Convert to actual values
            if unit_bua:
                monthly_df['actual'] = monthly_df['y'] * unit_bua
                forecast['yhat_actual'] = forecast['yhat'] * unit_bua
                forecast['yhat_lower_actual'] = forecast['yhat_lower'] * unit_bua
                forecast['yhat_upper_actual'] = forecast['yhat_upper'] * unit_bua
            else:
                monthly_df['actual'] = monthly_df['y']
                forecast['yhat_actual'] = forecast['yhat']
                forecast['yhat_lower_actual'] = forecast['yhat_lower']
                forecast['yhat_upper_actual'] = forecast['yhat_upper']
                
        except Exception as e:
            st.error(f"❌ Prophet model fitting failed: {str(e)}")
            st.info("💡 Try adjusting Prophet parameters or using a different growth model.")
            st.stop()
        # Plot historical vs forecast with CI band
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_df['ds'], y=monthly_df['actual'],
            mode='lines+markers', name='Historical',
            line=dict(color='blue'),
            marker=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_actual'],
            mode='lines+markers', name='Forecast',
            line=dict(color='lightblue'),
            marker=dict(color='lightblue')
        ))
        # --- Add Linear Regression Trend ---
        from sklearn.linear_model import LinearRegression
        import numpy as np
        # Prepare X for regression: month index as integer
        monthly_df = monthly_df.copy()
        monthly_df['month_idx'] = np.arange(len(monthly_df))
        X_hist = monthly_df['month_idx'].values.reshape(-1, 1)
        y_hist = monthly_df['actual'].values  # Use actual unit values, not price per sqft
        linreg = LinearRegression()
        linreg.fit(X_hist, y_hist)
        # Predict for both historical and future months
        n_future = len(forecast['ds']) - len(monthly_df)
        X_all = np.arange(len(monthly_df) + n_future).reshape(-1, 1)
        y_pred = linreg.predict(X_all)
        # Prepare dates for future months
        last_date = monthly_df['ds'].iloc[-1]
        future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(1), periods=n_future, freq='M')
        all_dates = pd.concat([monthly_df['ds'], pd.Series(future_dates)], ignore_index=True)
        fig.add_trace(go.Scatter(
            x=all_dates,
            y=y_pred,
            mode='lines+markers',  # Keep markers
            name='Linear Trend',
            line=dict(color='orange', dash='dash'),  # Dashed line
            marker=dict(color='orange', symbol='circle', size=8)
        ))
        # --- Highlight Current Month Value ---
        import pandas as pd
        now = pd.Timestamp.now().normalize()
        # Find the current month in monthly_df['ds']
        current_month = monthly_df[monthly_df['ds'].dt.to_period('M') == now.to_period('M')]
        if not current_month.empty:
            fig.add_trace(go.Scatter(
                x=current_month['ds'],
                y=current_month['actual'],
                mode='markers',
                name='Current Month',
                marker=dict(color='red', size=14, symbol='star'),
                showlegend=True
            ))
        # --- Highlighted Transactions by Subcommunity ---
        filtered_transactions = pd.DataFrame(filtered_transactions)
        selected_subcommunity = None
        if unit_number:
            # Get the subcommunity for the selected unit
            try:
                selected_subcommunity = all_transactions.loc[
                    all_transactions['Unit No.'] == unit_number, 'Sub Community / Building'
                ].iloc[0]
            except Exception:
                selected_subcommunity = None
        # Split transactions
        if not filtered_transactions.empty and 'Evidence Date' in filtered_transactions.columns and 'Price (AED)' in filtered_transactions.columns:
            transaction_df = filtered_transactions.copy()
            transaction_df = pd.DataFrame(transaction_df)
            transaction_df['Evidence Date'] = pd.to_datetime(transaction_df['Evidence Date'], errors='coerce')
            transaction_df = transaction_df.dropna(subset=['Evidence Date', 'Price (AED)'])
            if selected_subcommunity:
                txn_sel = transaction_df[transaction_df['Sub Community / Building'] == selected_subcommunity]
                txn_other = transaction_df[transaction_df['Sub Community / Building'] != selected_subcommunity]
                # Selected subcommunity (light purple)
                if not txn_sel.empty:
                    fig.add_trace(go.Scatter(
                        x=txn_sel['Evidence Date'],
                        y=txn_sel['Price (AED)'],
                        mode='markers',
                        marker=dict(symbol='circle', size=8, opacity=0.9, color='#B266FF'),
                        name=f'Transactions ({selected_subcommunity})',
                        text=txn_sel.apply(
                            lambda row: " | ".join(filter(None, [
                                f"Unit: {row['Unit No.']}" if pd.notnull(row.get('Unit No.')) else "",
                                f"Layout: {row['Layout Type']}" if pd.notnull(row.get('Layout Type')) else "",
                                (lambda v: f"BUA: {int(v)} sqft" if isinstance(v, (int, float)) else (f"BUA: {v} sqft" if pd.notnull(v) else ""))(row.get('Unit Size (sq ft)')),
                                (lambda v: f"Plot: {int(v)} sqft" if isinstance(v, (int, float)) else (f"Plot: {v} sqft" if pd.notnull(v) else ""))(row.get('Plot Size (sq ft)'))
                            ])),
                            axis=1
                        ),
                        hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<extra></extra>",
                        hoverlabel=dict(font=dict(color='white'))
                    ))
                # Other subcommunities (blue)
                if not txn_other.empty:
                    fig.add_trace(go.Scatter(
                        x=txn_other['Evidence Date'],
                        y=txn_other['Price (AED)'],
                        mode='markers',
                        marker=dict(symbol='circle', size=6, opacity=0.7, color='blue'),
                        name='Transactions (Other)',
                        text=txn_other.apply(
                            lambda row: " | ".join(filter(None, [
                                f"Unit: {row['Unit No.']}" if pd.notnull(row.get('Unit No.')) else "",
                                f"Layout: {row['Layout Type']}" if pd.notnull(row.get('Layout Type')) else "",
                                (lambda v: f"BUA: {int(v)} sqft" if isinstance(v, (int, float)) else (f"BUA: {v} sqft" if pd.notnull(v) else ""))(row.get('Unit Size (sq ft)')),
                                (lambda v: f"Plot: {int(v)} sqft" if isinstance(v, (int, float)) else (f"Plot: {v} sqft" if pd.notnull(v) else ""))(row.get('Plot Size (sq ft)'))
                            ])),
                            axis=1
                        ),
                        hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<extra></extra>"
                    ))
            else:
                # No selected subcommunity, plot all as blue
                fig.add_trace(go.Scatter(
                    x=transaction_df['Evidence Date'],
                    y=transaction_df['Price (AED)'],
                    mode='markers',
                    marker=dict(symbol='circle', size=6, opacity=0.7, color='blue'),
                    name='Transactions',
                    text=transaction_df.apply(
                        lambda row: " | ".join(filter(None, [
                            f"Unit: {row['Unit No.']}" if pd.notnull(row.get('Unit No.')) else "",
                            f"Layout: {row['Layout Type']}" if pd.notnull(row.get('Layout Type')) else "",
                            (lambda v: f"BUA: {int(v)} sqft" if isinstance(v, (int, float)) else (f"BUA: {v} sqft" if pd.notnull(v) else ""))(row.get('Unit Size (sq ft)')),
                            (lambda v: f"Plot: {int(v)} sqft" if isinstance(v, (int, float)) else (f"Plot: {v} sqft" if pd.notnull(v) else ""))(row.get('Plot Size (sq ft)'))
                        ])),
                        axis=1
                    ),
                    hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<extra></extra>"
                ))
        # --- Highlighted Listings by Subcommunity and Verification ---
        if not listing_df.empty and 'Verified' in listing_df.columns:
            ver_df = listing_df[listing_df['Verified'].str.lower() == 'yes']
            nonver_df = listing_df[listing_df['Verified'].str.lower() != 'yes']
        else:
            ver_df = listing_df.copy()
            nonver_df = listing_df.iloc[0:0]
        # Listings: split by selected subcommunity and verification
        subcol = 'Subcommunity' if 'Subcommunity' in listing_df.columns else 'Sub Community / Building'
        if selected_subcommunity and subcol in listing_df.columns:
            # Verified in selected subcommunity (light purple)
            ver_sel = ver_df[ver_df[subcol] == selected_subcommunity]
            if not isinstance(ver_sel, pd.DataFrame):
                ver_sel = pd.DataFrame(ver_sel)
            # Non-verified in selected subcommunity (orange)
            nonver_sel = nonver_df[nonver_df[subcol] == selected_subcommunity]
            if not isinstance(nonver_sel, pd.DataFrame):
                nonver_sel = pd.DataFrame(nonver_sel)
            # Verified in other subcommunities (green)
            ver_other = ver_df[ver_df[subcol] != selected_subcommunity]
            if not isinstance(ver_other, pd.DataFrame):
                ver_other = pd.DataFrame(ver_other)
            # Non-verified in other subcommunities (red)
            nonver_other = nonver_df[nonver_df[subcol] != selected_subcommunity]
            if not isinstance(nonver_other, pd.DataFrame):
                nonver_other = pd.DataFrame(nonver_other)
            if not ver_sel.empty:
                # Helper to build location string for listings (move to top of listing plotting section)
                fig.add_trace(go.Scatter(
                    x=ver_sel["Listed When"] if "Listed When" in ver_sel.columns else ver_sel.get("Listing Date", ver_sel.index),
                    y=ver_sel['Price (AED)'],
                    mode='markers',
                    marker=dict(symbol='diamond', size=10, opacity=0.95, color='#B266FF'),
                    name=f'Verified Listings ({selected_subcommunity})',
                    customdata=ver_sel["URL"],
                    text=ver_sel.apply(lambda row: \
                        ", ".join(filter(None, [
                            get_location_str(row),
                            f'{int(row["Days Listed"])} days ago' if pd.notnull(row.get("Days Listed")) else '',
                            row["Layout Type"] if pd.notnull(row.get("Layout Type")) else ''
                        ])), axis=1),
                    hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<br><a href='%{customdata}' target='_blank'>View Listing</a><extra></extra>",
                    hoverlabel=dict(font=dict(color='white'))
                ))
            if not nonver_sel.empty:
                fig.add_trace(go.Scatter(
                    x=nonver_sel["Listed When"] if "Listed When" in nonver_sel.columns else nonver_sel.get("Listing Date", nonver_sel.index),
                    y=nonver_sel['Price (AED)'],
                    mode='markers',
                    marker=dict(symbol='diamond', size=10, opacity=0.95, color='orange'),
                    name=f'Non-verified Listings ({selected_subcommunity})',
                    customdata=nonver_sel["URL"],
                    text=nonver_sel.apply(lambda row: \
                        ", ".join(filter(None, [
                            get_location_str(row),
                            f'{int(row["Days Listed"])} days ago' if pd.notnull(row.get("Days Listed")) else '',
                            row["Layout Type"] if pd.notnull(row.get("Layout Type")) else ''
                        ])), axis=1),
                    hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<br><a href='%{customdata}' target='_blank'>View Listing</a><extra></extra>",
                    hoverlabel=dict(font=dict(color='white'))
                ))
            if not ver_other.empty:
                fig.add_trace(go.Scatter(
                    x=ver_other["Listed When"] if "Listed When" in ver_other.columns else ver_other.get("Listing Date", ver_other.index),
                    y=ver_other['Price (AED)'],
                    mode='markers',
                    marker=dict(symbol='diamond', size=8, opacity=0.8, color='green'),
                    name='Verified Listings (Other)',
                    customdata=ver_other["URL"],
                    text=ver_other.apply(lambda row: \
                        ", ".join(filter(None, [
                            get_location_str(row),
                            f'{int(row["Days Listed"])} days ago' if pd.notnull(row.get("Days Listed")) else '',
                            row["Layout Type"] if pd.notnull(row.get("Layout Type")) else ''
                        ])), axis=1),
                    hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<br><a href='%{customdata}' target='_blank'>View Listing</a><extra></extra>",
                    hoverlabel=dict(font=dict(color='white'))
                ))
            if not nonver_other.empty:
                fig.add_trace(go.Scatter(
                    x=nonver_other["Listed When"] if "Listed When" in nonver_other.columns else nonver_other.get("Listing Date", nonver_other.index),
                    y=nonver_other['Price (AED)'],
                    mode='markers',
                    marker=dict(symbol='diamond', size=8, opacity=0.8, color='red'),
                    name='Non-verified Listings (Other)',
                    customdata=nonver_other["URL"],
                    text=nonver_other.apply(lambda row: \
                        ", ".join(filter(None, [
                            get_location_str(row),
                            f'{int(row["Days Listed"])} days ago' if pd.notnull(row.get("Days Listed")) else '',
                            row["Layout Type"] if pd.notnull(row.get("Layout Type")) else ''
                        ])), axis=1),
                    hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<br><a href='%{customdata}' target='_blank'>View Listing</a><extra></extra>",
                    hoverlabel=dict(font=dict(color='white'))
                ))
        else:
            # No selected subcommunity, use default coloring
            if not ver_df.empty:
                fig.add_trace(go.Scatter(
                    x=ver_df["Listed When"] if "Listed When" in ver_df.columns else ver_df.get("Listing Date", ver_df.index),
                    y=ver_df['Price (AED)'],
                    mode='markers',
                    marker=dict(symbol='diamond', size=8, opacity=0.8, color='green'),
                    name='Verified Listings',
                    customdata=ver_df["URL"],
                    text=ver_df.apply(lambda row: \
                        ", ".join(filter(None, [
                            get_location_str(row),
                            f'{int(row["Days Listed"])} days ago' if pd.notnull(row.get("Days Listed")) else '',
                            row["Layout Type"] if pd.notnull(row.get("Layout Type")) else ''
                        ])), axis=1),
                    hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<br><a href='%{customdata}' target='_blank'>View Listing</a><extra></extra>",
                    hoverlabel=dict(font=dict(color='white'))
                ))
            if not nonver_df.empty:
                fig.add_trace(go.Scatter(
                    x=nonver_df["Listed When"] if "Listed When" in nonver_df.columns else nonver_df.get("Listing Date", nonver_df.index),
                    y=nonver_df['Price (AED)'],
                    mode='markers',
                    marker=dict(symbol='diamond', size=8, opacity=0.8, color='red'),
                    name='Non-verified Listings',
                    customdata=nonver_df["URL"],
                    text=nonver_df.apply(lambda row: \
                        ", ".join(filter(None, [
                            get_location_str(row),
                            f'{int(row["Days Listed"])} days ago' if pd.notnull(row.get("Days Listed")) else '',
                            row["Layout Type"] if pd.notnull(row.get("Layout Type")) else ''
                        ])), axis=1),
                    hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<br><a href='%{customdata}' target='_blank'>View Listing</a><extra></extra>",
                    hoverlabel=dict(font=dict(color='white'))
                ))
        fig.update_layout(
            title='Prophet Forecast (Actual Value)',
            xaxis_title='Month', yaxis_title='AED',
            height=800
        )
        # Render interactive Plotly chart with clickable listings
        html_str = fig.to_html(include_plotlyjs='include')
        components.html(f"""
            {html_str}
            <script>
              const gd = document.querySelectorAll('.plotly-graph-div')[0];
              gd.on('plotly_click', function(event) {{
                const url = event.points[0].customdata;
                if (url) window.open(url);
              }});
            </script>
        """, height=800, scrolling=True)
        # Table
        fc = forecast[['ds','yhat_actual','yhat_lower_actual','yhat_upper_actual']].tail(6).copy()
        fc = pd.DataFrame(fc)
        fc['Month'] = fc['ds'].dt.strftime('%B %Y')
        display_df = pd.DataFrame(fc[['Month','yhat_actual','yhat_lower_actual','yhat_upper_actual']])
        display_df = display_df.rename(columns={
            'yhat_actual':'Forecast Value','yhat_lower_actual':'Lower CI','yhat_upper_actual':'Upper CI'
        })
        # Ensure Arrow compatibility for st.table
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].astype(str)
        st.table(display_df)
        # Download data as CSV only
        st.download_button(
            "Download Data as CSV",
            display_df.to_csv(index=False).encode(),
            file_name="forecast.csv"
        )

    # Prophet Settings block (moved here from above)
    with st.expander("Prophet Settings", expanded=False):
        growth = st.selectbox(
            "Growth Model",
            options=["linear", "logistic"],
            index=0,
            help="Choose 'linear' for a straight trend, 'logistic' to enforce capacity limits."
        )
        # If using logistic growth, specify capacity
        cap = None
        if growth == "logistic":
            default_cap = int(monthly_df['y'].max() * 1.1) if not monthly_df.empty else 0
            cap = st.number_input(
                "Capacity (Max price)",
                min_value=0, value=default_cap, step=1000,
                help="Maximum market price for logistic growth."
            )
        n_changepoints = st.slider(
            "Number of Changepoints",
            min_value=0, max_value=50, value=25,
            help="How many potential trend shift points Prophet will consider."
        )
        changepoint_range = st.slider(
            "Changepoint Range",
            min_value=0.5, max_value=1.0, step=0.05, value=0.8,
            help="Fraction of history in which to look for trend shifts."
        )
        changepoint_prior_scale = st.slider(
            "Changepoint Prior Scale",
            min_value=0.001, max_value=1.0, step=0.001, value=0.05,
            help="Trend flexibility: smaller = smoother trend."
        )
        yearly_seasonality = st.checkbox(
            "Yearly Seasonality",
            value=True,
            help="Enable annual cyclic pattern."
        )
        weekly_seasonality = st.checkbox(
            "Weekly Seasonality",
            value=False,
            help="Enable weekly pattern (usually off)."
        )
        daily_seasonality = st.checkbox(
            "Daily Seasonality",
            value=False,
            help="Enable daily pattern (usually off)."
        )
        seasonality_prior_scale = st.slider(
            "Seasonality Prior Scale",
            min_value=0.1, max_value=20.0, step=0.1, value=10.0,
            help="Strength of seasonal patterns: smaller = damped."
        )
        interval_width = st.slider(
            "Interval Width",
            min_value=0.5, max_value=0.99, step=0.01, value=0.8,
            help="Width of the uncertainty interval."
        )
        scenario_pct = st.slider("Scenario adjustment (%)", -20, 20, 0,
                                 help="Apply a manual premium or discount to the forecast.")

    # Add data quality metrics
    st.metric("Data Quality", f"{len(monthly_df)} months", 
              f"{monthly_df['y'].isnull().sum()} missing values")

    # Add confidence level indicator
    confidence = "High" if len(monthly_df) >= 24 else "Medium" if len(monthly_df) >= 12 else "Low"
    st.info(f"📊 Forecast Confidence: {confidence}")

    # --- Prophet Hyperparameter Selection Workflow ---
    # 1. Tune and display both best and current params + MAPEs
    # 2. Let user choose which to use for main forecast

    # Run tuning if button is clicked
    if st.button('Tune Prophet Hyperparameters'):
        with st.spinner('Tuning Prophet hyperparameters...'):
            best_params, best_mape = tune_prophet_hyperparameters(monthly_df)
        if best_params:
            st.success(f'Best Params: {best_params}')
            st.info(f'Best MAPE: {best_mape:.2f}%')
            # Save to session state for later use
            st.session_state['tuned_params'] = best_params
            st.session_state['tuned_mape'] = best_mape
        else:
            st.error('Tuning failed or not enough data.')

    # Helper to robustly convert to bool
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    # Explicitly cast each Prophet argument variable before use
    growth = str(growth)
    n_changepoints = int(n_changepoints)
    changepoint_range = float(changepoint_range)
    changepoint_prior_scale = float(changepoint_prior_scale)
    yearly_seasonality = to_bool(yearly_seasonality)
    weekly_seasonality = to_bool(weekly_seasonality)
    daily_seasonality = to_bool(daily_seasonality)
    seasonality_prior_scale = float(seasonality_prior_scale)
    interval_width = float(interval_width)
    uncertainty_samples = int(1000)
    try:
        # Convert boolean seasonality to proper format expected by Prophet
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
        if yearly_seasonality:
            prophet_params['yearly_seasonality'] = "auto"
        if weekly_seasonality:
            prophet_params['weekly_seasonality'] = "auto"
        if daily_seasonality:
            prophet_params['daily_seasonality'] = "auto"
        
        m_temp = Prophet(**prophet_params)  # type: ignore
        m_temp.fit(monthly_df)
        f = m_temp.predict(monthly_df[['ds']])
        current_mape = mean_absolute_percentage_error(monthly_df['y'], f['yhat']) * 100
    except Exception:
        current_mape = None

    # Show both options and let user choose
    options = ['Current Parameters']
    if 'tuned_params' in st.session_state:
        options.append('Tuned Parameters')
    choice = st.radio('Choose Prophet parameters for main forecast:', options)

    if choice == 'Tuned Parameters' and 'tuned_params' in st.session_state:
        use_params = st.session_state['tuned_params']
        st.info(f'Using tuned parameters: {use_params} (MAPE: {st.session_state["tuned_mape"]:.2f}%)')
    else:
        use_params = dict(
            growth=growth,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            changepoint_prior_scale=changepoint_prior_scale,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples
        )
        st.info(f'Using current parameters (MAPE: {current_mape:.2f}%)')

    # Use use_params for main Prophet forecast below

    st.markdown("<!-- TREND & VALUATION TAB END -->")

with tab5:
    st.markdown("<!-- TRACKER TAB START -->")
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
        if pd.notnull(start) and pd.notnull(end):
            if start <= today <= end:
                if left < 31:
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
    
    # Show data count
    if filtered_rental_data is not None:
        st.info(f"📊 Showing {len(filtered_rental_data)} rental records")
    else:
        st.info("📊 No rental data available")
    
    # Simple metrics display
    if filtered_rental_data is not None and not filtered_rental_data.empty:
        st.subheader("📊 Rental Overview")
        
        # Status counts from rental data
        status_counts = filtered_rental_data['Status'].value_counts()
        rented_units = status_counts.get('🔴', 0)
        expiring_soon = status_counts.get('🟡', 0)
        expiring_30_days = status_counts.get('🟣', 0)
        recently_vacant = status_counts.get('🔵', 0)
        available_units = status_counts.get('🟢', 0)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Units", len(filtered_rental_data))
            st.metric("Available Units", available_units)
        
        with col2:
            st.metric("Rented Units", rented_units)
            st.metric("Expiring Soon", expiring_soon)
        
        with col3:
            st.metric("Expiring <30 days", expiring_30_days)
            st.metric("Recently Vacant", recently_vacant)
        
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
    
    # --- Table Display ---
    if filtered_rental_data is not None and not filtered_rental_data.empty:
        st.subheader("Rental Data Table")
        
        # Prepare data for display
        display_data = filtered_rental_data.copy()
        
        # Deduplicate: keep only the most recent rental record per unit
        if 'Unit No.' in display_data.columns:
            display_data = display_data.sort_values(['Unit No.'], ascending=[True])
            display_data = display_data.drop_duplicates(subset=['Unit No.'], keep='first')
        
        # Select columns to display
        columns_to_show = [
            'Status', 'Unit No.', 'All Developments', 'Community/Building', 'Sub Community/Building',
            'Layout Type', 'Beds', 'Unit Size (sq ft)'
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
            'Annualised Rental Price(AED)': 'Annual Rent (AED)',
            'Annualised Rental Price (AED)': 'Annual Rent (AED)',
            'Rent (AED)': 'Rent (AED)',
            'Annual Rent': 'Annual Rent (AED)',
            'Rent AED': 'Rent (AED)',
            'Rent': 'Rent (AED)'
        }
        
        display_data = display_data.rename(columns=column_mapping)
        
        # Use AgGrid for interactive table
        gb = GridOptionsBuilder.from_dataframe(display_data)
        gb.configure_selection('single', use_checkbox=False, rowMultiSelectWithClick=False)
        gb.configure_grid_options(domLayout='normal')
        
        # Configure column properties
        for col in display_data.columns:
            if 'Rent' in col or 'AED' in col:
                gb.configure_column(col, type=["numericColumn", "numberColumnFilter"], valueFormatter="value.toLocaleString()")
            elif col == 'Status':
                gb.configure_column(col, width=100)
            elif col == 'Unit No.':
                gb.configure_column(col, width=100)
            else:
                gb.configure_column(col, width=150)
        
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
        
        # Handle row selection
        selected_rows = grid_response['selected_rows']
        selected_row = None
        
        # Check if selected_rows is a list with items or a DataFrame with rows
        if isinstance(selected_rows, list) and len(selected_rows) > 0:
            selected_row = selected_rows[0]
        elif isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
            selected_row = selected_rows.iloc[0].to_dict()
        
        if selected_row is not None:
            st.subheader("Selected Unit Details")
            
            # Create a nice display of selected unit info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Unit Number:** {selected_row.get('Unit No.', 'N/A')}")
                st.markdown(f"**Development:** {selected_row.get('Development', 'N/A')}")
                st.markdown(f"**Community:** {selected_row.get('Community', 'N/A')}")
                st.markdown(f"**Sub Community:** {selected_row.get('Sub Community', 'N/A')}")
                st.markdown(f"**Layout Type:** {selected_row.get('Layout Type', 'N/A')}")
            
            with col2:
                st.markdown(f"**Status:** {selected_row.get('Status', 'N/A')}")
                st.markdown(f"**Bedrooms:** {selected_row.get('Beds', 'N/A')}")
                st.markdown(f"**BUA:** {selected_row.get('BUA (sq ft)', 'N/A')}")
                
                # Show rent amount if available
                rent_amount = None
                for col in ['Annual Rent (AED)', 'Rent (AED)']:
                    if col in selected_row and pd.notnull(selected_row[col]):
                        rent_amount = selected_row[col]
                        break
                
                if rent_amount is not None:
                    st.markdown(f"**Annual Rent:** AED {rent_amount:,.0f}")
    
    st.markdown("<!-- TRACKER TAB END -->")

