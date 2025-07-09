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
from scipy.stats import norm
 
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

    # --- Filter Mode ---
    st.subheader("Filter Mode")
    filter_mode = st.radio("Select filter mode", ["Unit Selection", "Manual Selection"], key="filter_mode", horizontal=True)

    # --- Property Filters ---
    # Show property info filters only in Manual Selection mode, wrapped in an expander
    if filter_mode == "Manual Selection":
        st.subheader("Property Info")
        with st.expander("🛠️ Manual Property Info", expanded=True):
            # Get current values from session state
            current_property_type = st.session_state.get("property_type", "")
            current_bedrooms = st.session_state.get("bedrooms", "")
            current_bua = st.session_state.get("bua", "")
            current_plot_size = st.session_state.get("plot_size", "")

            prop_type_col = all_transactions['Unit Type']
            if not isinstance(prop_type_col, pd.Series):
                prop_type_col = pd.Series(prop_type_col)
            prop_type_options = prop_type_col.dropna().unique().tolist()
            
            # Use session state for property type
            property_type = st.selectbox(
                "Property Type",
                options=[""] + sorted(prop_type_options),
                index=([""] + sorted(prop_type_options)).index(current_property_type) if current_property_type in prop_type_options else 0,
                key="property_type"
            )

            beds_col = all_transactions['Beds']
            if not isinstance(beds_col, pd.Series):
                beds_col = pd.Series(beds_col)
            bedrooms_options = beds_col.dropna().astype(str).unique().tolist()
            
            # Use session state for bedrooms
            bedrooms = st.selectbox(
                "Bedrooms",
                options=[""] + sorted(bedrooms_options),
                index=([""] + sorted(bedrooms_options)).index(current_bedrooms) if current_bedrooms in bedrooms_options else 0,
                key="bedrooms"
            )

            # Use session state for BUA and plot size
            bua = st.text_input("BUA (sq ft)", value=current_bua, key="bua")
            plot_size = st.text_input("Plot Size (sq ft)", value=current_plot_size, key="plot_size")
    else:
        # In Unit Selection mode, get values from session state but don't show input fields
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
    
    if filter_mode == "Unit Selection":
        unit_number = st.selectbox(
            "Unit Number",
            options=options,
            index=default_idx,
            key="unit_number",
            on_change=_on_unit_number_change,
            placeholder=""
        )
    else:
        # Manual Selection: Hide/disable unit number
        st.markdown("**Unit Number:** _(Disabled in Manual Selection)_")
        unit_number = ""
        st.session_state["unit_number"] = ""

    # --- Layout Type Filter (by Project, strictly by community/subcommunity) ---
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
    else:
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

        # Get layout options from filtered layout data
        layout_type_col = layout_df_filtered['Layout Type']
        if not isinstance(layout_type_col, pd.Series):
            layout_type_col = pd.Series(layout_type_col)
        layout_options = sorted(layout_type_col.dropna().unique())

    # Debug: Show layout filtering info
    with st.expander("🔧 Debug Layout Filtering", expanded=False):
        st.write(f"**Layout Map DF Shape:** {layout_map_df.shape}")
        st.write(f"**Filtered Layout DF Shape:** {layout_df_filtered.shape}")
        st.write(f"**Filtered Unit Numbers:** {len(filtered_unit_nos)}")
        st.write(f"**Current Unit:** {unit_number}")
        st.write(f"**Mapped Layout:** {mapped_layout}")
        st.write(f"**Layout Options:** {layout_options}")
        st.write(f"**Current Layout Type:** {current_layout_type}")
        st.write(f"**Current Community:** {current_community}")
        st.write(f"**Current Subcommunity:** {current_subcommunity}")
    
    # Use session state for layout type with proper default handling
    layout_type = st.multiselect(
        "Layout Type",
        options=layout_options,
        default=current_layout_type if current_layout_type and all(l in layout_options for l in current_layout_type) else ([mapped_layout] if mapped_layout in layout_options else []),
        key="layout_type"
    )

    # --- Location Filters ---
    st.subheader("Location")
    lock_location_filters = st.checkbox("🔒 Lock Location Filters", value=st.session_state.get("lock_location_filters", False), key="lock_location_filters")
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

    # --- Filter Mode logic for autofill/disable ---
    # In Unit Selection mode, use unit_number to autofill fields. In Manual, allow manual input.
    # Get options for property/beds from data
    if not all_transactions.empty:
        prop_type_col = all_transactions['Unit Type']
        if not isinstance(prop_type_col, pd.Series):
            prop_type_col = pd.Series(prop_type_col)
        prop_type_options = sorted(prop_type_col.dropna().unique())
        beds_col = all_transactions['Beds']
        if not isinstance(beds_col, pd.Series):
            beds_col = pd.Series(beds_col)
        bedrooms_options = sorted(beds_col.dropna().astype(str).unique())
        sales_rec_col = all_transactions['Sales Recurrence']
        if not isinstance(sales_rec_col, pd.Series):
            sales_rec_col = pd.Series(sales_rec_col)
        sales_rec_options = ['All'] + sorted(sales_rec_col.dropna().unique())
    else:
        prop_type_options = bedrooms_options = []
        sales_rec_options = ['All']

    # --- Determine selected values for autofill ---
    if filter_mode == "Unit Selection":
        if unit_number:
            unit_row = all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]  # type: ignore

            if not st.session_state.get("lock_location_filters", False):
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

# --- Apply sidebar filters conditionally based on filter mode ---
if filter_mode == "Unit Selection":
    # Unit Selection: bedrooms, bua, plot_size are autofilled and used from selected unit
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
elif filter_mode == "Manual Selection":
    # Manual Selection: use user-specified bedrooms, bua, plot_size for filtering
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
    if property_type == "Apartment" and st.session_state.get("enable_floor_tol", False):
        tol = st.session_state.get("floor_tolerance", 0)
        try:
            # Use session_state or manual input for floor (not provided, so skip unless added)
            pass  # Not filtering by floor in manual mode unless input provided
        except Exception:
            pass
    if layout_type:
        filtered_transactions = filtered_transactions[filtered_transactions['Layout Type'].isin(layout_type)]  # type: ignore
    # BUA tolerance filter if enabled
    if st.session_state.get("enable_bua_tol", False):
        tol = st.session_state.get("bua_tolerance", 0)
        if tol > 0:
            try:
                if bua:
                    selected_bua = float(bua)
                    low = selected_bua - tol
                    high = selected_bua + tol
                    filtered_transactions = filtered_transactions[
                        (filtered_transactions['Unit Size (sq ft)'] >= low) &
                        (filtered_transactions['Unit Size (sq ft)'] <= high)
                    ]
            except Exception:
                pass
    # Plot Size tolerance filter if enabled
    if st.session_state.get("enable_plot_tol", False):
        tol = st.session_state.get("plot_tolerance", 0)
        if tol > 0:
            try:
                if plot_size:
                    selected_plot = float(plot_size)
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

 # --- Main Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Live Listings", "Rent Listings", "Trend & Valuation", "Rentals"])

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
    # Remove unnecessary and error-prone variable deletion
    st.markdown("<!-- LIVE LISTINGS TAB START -->")
    if isinstance(all_listings, pd.DataFrame) and all_listings.shape[0] > 0:
        st.subheader("All Live Listings")
        
        # --- Filter Mode Switch ---
        filter_mode = st.radio(
            "Filter Mode:",
            ["Use Sidebar Filters", "Use Tab Filters"],
            horizontal=True,
            key="live_listings_filter_mode"
        )
        
        # Apply sidebar filters to live listings (NO date-based or "Days Listed"/"Listed When"/"Listed Date" filtering here)
        filtered_listings = all_listings.copy()
        import pandas as pd
        if not isinstance(filtered_listings, pd.DataFrame):
            filtered_listings = pd.DataFrame(filtered_listings)
        
        if filter_mode == "Use Sidebar Filters":
            # --- Existing Sidebar Filter Logic ---
            def norm(x):
                return str(x).strip().lower() if pd.notnull(x) else ""
            norm_community = [norm(c) for c in community] if community else []
            norm_subcommunity = [norm(s) for s in subcommunity] if subcommunity else []
            norm_layout_type = [norm(l) for l in layout_type] if layout_type else []
            # Always apply subcommunity/layout type filters if selected, regardless of other filters
            if 'Development' in filtered_listings.columns and development:
                filtered_listings = filtered_listings[filtered_listings['Development'].apply(norm) == norm(development)]
            if 'Community' in filtered_listings.columns and community:
                filtered_listings = filtered_listings[filtered_listings['Community'].apply(norm).isin(norm_community)]
            if 'Subcommunity' in filtered_listings.columns and subcommunity:
                filtered_listings = filtered_listings[filtered_listings['Subcommunity'].apply(norm).isin(norm_subcommunity)]
            if 'Layout Type' in filtered_listings.columns and layout_type:
                filtered_listings = filtered_listings[filtered_listings['Layout Type'].apply(norm).isin(norm_layout_type)]
            # Debug output
            # st.write('DEBUG: Listings after Development filter:', len(filtered_listings))
            # st.write('DEBUG: Listings after Community filter:', len(filtered_listings))
            # st.write('DEBUG: Listings after Subcommunity filter:', len(filtered_listings))
            # st.write('DEBUG: Listings after Layout Type filter:', len(filtered_listings))
            # st.write('DEBUG: Unique Subcommunity values:', filtered_listings["Subcommunity"].unique() if 'Subcommunity' in filtered_listings.columns else 'N/A')
            # st.write('DEBUG: Unique Layout Type values:', filtered_listings["Layout Type"].unique() if 'Layout Type' in filtered_listings.columns else 'N/A')
            if not isinstance(filtered_listings, pd.DataFrame):
                filtered_listings = pd.DataFrame(filtered_listings)
            if property_type and 'Type' in filtered_listings.columns:
                filtered_listings = filtered_listings[filtered_listings['Type'] == property_type]
                if not isinstance(filtered_listings, pd.DataFrame):
                    filtered_listings = pd.DataFrame(filtered_listings)
            if bedrooms and 'Beds' in filtered_listings.columns:
                filtered_listings = filtered_listings[filtered_listings['Beds'].astype(str) == bedrooms]
                if not isinstance(filtered_listings, pd.DataFrame):
                    filtered_listings = pd.DataFrame(filtered_listings)
            # Floor tolerance filter for live listings (optional, not specified in instructions)
            if property_type == "Apartment" and unit_number and 'floor_tolerance' in st.session_state and 'Floor Level' in filtered_listings.columns:
                try:
                    selected_floor = int(
                        all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Floor Level']  # type: ignore
                    )
                    tol = st.session_state['floor_tolerance']
                    low = selected_floor - tol
                    high = selected_floor + tol
                    filtered_listings = filtered_listings[
                        (filtered_listings['Floor Level'] >= low) &
                        (filtered_listings['Floor Level'] <= high)
                    ]
                    if not isinstance(filtered_listings, pd.DataFrame):
                        filtered_listings = pd.DataFrame(filtered_listings)
                except Exception:
                    pass
            if layout_type and 'Layout Type' in filtered_listings.columns:
                filtered_listings = filtered_listings[filtered_listings['Layout Type'].isin(layout_type)]  # type: ignore
                if not isinstance(filtered_listings, pd.DataFrame):
                    filtered_listings = pd.DataFrame(filtered_listings)
            if sales_recurrence != "All" and 'Sales Recurrence' in filtered_listings.columns:
                filtered_listings = filtered_listings[filtered_listings['Sales Recurrence'] == sales_recurrence]
                if not isinstance(filtered_listings, pd.DataFrame):
                    filtered_listings = pd.DataFrame(filtered_listings)
        else:
            # --- Tab-Specific Filter Logic ---
            with st.expander("Tab Filters", expanded=False):
                tab_filtered = filtered_listings.copy()
                import pandas as pd
                if not isinstance(tab_filtered, pd.DataFrame):
                    tab_filtered = pd.DataFrame(tab_filtered)

                # 3 columns, 2 rows
                col1, col2, col3 = st.columns(3)

                # Row 1: Development, Community, Bedrooms
                with col1:
                    dev_options = sorted(tab_filtered['Development'].dropna().unique()) if 'Development' in tab_filtered.columns else []
                    tab_development = st.selectbox("Development", options=["All"] + dev_options, key="tab_live_dev")
                with col2:
                    # Community options depend on Development
                    comm_df = tab_filtered.copy()
                    if tab_development != "All" and 'Development' in comm_df.columns:
                        comm_df = comm_df[comm_df['Development'] == tab_development]
                    com_options = sorted(comm_df['Community'].dropna().unique()) if 'Community' in comm_df.columns else []
                    tab_community = st.multiselect("Community", options=com_options, key="tab_live_comm")
                with col3:
                    # Bedrooms options depend on Development and Community
                    beds_df = comm_df.copy()
                    if tab_community and 'Community' in beds_df.columns:
                        beds_df = beds_df[beds_df['Community'].isin(tab_community)]
                    beds_options = sorted(beds_df['Beds'].dropna().unique()) if 'Beds' in beds_df.columns else []
                    tab_bedrooms = st.selectbox("Bedrooms", options=["All"] + [str(x) for x in beds_options], key="tab_live_beds")

                # Row 2: Subcommunity, Property Type, Layout Type
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Subcommunity options depend on Development and Community
                    subcom_df = comm_df.copy()
                    if tab_community and 'Community' in subcom_df.columns:
                        subcom_df = subcom_df[subcom_df['Community'].isin(tab_community)]
                    subcom_options = sorted(subcom_df['Subcommunity'].dropna().unique()) if 'Subcommunity' in subcom_df.columns else []
                    tab_subcommunity = st.multiselect("Subcommunity", options=subcom_options, key="tab_live_subcomm")
                with col2:
                    # Property Type options depend on all previous filters
                    type_df = subcom_df.copy()
                    if tab_subcommunity and 'Subcommunity' in type_df.columns:
                        type_df = type_df[type_df['Subcommunity'].isin(tab_subcommunity)]
                    if tab_bedrooms != "All" and 'Beds' in type_df.columns:
                        type_df = type_df[type_df['Beds'].astype(str) == tab_bedrooms]
                    type_options = sorted(type_df['Type'].dropna().unique()) if 'Type' in type_df.columns else []
                    tab_property_type = st.selectbox("Property Type", options=["All"] + type_options, key="tab_live_type")
                with col3:
                    # Layout Type options depend on all previous filters
                    layout_df = type_df.copy()
                    if tab_property_type != "All" and 'Type' in layout_df.columns:
                        layout_df = layout_df[layout_df['Type'] == tab_property_type]
                    layout_options = sorted(layout_df['Layout Type'].dropna().unique()) if 'Layout Type' in layout_df.columns else []
                    tab_layout_type = st.multiselect("Layout Type", options=layout_options, key="tab_live_layout")

                # Apply all filters to tab_filtered for display
                if tab_development != "All" and 'Development' in tab_filtered.columns:
                    tab_filtered = tab_filtered[tab_filtered['Development'] == tab_development]
                if tab_community and 'Community' in tab_filtered.columns:
                    tab_filtered = tab_filtered[tab_filtered['Community'].isin(tab_community)]
                if tab_subcommunity and 'Subcommunity' in tab_filtered.columns:
                    tab_filtered = tab_filtered[tab_filtered['Subcommunity'].isin(tab_subcommunity)]
                if tab_property_type != "All" and 'Type' in tab_filtered.columns:
                    tab_filtered = tab_filtered[tab_filtered['Type'] == tab_property_type]
                if tab_bedrooms != "All" and 'Beds' in tab_filtered.columns:
                    tab_filtered = tab_filtered[tab_filtered['Beds'].astype(str) == tab_bedrooms]
                if tab_layout_type and 'Layout Type' in tab_filtered.columns:
                    tab_filtered = tab_filtered[tab_filtered['Layout Type'].isin(tab_layout_type)]

                filtered_listings = tab_filtered

        
        # Exclude listings marked as not available
        if 'Availability' in filtered_listings.columns:
            avail_col = filtered_listings['Availability']
            if not isinstance(avail_col, pd.Series):
                avail_col = pd.Series(avail_col)
            filtered_listings = filtered_listings[avail_col.astype(str).str.strip().str.lower() != "not available"]
            if not isinstance(filtered_listings, pd.DataFrame):
                filtered_listings = pd.DataFrame(filtered_listings)

        # Hide certain columns but keep them in the DataFrame (do NOT filter by Days Listed or any date here)
        columns_to_hide = ["Reference Number", "URL", "Source File", "Unit No.", "Unit Number", "Listed When", "Listed when", "DLD Permit Number", "Description"]
        visible_columns = [c for c in filtered_listings.columns if c not in columns_to_hide] + ["URL"]

        # Show count of live listings and unique listings
        total_listings = filtered_listings.shape[0]
        if "DLD Permit Number" in filtered_listings.columns:
            dld_col = filtered_listings["DLD Permit Number"]
            import pandas as pd
            if not isinstance(dld_col, pd.Series):
                dld_col = pd.Series(dld_col)
            unique_dld = dld_col.dropna()
            unique_dld = unique_dld[unique_dld.astype(str).str.strip() != ""]
            total_unique_listings = unique_dld.nunique()
        else:
            total_unique_listings = total_listings
        if total_listings:
            ratio_pct = (total_unique_listings / total_listings) * 100
            ratio_str = f"{ratio_pct:.1f}%"
        else:
            ratio_str = "N/A"
        st.markdown(f"**Showing {total_listings} live listings | {total_unique_listings} unique listings | Ratio: {ratio_str}**")

        # Add a switch to filter listings: all, only unique, only duplicates
        show_filter_mode = st.radio(
            "Show:",
            ["All listings", "Only unique listings", "Only duplicate listings"],
            index=0,
            horizontal=True,
            key="live_listings_show_filter_mode"
        )
        # Compute DLD counts for filtering
        dld_col = filtered_listings["DLD Permit Number"] if "DLD Permit Number" in filtered_listings.columns else pd.Series([])
        if not isinstance(dld_col, pd.Series):
            dld_col = pd.Series(dld_col)
        dld_counts = dld_col.value_counts()
        unique_dlds = [dld for dld in dld_counts.index if dld_counts[dld] == 1 and str(dld).strip() != ""]
        duplicate_dlds = [dld for dld in dld_counts.index if dld_counts[dld] > 1 and str(dld).strip() != ""]
        # Filter listings based on switch (corrected logic)
        if show_filter_mode == "Only unique listings":
            # For each DLD Permit Number (excluding empty/null), show only the first occurrence
            mask = (
                filtered_listings["DLD Permit Number"].notna() &
                (filtered_listings["DLD Permit Number"].astype(str).str.strip() != "")
            )
            filtered_listings = filtered_listings[mask]
            filtered_listings = filtered_listings.drop_duplicates(subset=["DLD Permit Number"], keep="first")
        elif show_filter_mode == "Only duplicate listings":
            filtered_listings = filtered_listings[filtered_listings["DLD Permit Number"].isin(duplicate_dlds)]
        # (rest of code unchanged)

        # --- Asking Price Market Metrics (after filter is applied) ---
        st.markdown("---")
        st.subheader("Asking Price Market Metrics")
        asking_price = st.number_input("Enter your asking price (AED):", min_value=0, step=10000, value=0, format="%d")
        
        # Calculate metrics based on the filtered listings (respecting the "Show" filter)
        if not filtered_listings.empty and "Price (AED)" in filtered_listings.columns:
            # Only consider listings with a valid price
            price_col = "Price (AED)"
            price_series = filtered_listings[price_col]
            if not isinstance(price_series, pd.Series):
                price_series = pd.Series(price_series)
            valid_prices = price_series.notna() & (price_series > 0)
            prices = filtered_listings[valid_prices][price_col].astype(float)
            
            n_total = len(prices)
            n_below = (prices < asking_price).sum() if asking_price > 0 else 0
            n_above = (prices > asking_price).sum() if asking_price > 0 else 0
            n_same = (prices == asking_price).sum() if asking_price > 0 else 0
            pct_below = (n_below / n_total * 100) if n_total and asking_price > 0 else 0
            pct_above = (n_above / n_total * 100) if n_total and asking_price > 0 else 0
            pct_same = (n_same / n_total * 100) if n_total and asking_price > 0 else 0
            # Percentile rank
            percentile = (prices < asking_price).sum() / n_total * 100 if n_total and asking_price > 0 else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Listings Below", n_below, f"{pct_below:.1f}%")
            col2.metric("Listings Above", n_above, f"{pct_above:.1f}%")
            col3.metric("Listings At Same Price", n_same, f"{pct_same:.1f}%")
            col4.metric("Percentile", f"{percentile:.1f}%")
        else:
            st.info("No valid price data available for the selected filter.")
        
        st.markdown("---")

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
            key="live_listings_grid"
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
        st.info("No live listings data found.")
    st.markdown("<!-- LIVE LISTINGS TAB END -->")

with tab3:
    # Remove unnecessary and error-prone variable deletion
    st.markdown("<!-- RENT LISTINGS TAB START -->")
    if isinstance(all_rent_listings, pd.DataFrame) and all_rent_listings.shape[0] > 0:
        st.subheader("All Rent Listings")
        
        # --- Filter Mode Switch ---
        filter_mode = st.radio(
            "Filter Mode:",
            ["Use Sidebar Filters", "Use Tab Filters"],
            horizontal=True,
            key="rent_listings_filter_mode"
        )
        
        # Apply sidebar filters to rent listings (NO date-based or "Days Listed"/"Listed When"/"Listed Date" filtering here)
        filtered_rent_listings = all_rent_listings.copy()
        import pandas as pd
        if not isinstance(filtered_rent_listings, pd.DataFrame):
            filtered_rent_listings = pd.DataFrame(filtered_rent_listings)
        
        if filter_mode == "Use Sidebar Filters":
            # --- Existing Sidebar Filter Logic ---
            def norm(x):
                return str(x).strip().lower() if pd.notnull(x) else ""
            norm_community = [norm(c) for c in community] if community else []
            norm_subcommunity = [norm(s) for s in subcommunity] if subcommunity else []
            norm_layout_type = [norm(l) for l in layout_type] if layout_type else []
            # Always apply subcommunity/layout type filters if selected, regardless of other filters
            if 'Development' in filtered_rent_listings.columns and development:
                filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['Development'].apply(norm) == norm(development)]
            if 'Community' in filtered_rent_listings.columns and community:
                filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['Community'].apply(norm).isin(norm_community)]
            if 'Subcommunity' in filtered_rent_listings.columns and subcommunity:
                filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['Subcommunity'].apply(norm).isin(norm_subcommunity)]
            if 'Layout Type' in filtered_rent_listings.columns and layout_type:
                filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['Layout Type'].apply(norm).isin(norm_layout_type)]
            if not isinstance(filtered_rent_listings, pd.DataFrame):
                filtered_rent_listings = pd.DataFrame(filtered_rent_listings)
            if property_type and 'Type' in filtered_rent_listings.columns:
                filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['Type'] == property_type]
                if not isinstance(filtered_rent_listings, pd.DataFrame):
                    filtered_rent_listings = pd.DataFrame(filtered_rent_listings)
            if bedrooms and 'Beds' in filtered_rent_listings.columns:
                filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['Beds'].astype(str) == bedrooms]
                if not isinstance(filtered_rent_listings, pd.DataFrame):
                    filtered_rent_listings = pd.DataFrame(filtered_rent_listings)
            # Floor tolerance filter for rent listings (optional, not specified in instructions)
            if property_type == "Apartment" and unit_number and 'floor_tolerance' in st.session_state and 'Floor Level' in filtered_rent_listings.columns:
                try:
                    selected_floor = int(
                        all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Floor Level']  # type: ignore
                    )
                    tol = st.session_state['floor_tolerance']
                    low = selected_floor - tol
                    high = selected_floor + tol
                    filtered_rent_listings = filtered_rent_listings[
                        (filtered_rent_listings['Floor Level'] >= low) &
                        (filtered_rent_listings['Floor Level'] <= high)
                    ]
                    if not isinstance(filtered_rent_listings, pd.DataFrame):
                        filtered_rent_listings = pd.DataFrame(filtered_rent_listings)
                except Exception:
                    pass
            if layout_type and 'Layout Type' in filtered_rent_listings.columns:
                filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['Layout Type'].isin(layout_type)]  # type: ignore
                if not isinstance(filtered_rent_listings, pd.DataFrame):
                    filtered_rent_listings = pd.DataFrame(filtered_rent_listings)
            if sales_recurrence != "All" and 'Sales Recurrence' in filtered_rent_listings.columns:
                filtered_rent_listings = filtered_rent_listings[filtered_rent_listings['Sales Recurrence'] == sales_recurrence]
                if not isinstance(filtered_rent_listings, pd.DataFrame):
                    filtered_rent_listings = pd.DataFrame(filtered_rent_listings)
        else:
            # --- Tab-Specific Filter Logic ---
            st.markdown("### Tab Filters")

            # Start with all rent listings for tab filters
            tab_filtered = filtered_rent_listings.copy()
            import pandas as pd
            if not isinstance(tab_filtered, pd.DataFrame):
                tab_filtered = pd.DataFrame(tab_filtered)

            # Get unique values for each filter based on current selections
            dev_options = sorted(tab_filtered['Development'].dropna().unique()) if 'Development' in tab_filtered.columns else []
            tab_development = st.selectbox("Development", options=["All"] + dev_options, key="tab_rent_dev")
            if tab_development != "All" and 'Development' in tab_filtered.columns:
                tab_filtered = tab_filtered[tab_filtered['Development'] == tab_development]

            com_options = sorted(tab_filtered['Community'].dropna().unique()) if 'Community' in tab_filtered.columns else []
            tab_community = st.multiselect("Community", options=com_options, key="tab_rent_comm")
            if tab_community and 'Community' in tab_filtered.columns:
                tab_filtered = tab_filtered[tab_filtered['Community'].isin(tab_community)]

            subcom_options = sorted(tab_filtered['Subcommunity'].dropna().unique()) if 'Subcommunity' in tab_filtered.columns else []
            tab_subcommunity = st.multiselect("Subcommunity", options=subcom_options, key="tab_rent_subcomm")
            if tab_subcommunity and 'Subcommunity' in tab_filtered.columns:
                tab_filtered = tab_filtered[tab_filtered['Subcommunity'].isin(tab_subcommunity)]

            type_options = sorted(tab_filtered['Type'].dropna().unique()) if 'Type' in tab_filtered.columns else []
            tab_property_type = st.selectbox("Property Type", options=["All"] + type_options, key="tab_rent_type")
            if tab_property_type != "All" and 'Type' in tab_filtered.columns:
                tab_filtered = tab_filtered[tab_filtered['Type'] == tab_property_type]

            beds_options = sorted(tab_filtered['Beds'].dropna().unique()) if 'Beds' in tab_filtered.columns else []
            tab_bedrooms = st.selectbox("Bedrooms", options=["All"] + [str(x) for x in beds_options], key="tab_rent_beds")
            if tab_bedrooms != "All" and 'Beds' in tab_filtered.columns:
                tab_filtered = tab_filtered[tab_filtered['Beds'].astype(str) == tab_bedrooms]

            layout_options = sorted(tab_filtered['Layout Type'].dropna().unique()) if 'Layout Type' in tab_filtered.columns else []
            tab_layout_type = st.multiselect("Layout Type", options=layout_options, key="tab_rent_layout")
            if tab_layout_type and 'Layout Type' in tab_filtered.columns:
                tab_filtered = tab_filtered[tab_filtered['Layout Type'].isin(tab_layout_type)]

            # After all filters, assign to filtered_rent_listings for display
            filtered_rent_listings = tab_filtered

        
        # Exclude listings marked as not available
        if 'Availability' in filtered_rent_listings.columns:
            avail_col = filtered_rent_listings['Availability']
            if not isinstance(avail_col, pd.Series):
                avail_col = pd.Series(avail_col)
            filtered_rent_listings = filtered_rent_listings[avail_col.astype(str).str.strip().str.lower() != "not available"]
            if not isinstance(filtered_rent_listings, pd.DataFrame):
                filtered_rent_listings = pd.DataFrame(filtered_rent_listings)

        # Hide certain columns but keep them in the DataFrame (do NOT filter by Days Listed or any date here)
        columns_to_hide = ["Reference Number", "URL", "Source File", "Unit No.", "Unit Number", "Listed When", "Listed when", "DLD Permit Number", "Description"]
        visible_columns = [c for c in filtered_rent_listings.columns if c not in columns_to_hide] + ["URL"]

        # Show count of rent listings and unique listings
        total_rent_listings = filtered_rent_listings.shape[0]
        if "DLD Permit Number" in filtered_rent_listings.columns:
            dld_col = filtered_rent_listings["DLD Permit Number"]
            import pandas as pd
            if not isinstance(dld_col, pd.Series):
                dld_col = pd.Series(dld_col)
            unique_dld = dld_col.dropna()
            unique_dld = unique_dld[unique_dld.astype(str).str.strip() != ""]
            total_unique_rent_listings = unique_dld.nunique()
        else:
            total_unique_rent_listings = total_rent_listings
        if total_rent_listings:
            ratio_pct = (total_unique_rent_listings / total_rent_listings) * 100
            ratio_str = f"{ratio_pct:.1f}%"
        else:
            ratio_str = "N/A"
        st.markdown(f"**Showing {total_rent_listings} rent listings | {total_unique_rent_listings} unique listings | Ratio: {ratio_str}**")

        # Add a switch to filter listings: all, only unique, only duplicates
        show_filter_mode = st.radio(
            "Show:",
            ["All listings", "Only unique listings", "Only duplicate listings"],
            index=0,
            horizontal=True,
            key="rent_listings_show_filter_mode"
        )
        # Compute DLD counts for filtering
        dld_col = filtered_rent_listings["DLD Permit Number"] if "DLD Permit Number" in filtered_rent_listings.columns else pd.Series([])
        if not isinstance(dld_col, pd.Series):
            dld_col = pd.Series(dld_col)
        dld_counts = dld_col.value_counts()
        unique_dlds = [dld for dld in dld_counts.index if dld_counts[dld] == 1 and str(dld).strip() != ""]
        duplicate_dlds = [dld for dld in dld_counts.index if dld_counts[dld] > 1 and str(dld).strip() != ""]
        # Filter listings based on switch (corrected logic)
        if show_filter_mode == "Only unique listings":
            # For each DLD Permit Number (excluding empty/null), show only the first occurrence
            mask = (
                filtered_rent_listings["DLD Permit Number"].notna() &
                (filtered_rent_listings["DLD Permit Number"].astype(str).str.strip() != "")
            )
            filtered_rent_listings = filtered_rent_listings[mask]
            filtered_rent_listings = filtered_rent_listings.drop_duplicates(subset=["DLD Permit Number"], keep="first")
        elif show_filter_mode == "Only duplicate listings":
            filtered_rent_listings = filtered_rent_listings[filtered_rent_listings["DLD Permit Number"].isin(duplicate_dlds)]
        # (rest of code unchanged)

        # --- Asking Price Market Metrics (after filter is applied) ---
        st.markdown("---")
        st.subheader("Asking Price Market Metrics")
        asking_price = st.number_input("Enter your asking price (AED):", min_value=0, step=10000, value=0, format="%d", key="rent_asking_price")
        
        # Calculate metrics based on the filtered listings (respecting the "Show" filter)
        if not filtered_rent_listings.empty and "Price (AED)" in filtered_rent_listings.columns:
            # Only consider listings with a valid price
            price_col = "Price (AED)"
            price_series = filtered_rent_listings[price_col]
            if not isinstance(price_series, pd.Series):
                price_series = pd.Series(price_series)
            valid_prices = price_series.notna() & (price_series > 0)
            prices = filtered_rent_listings[valid_prices][price_col].astype(float)
            
            n_total = len(prices)
            n_below = (prices < asking_price).sum() if asking_price > 0 else 0
            n_above = (prices > asking_price).sum() if asking_price > 0 else 0
            n_same = (prices == asking_price).sum() if asking_price > 0 else 0
            pct_below = (n_below / n_total * 100) if n_total and asking_price > 0 else 0
            pct_above = (n_above / n_total * 100) if n_total and asking_price > 0 else 0
            pct_same = (n_same / n_total * 100) if n_total and asking_price > 0 else 0
            # Percentile rank
            percentile = (prices < asking_price).sum() / n_total * 100 if n_total and asking_price > 0 else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Listings Below", n_below, f"{pct_below:.1f}%")
            col2.metric("Listings Above", n_above, f"{pct_above:.1f}%")
            col3.metric("Listings At Same Price", n_same, f"{pct_same:.1f}%")
            col4.metric("Percentile", f"{percentile:.1f}%")
        else:
            st.info("No valid price data available for the selected filter.")
        
        st.markdown("---")

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
    st.markdown("<!-- RENT LISTINGS TAB END -->")

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
    st.markdown("<!-- RENTALS TAB START -->")
    st.title("Rental Transactions")
    
    # --- Step 1: Set up filters and logic ---
    
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
                    val_str = str(val).replace('\u00A0', '').strip()
                    if '-' in val_str:
                        parts = val_str.split('-')
                        if len(parts) == 2:
                            start = pd.to_datetime(parts[0].strip(), errors='coerce')
                            end = pd.to_datetime(parts[1].strip(), errors='coerce')
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
    
    # --- Filter Options ---
    st.subheader("Filter Options")
    
    # Row 1: Development, Community, Sub Community (context-aware)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Development filter - get from rental data
        development_options = sorted(rental_df['All Developments'].dropna().unique()) if not rental_df.empty and 'All Developments' in rental_df.columns else []
        selected_development = st.selectbox("Development", options=["All"] + development_options, key="rental_development_filter")
    
    with col2:
        # Community filter - context-aware based on selected development
        if selected_development != "All" and not rental_df.empty:
            comm_df = rental_df[rental_df['All Developments'] == selected_development]
            community_options = sorted(comm_df['Community/Building'].dropna().unique()) if 'Community/Building' in comm_df.columns else []
        else:
            community_options = sorted(rental_df['Community/Building'].dropna().unique()) if not rental_df.empty and 'Community/Building' in rental_df.columns else []
        selected_community = st.selectbox("Community", options=["All"] + community_options, key="rental_community_filter")
    
    with col3:
        # Sub Community filter - context-aware based on selected development and community
        if selected_development != "All" and selected_community != "All" and not rental_df.empty:
            subcom_df = rental_df[rental_df['All Developments'] == selected_development]
            subcom_df = subcom_df[subcom_df['Community/Building'] == selected_community]
            subcom_options = sorted(subcom_df['Sub Community/Building'].dropna().unique()) if 'Sub Community/Building' in subcom_df.columns else []
        elif selected_development != "All" and not rental_df.empty:
            subcom_df = rental_df[rental_df['All Developments'] == selected_development]
            subcom_options = sorted(subcom_df['Sub Community/Building'].dropna().unique()) if 'Sub Community/Building' in subcom_df.columns else []
        else:
            subcom_options = sorted(rental_df['Sub Community/Building'].dropna().unique()) if not rental_df.empty and 'Sub Community/Building' in rental_df.columns else []
        selected_subcommunity = st.selectbox("Sub Community/Building", options=["All"] + subcom_options, key="rental_subcommunity_filter")
    
    # Row 2: Status, Layout Type, (empty for spacing)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Status filter
        status_options = ["All", "🟢 Available", "🔴 Rented", "🟡 Expiring Soon", "🟣 Expiring <30 days", "🔵 Recently Vacant"]
        selected_status = st.selectbox("Status", options=status_options, key="rental_status_filter")
    
    with col2:
        # Layout Type filter - context-aware based on all location filters
        if selected_development != "All" or selected_community != "All" or selected_subcommunity != "All":
            layout_df = rental_df.copy()
            if selected_development != "All":
                layout_df = layout_df[layout_df['All Developments'] == selected_development]
            if selected_community != "All":
                layout_df = layout_df[layout_df['Community/Building'] == selected_community]
            if selected_subcommunity != "All":
                layout_df = layout_df[layout_df['Sub Community/Building'] == selected_subcommunity]
            layout_options = sorted(layout_df['Layout Type'].dropna().unique()) if 'Layout Type' in layout_df.columns else []
        else:
            layout_options = sorted(rental_df['Layout Type'].dropna().unique()) if not rental_df.empty and 'Layout Type' in rental_df.columns else []
        selected_layout = st.selectbox("Layout Type", options=["All"] + layout_options, key="rental_layout_filter")
    
    with col3:
        # Empty column for spacing
        pass
    
    # --- Data Filtering Logic ---
    def get_filtered_rental_data():
        """Get filtered rental data based on user selections."""
        if rental_df.empty:
            return None
        
        filtered_data = rental_df.copy()
        
        # Apply location filters
        if selected_development != "All":
            filtered_data = filtered_data[filtered_data['All Developments'] == selected_development]
        if selected_community != "All":
            filtered_data = filtered_data[filtered_data['Community/Building'] == selected_community]
        if selected_subcommunity != "All":
            filtered_data = filtered_data[filtered_data['Sub Community/Building'] == selected_subcommunity]
        
        # Apply layout type filter
        if selected_layout != "All":
            filtered_data = filtered_data[filtered_data['Layout Type'] == selected_layout]
        
        # Calculate status for all units
        today = pd.Timestamp.now().normalize()
        filtered_data['Contract Start'] = pd.to_datetime(filtered_data['Contract Start'], errors='coerce')
        filtered_data['Contract End'] = pd.to_datetime(filtered_data['Contract End'], errors='coerce')
        
        # Calculate days left and status
        days_left = (filtered_data['Contract End'] - today).dt.days
        days_since_end = (today - filtered_data['Contract End']).dt.days
        
        # Determine status
        conditions = [
            (filtered_data['Contract Start'].notna() & filtered_data['Contract End'].notna() & 
             (filtered_data['Contract Start'] <= today) & (filtered_data['Contract End'] >= today) & (days_left < 31), '🟣'),
            (filtered_data['Contract Start'].notna() & filtered_data['Contract End'].notna() & 
             (filtered_data['Contract Start'] <= today) & (filtered_data['Contract End'] >= today) & (days_left <= 90), '🟡'),
            (filtered_data['Contract Start'].notna() & filtered_data['Contract End'].notna() & 
             (filtered_data['Contract Start'] <= today) & (filtered_data['Contract End'] >= today), '🔴'),
            ((days_since_end > 0) & (days_since_end <= 60), '🔵'),
        ]
        filtered_data['Status'] = np.select([cond for cond, _ in conditions], [status for _, status in conditions], default='🟢')
        
        # Apply status filter
        if selected_status != "All":
            status_map = {
                "🟢 Available": "🟢",
                "🔴 Rented": "🔴",
                "🟡 Expiring Soon": "🟡",
                "🟣 Expiring <30 days": "🟣",
                "🔵 Recently Vacant": "🔵"
            }
            if selected_status in status_map:
                filtered_data = filtered_data[filtered_data['Status'] == status_map[selected_status]]
        
        return filtered_data
    
    # Get filtered data
    filtered_rental_data = get_filtered_rental_data()
    
    # Show data count
    if filtered_rental_data is not None:
        st.info(f"📊 Showing {len(filtered_rental_data)} rental records")
    else:
        st.info("📊 No rental data available")
    
    # --- Step 3: Metrics Dashboard ---
    if filtered_rental_data is not None and not filtered_rental_data.empty:
        st.subheader("📊 Rental Metrics Dashboard")
        
        # Calculate metrics
        total_units = len(filtered_rental_data)
        
        # Status counts
        status_counts = filtered_rental_data['Status'].value_counts()
        available_units = status_counts.get('🟢', 0)
        rented_units = status_counts.get('🔴', 0)
        expiring_soon = status_counts.get('🟡', 0)
        expiring_30_days = status_counts.get('🟣', 0)
        recently_vacant = status_counts.get('🔵', 0)
        
        # Calculate occupancy rate
        occupied_units = rented_units + expiring_soon + expiring_30_days
        occupancy_rate = (occupied_units / total_units * 100) if total_units > 0 else 0
        
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
        
        avg_rent = None
        if rent_col:
            valid_rents = filtered_rental_data[rent_col].dropna()
            if not valid_rents.empty:
                avg_rent = valid_rents.mean()
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Units", total_units)
            st.metric("Available Units", available_units, f"{available_units/total_units*100:.1f}%" if total_units > 0 else "0%")
        
        with col2:
            st.metric("Rented Units", rented_units, f"{rented_units/total_units*100:.1f}%" if total_units > 0 else "0%")
            st.metric("Occupancy Rate", f"{occupancy_rate:.1f}%")
        
        with col3:
            st.metric("Expiring <90 days", expiring_soon)
            st.metric("Expiring <30 days", expiring_30_days)
        
        with col4:
            st.metric("Recently Vacant", recently_vacant)
            if avg_rent is not None:
                st.metric("Avg Annual Rent", f"AED {avg_rent:,.0f}")
            else:
                st.metric("Avg Annual Rent", "N/A")
        
        # Status breakdown chart
        st.subheader("Status Breakdown")
        status_data = {
            'Available': available_units,
            'Rented': rented_units,
            'Expiring Soon': expiring_soon,
            'Expiring <30d': expiring_30_days,
            'Recently Vacant': recently_vacant
        }
        
        # Create a simple bar chart
        status_df = pd.DataFrame(list(status_data.items()), columns=['Status', 'Count'])
        status_df = status_df[status_df['Count'] > 0]  # Only show non-zero values
        
        if not status_df.empty:
            fig = go.Figure(data=[
                go.Bar(
                    x=status_df['Status'],
                    y=status_df['Count'],
                    marker_color=['#28a745', '#dc3545', '#ffc107', '#6f42c1', '#17a2b8']
                )
            ])
            fig.update_layout(
                title="Rental Status Distribution",
                xaxis_title="Status",
                yaxis_title="Number of Units",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Step 2: Table Display ---
    if filtered_rental_data is not None and not filtered_rental_data.empty:
        st.subheader("Rental Data Table")
        
        # Prepare data for display
        display_data = filtered_rental_data.copy()
        
        # Always split Evidence Date into Start Date and End Date
        if 'Evidence Date' in display_data.columns:
            def split_evidence_date(val):
                if pd.isnull(val):
                    return pd.NaT, pd.NaT
                val_str = str(val).replace('\u00A0', '').strip()
                if '/' in val_str:
                    parts = val_str.split('/')
                    if len(parts) == 2:
                        start = pd.to_datetime(parts[0].strip(), errors='coerce', dayfirst=True)
                        end = pd.to_datetime(parts[1].strip(), errors='coerce', dayfirst=True)
                        return start, end
                return pd.NaT, pd.NaT
            start_end = display_data['Evidence Date'].apply(split_evidence_date)
            display_data['Start Date'] = [d[0].strftime('%Y-%m-%d') if pd.notnull(d[0]) else '' for d in start_end]
            display_data['End Date'] = [d[1].strftime('%Y-%m-%d') if pd.notnull(d[1]) else '' for d in start_end]
        else:
            display_data['Start Date'] = ''
            display_data['End Date'] = ''
        
        # Calculate days left for display using End Date
        today = pd.Timestamp.now().normalize()
        if 'End Date' in display_data.columns:
            contract_end_dates = pd.to_datetime(display_data['End Date'], errors='coerce')
            days_left = (contract_end_dates - today).dt.days
            display_data['Days Left'] = days_left.apply(lambda x: f"{int(x)} days" if pd.notnull(x) and x >= 0 else "Expired" if pd.notnull(x) else "N/A")
        else:
            display_data['Days Left'] = "N/A"
        
        # Select columns to display
        columns_to_show = [
            'Status', 'Unit No.', 'All Developments', 'Community/Building', 'Sub Community/Building',
            'Layout Type', 'Beds', 'Unit Size (sq ft)', 'Start Date', 'End Date', 'Days Left'
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
            elif col == 'Days Left':
                gb.configure_column(col, width=120)
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
                st.markdown(f"**Start Date:** {selected_row.get('Start Date', 'N/A')}")
                st.markdown(f"**End Date:** {selected_row.get('End Date', 'N/A')}")
                st.markdown(f"**Days Left:** {selected_row.get('Days Left', 'N/A')}")
                
                # Show rent amount if available
                rent_amount = None
                for col in ['Annual Rent (AED)', 'Rent (AED)']:
                    if col in selected_row and pd.notnull(selected_row[col]):
                        rent_amount = selected_row[col]
                        break
                
                if rent_amount is not None:
                    st.markdown(f"**Annual Rent:** AED {rent_amount:,.0f}")
    
    # --- Step 4: Unit Selection with Detailed Info ---
    if selected_row is not None:
        st.markdown("---")
        st.subheader("📋 Detailed Unit Information")
        
        # Get the original unit number for transaction lookup
        unit_number = selected_row.get('Unit No.', '')
        
        if unit_number:
            # Look up transaction history for this unit
            unit_transactions = all_transactions[all_transactions['Unit No.'] == unit_number]
            
            # Display transaction history if available
            if not unit_transactions.empty:
                st.subheader("🏠 Transaction History")
                
                # Prepare transaction data for display
                txn_display = unit_transactions.copy()
                
                # Format dates
                if 'Evidence Date' in txn_display.columns:
                    txn_display['Evidence Date'] = pd.to_datetime(txn_display['Evidence Date'], errors='coerce')
                    txn_display['Evidence Date'] = txn_display['Evidence Date'].dt.strftime('%Y-%m-%d')
                
                # Select relevant columns
                txn_columns = [
                    'Evidence Date', 'Price (AED)', 'Unit Size (sq ft)', 'Beds', 
                    'Floor Level', 'Layout Type', 'Sales Recurrence'
                ]
                available_txn_columns = [col for col in txn_columns if col in txn_display.columns]
                txn_display = txn_display[available_txn_columns]
                
                # Rename columns for display
                txn_column_mapping = {
                    'Evidence Date': 'Transaction Date',
                    'Price (AED)': 'Sale Price (AED)',
                    'Unit Size (sq ft)': 'BUA (sq ft)',
                    'Floor Level': 'Floor'
                }
                txn_display = txn_display.rename(columns=txn_column_mapping)
                
                # Display transaction table
                st.dataframe(txn_display, use_container_width=True)
                
                # Calculate transaction statistics
                if 'Sale Price (AED)' in txn_display.columns:
                    prices = pd.to_numeric(txn_display['Sale Price (AED)'], errors='coerce').dropna()
                    if not prices.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Latest Sale", f"AED {prices.iloc[-1]:,.0f}")
                        with col2:
                            st.metric("Average Sale", f"AED {prices.mean():,.0f}")
                        with col3:
                            st.metric("Min Sale", f"AED {prices.min():,.0f}")
                        with col4:
                            st.metric("Max Sale", f"AED {prices.max():,.0f}")
            else:
                st.info("No transaction history found for this unit.")
            
            # Look up rental history for this unit
            unit_rentals = rental_df[rental_df['Unit No.'] == unit_number]
            
            if not unit_rentals.empty:
                st.subheader("🏠 Rental History")
                
                # Prepare rental history data
                rental_history = unit_rentals.copy()
                
                # Format dates
                if 'Contract Start' in rental_history.columns:
                    rental_history['Contract Start'] = pd.to_datetime(rental_history['Contract Start'], errors='coerce')
                    rental_history['Contract Start'] = rental_history['Contract Start'].dt.strftime('%Y-%m-%d')
                if 'Contract End' in rental_history.columns:
                    rental_history['Contract End'] = pd.to_datetime(rental_history['Contract End'], errors='coerce')
                    rental_history['Contract End'] = rental_history['Contract End'].dt.strftime('%Y-%m-%d')
                
                # Select relevant columns
                rental_columns = [
                    'Contract Start', 'Contract End', 'Layout Type', 'Beds', 'Unit Size (sq ft)'
                ]
                
                # Add rent amount column if available
                rent_col_candidates = [
                    'Annualised Rental Price(AED)',
                    'Annualised Rental Price (AED)',
                    'Rent (AED)', 'Annual Rent', 'Rent AED', 'Rent'
                ]
                for col in rent_col_candidates:
                    if col in rental_history.columns:
                        rental_columns.append(col)
                        break
                
                available_rental_columns = [col for col in rental_columns if col in rental_history.columns]
                rental_history = rental_history[available_rental_columns]
                
                # Rename columns for display
                rental_column_mapping = {
                    'Contract Start': 'Start Date',
                    'Contract End': 'End Date',
                    'Unit Size (sq ft)': 'BUA (sq ft)'
                }
                rental_history = rental_history.rename(columns=rental_column_mapping)
                
                # Display rental history table
                st.dataframe(rental_history, use_container_width=True)
            else:
                st.info("No rental history found for this unit.")
        
        # Show rental vs sale comparison if both available
        if not unit_transactions.empty and not unit_rentals.empty:
            st.subheader("📊 Rental vs Sale Analysis")
            
            # Get latest transaction and rental
            latest_txn = unit_transactions.iloc[-1]
            latest_rental = unit_rentals.iloc[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Latest Sale:**")
                if 'Price (AED)' in latest_txn:
                    sale_price = latest_txn['Price (AED)']
                    st.metric("Sale Price", f"AED {sale_price:,.0f}")
                    
                    # Calculate rental yield if both available
                    rent_amount = None
                    for col in rent_col_candidates:
                        if col in latest_rental and pd.notnull(latest_rental[col]):
                            rent_amount = latest_rental[col]
                            break
                    
                    if rent_amount is not None:
                        rental_yield = (rent_amount / sale_price * 100) if sale_price > 0 else 0
                        st.metric("Rental Yield", f"{rental_yield:.2f}%")
            
            with col2:
                st.markdown("**Current Rental:**")
                rent_amount = None
                for col in rent_col_candidates:
                    if col in latest_rental and pd.notnull(latest_rental[col]):
                        rent_amount = latest_rental[col]
                        break
                
                if rent_amount is not None:
                    st.metric("Annual Rent", f"AED {rent_amount:,.0f}")
                    monthly_rent = rent_amount / 12
                    st.metric("Monthly Rent", f"AED {monthly_rent:,.0f}")
    
    # Only show table/metrics if at least one filter is selected
    if selected_development == "All" and selected_community == "All" and selected_subcommunity == "All":
        st.info("Please select at least one filter (Development, Community, or Sub Community) to view rental data.")
    
    st.markdown("<!-- RENTALS TAB END -->")

