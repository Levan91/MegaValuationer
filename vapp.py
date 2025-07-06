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
    # Clear other filters when unit is selected to avoid conflicts
    unit_number = st.session_state.get("unit_number", "")
    if unit_number:
        # Clear location filters when unit is selected (unless locked)
        if not st.session_state.get("lock_location_filters", False):
            st.session_state.pop("development", None)
            st.session_state.pop("community", None)
            st.session_state.pop("subcommunity", None)
        # Clear property filters
        st.session_state.pop("property_type", None)
        st.session_state.pop("bedrooms", None)
        st.session_state.pop("bua", None)
        st.session_state.pop("plot_size", None)
        st.session_state.pop("layout_type", None)

def _on_development_change():
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

    st.session_state.pop("community", None)
    st.session_state.pop("subcommunity", None)

def _on_community_change():
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

    st.session_state.pop("subcommunity", None)

def _on_subcommunity_change():
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
    else:
        st.session_state.pop("unit_number", None)

def _on_bedrooms_change():
    # Clear only the unit number selection when bedrooms changes
    st.session_state["unit_number"] = ""

# --- Page Config ---
st.set_page_config(page_title="Valuation App V2", layout="wide")

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
rental_file_path = os.path.join(os.path.dirname(__file__), "Data", "Rentals", "Maple_rentals.xlsx")
if os.path.exists(rental_file_path):
    try:
        rental_df = pd.read_excel(rental_file_path)
        rental_df.columns = rental_df.columns.str.replace('\xa0', '', regex=False).str.strip()
        # Normalize unit numbers for matching
        rental_df['Unit No.'] = rental_df['Unit No.'].astype(str).str.strip().str.upper()
        # Split Evidence Date into start/end
        rental_df[['Contract Start', 'Contract End']] = (
            rental_df['Evidence Date']
            .str.replace('\u00A0', '', regex=False)
            .str.split('/', expand=True)
        )
        rental_df['Contract Start'] = pd.to_datetime(
            rental_df['Contract Start'].str.strip(), errors='coerce', dayfirst=True
        )
        rental_df['Contract End'] = pd.to_datetime(
            rental_df['Contract End'].str.strip(), errors='coerce', dayfirst=True
        )
    except Exception as e:
        st.warning(f"Failed to load rental data: {e}")
else:
    import pandas as pd
    rental_df = pd.DataFrame(columns=pd.Index(['Unit No.', 'Contract Start', 'Contract End']))

# --- Cached loader for all transaction files ---
@cache_data
def load_all_listings(listings_dir):
    """Load and concatenate all Excel listing files, with caching."""
    files = [f for f in os.listdir(listings_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
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
        if "included_txn_files" not in st.session_state:
            st.session_state["included_txn_files"] = all_txn_files
        st.multiselect(
            "Include transaction files:",
            options=all_txn_files,
            key="included_txn_files"
        )

    # --- Filter Mode ---
    st.subheader("Filter Mode")
    if "filter_mode" not in st.session_state:
        st.session_state["filter_mode"] = "Unit Selection"
    filter_mode = st.radio("Select filter mode", ["Unit Selection", "Manual Selection"], key="filter_mode", horizontal=True)

    # --- Property Filters ---
    # Show property info filters only in Manual Selection mode, wrapped in an expander
    if filter_mode == "Manual Selection":
        st.subheader("Property Info")
        with st.expander("🛠️ Manual Property Info", expanded=True):
            if "property_type" not in st.session_state:
                st.session_state["property_type"] = ""
            if "bedrooms" not in st.session_state:
                st.session_state["bedrooms"] = ""
            if "bua" not in st.session_state:
                st.session_state["bua"] = ""
            if "plot_size" not in st.session_state:
                st.session_state["plot_size"] = ""
            prop_type_col = all_transactions['Unit Type']
            if not isinstance(prop_type_col, pd.Series):
                prop_type_col = pd.Series(prop_type_col)
            prop_type_options = prop_type_col.dropna().unique().tolist()
            property_type = st.selectbox(
                "Property Type",
                options=[""] + sorted(prop_type_options),
                index=0,
                key="property_type"
            )
            beds_col = all_transactions['Beds']
            if not isinstance(beds_col, pd.Series):
                beds_col = pd.Series(beds_col)
            bedrooms_options = beds_col.dropna().astype(str).unique().tolist()
            bedrooms = st.selectbox(
                "Bedrooms",
                options=[""] + sorted(bedrooms_options),
                index=0,
                key="bedrooms"
            )
            bua = st.text_input("BUA (sq ft)", value=st.session_state["bua"], key="bua")
            plot_size = st.text_input("Plot Size (sq ft)", value=st.session_state["plot_size"], key="plot_size")
    else:
        # In Unit Selection mode, show as disabled info fields (do not show property info fields)
        if "property_type" not in st.session_state:
            st.session_state["property_type"] = ""
        if "bedrooms" not in st.session_state:
            st.session_state["bedrooms"] = ""
        if "bua" not in st.session_state:
            st.session_state["bua"] = ""
        if "plot_size" not in st.session_state:
            st.session_state["plot_size"] = ""
        property_type = st.session_state.get("property_type", "")
        bedrooms = st.session_state.get("bedrooms", "")
        bua = st.session_state.get("bua", "")
        plot_size = st.session_state.get("plot_size", "")

    if "unit_number" not in st.session_state:
        st.session_state["unit_number"] = ""
    unit_number = st.session_state.get("unit_number", "")
    # Ensure variables are always defined to avoid NameError in layout filtering
    if "development" not in st.session_state:
        st.session_state["development"] = ""
    if "community" not in st.session_state:
        st.session_state["community"] = []
    if "subcommunity" not in st.session_state:
        st.session_state["subcommunity"] = ""
    development = st.session_state.get("development", "")
    community = st.session_state.get("community", [])
    subcommunity = st.session_state.get("subcommunity", "")

    # --- If in Live Listings context, sync location/layout filters from selected unit ---
    # This block ensures that when a unit is selected, the location and layout filters update accordingly
    # (applies to both transactions and listings, but especially relevant for listings)
    # Use all_listings columns if available, else fallback to all_transactions
    units_df = all_listings if 'all_listings' in locals() and not all_listings.empty else all_transactions
    selected_development = development
    selected_community = community
    selected_sub_communities = community if isinstance(community, list) else [community] if community else []
    layout_type_options = []
    selected_layout_type = None
    if unit_number:
        unit_row_df = all_transactions[all_transactions['Unit No.'] == unit_number]  # type: ignore
        if not isinstance(unit_row_df, pd.DataFrame):
            unit_row_df = pd.DataFrame(unit_row_df)
        unit_row = unit_row_df.iloc[0]
        if not unit_row.empty:
            # Fix: handle Series vs DataFrame for .columns and .values
            if isinstance(unit_row, pd.Series):
                selected_development = unit_row.get("Development", unit_row.get("All Developments", ""))
                selected_community = unit_row.get("Community", unit_row.get("Community/Building", ""))
                sub_community = unit_row.get("Subcommunity", unit_row.get("Sub Community / Building", ""))
                # Fix: robustly handle sub_community as scalar or iterable (but not string/None/DataFrame)
                import numpy as np
                if sub_community is None:
                    sub_community_val = None
                elif isinstance(sub_community, pd.DataFrame):
                    vals = sub_community.values.flatten()
                    sub_community_val = next((v for v in vals if pd.notna(v)), None)
                elif isinstance(sub_community, pd.Series):
                    vals = sub_community.tolist()
                    sub_community_val = next((v for v in vals if pd.notna(v)), None)
                elif hasattr(sub_community, '__iter__') and not isinstance(sub_community, str):
                    vals = list(sub_community)
                    sub_community_val = next((v for v in vals if pd.notna(v)), None)
                else:
                    sub_community_val = sub_community if pd.notna(sub_community) else None
                selected_sub_communities = [sub_community_val] if sub_community_val is not None else []
                layout = unit_row.get("Layout Type", "")
            else:
                selected_development = unit_row["Development"].values[0] if "Development" in unit_row.columns else unit_row["All Developments"].values[0] if "All Developments" in unit_row.columns else ""
                selected_community = unit_row["Community"].values[0] if "Community" in unit_row.columns else unit_row["Community/Building"].values[0] if "Community/Building" in unit_row.columns else ""
                sub_community = unit_row["Subcommunity"].values[0] if "Subcommunity" in unit_row.columns else unit_row["Sub Community / Building"].values[0] if "Sub Community / Building" in unit_row.columns else ""
                # Fix: robustly handle sub_community as scalar or iterable (but not string/None/DataFrame)
                import numpy as np
                if sub_community is None:
                    sub_community_val = None
                elif isinstance(sub_community, pd.DataFrame):
                    vals = sub_community.values.flatten()
                    sub_community_val = next((v for v in vals if pd.notna(v)), None)
                elif isinstance(sub_community, pd.Series):
                    vals = sub_community.tolist()
                    sub_community_val = next((v for v in vals if pd.notna(v)), None)
                elif hasattr(sub_community, '__iter__') and not isinstance(sub_community, str):
                    vals = list(sub_community)
                    sub_community_val = next((v for v in vals if pd.notna(v)), None)
                else:
                    sub_community_val = sub_community if pd.notna(sub_community) else None
                selected_sub_communities = [sub_community_val] if sub_community_val is not None else []
                layout = unit_row["Layout Type"].values[0] if "Layout Type" in unit_row.columns else ""


    # --- Unit Number Filter ---
    st.subheader("Unit Number")
    if not all_transactions.empty:
        unit_df = all_transactions.copy()
        
        # Get current filter values from session state
        current_development = st.session_state.get("development", "")
        current_community = st.session_state.get("community", [])
        current_subcommunity = st.session_state.get("subcommunity", "")
        
        # Apply filters based on current session state values
        if current_development:
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
            # Ensure current_subcommunity is a list
            if not isinstance(current_subcommunity, (list, tuple, set)):
                current_subcommunity = [current_subcommunity] if current_subcommunity else []
            unit_df = unit_df[subcommunity_col.isin(current_subcommunity)]
        
        unit_no_col = unit_df['Unit No.']
        if not isinstance(unit_no_col, pd.Series):
            unit_no_col = pd.Series(unit_no_col)
        unit_number_options = sorted(unit_no_col.dropna().unique())
    else:
        unit_number_options = []
    
    current = st.session_state.get("unit_number", "")
    options = [""] + unit_number_options
    # Safer index calculation
    try:
        default_idx = options.index(current) if current in options else 0
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

    # If the unit selection logic above found layout_type_options, use them
    if layout_type_options:
        layout_options = layout_type_options
    else:
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

        # If the unit selection logic above found layout_type_options, use them
        if layout_type_options:
            layout_options = layout_type_options
        else:
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
        st.write(f"**Selected Layout Type:** {selected_layout_type}")
        st.write(f"**Current Community:** {current_community}")
        st.write(f"**Current Subcommunity:** {current_subcommunity}")
    
    # If the unit selection logic above found selected_layout_type, use it
    layout_type = st.multiselect(
        "Layout Type",
        options=layout_options,
        default=[selected_layout_type] if selected_layout_type and selected_layout_type in layout_options else ([mapped_layout] if mapped_layout in layout_options else []),
        key="layout_type"
    )

    # --- Location Filters ---
    st.subheader("Location")
    lock_location_filters = st.checkbox("🔒 Lock Location Filters", value=False, key="lock_location_filters")
    if not all_transactions.empty:
        dev_options = sorted(pd.Series(all_transactions['All Developments']).dropna().unique())
        com_options = sorted(pd.Series(all_transactions['Community/Building']).dropna().unique())
        subcom_options = sorted(pd.Series(all_transactions['Sub Community / Building']).dropna().unique())
    else:
        dev_options = com_options = subcom_options = []

    development = st.session_state.get("development", "")
    community = st.session_state.get("community", [])
    subcommunity = st.session_state.get("subcommunity", "")
    property_type = ""
    bedrooms = ""
    floor = ""
    bua = ""
    plot_size = ""
    sales_recurrence = "All"

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
            development = st.session_state.get("development", "")
            community = st.session_state.get("community", [])
            subcommunity = st.session_state.get("subcommunity", "")
            property_type = st.session_state.get("property_type", "")
            bedrooms = st.session_state.get("bedrooms", "")
            bua = st.session_state.get("bua", "")
            plot_size = st.session_state.get("plot_size", "")
    else:
        development = st.session_state.get("development", "")
        community = st.session_state.get("community", [])
        subcommunity = st.session_state.get("subcommunity", "")
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
    # Ensure subcom_options is a list (not a generator)
    subcom_options = list(subcom_options) if not isinstance(subcom_options, list) else subcom_options
    subcommunity = st.multiselect(
        "Sub community / Building",
        options=subcom_options,
        default=[subcommunity] if subcommunity in subcom_options else [],
        key="subcommunity",
        on_change=_on_subcommunity_change
    )


    # --- Time Period Filter ---
    st.subheader("Time Period")
    time_filter_mode = st.selectbox("Time Filter Mode", ["", "Last N Days", "After Date", "From Date to Date"], index=1, key="time_filter_mode", placeholder="")
    last_n_days = None
    after_date = None
    date_range = (None, None)
    if time_filter_mode == "Last N Days":
        last_n_days = st.number_input("Enter number of days", min_value=1, step=1, value=365, key="last_n_days")
    elif time_filter_mode == "After Date":
        after_date = st.date_input("Select start date", key="after_date")
    elif time_filter_mode == "From Date to Date":
        date_range = st.date_input("Select date range", value=(None, None), key="date_range")
    st.markdown("---")


    # Floor Tolerance filter toggle
    enable_floor_tol = st.checkbox("Enable floor tolerance", value=False, key="enable_floor_tol")
    if enable_floor_tol and property_type == "Apartment":
        floor_tolerance = st.number_input(
            "Floor tolerance (± floors)", min_value=0, step=1, value=0, key="floor_tolerance"
        )
    else:
        floor_tolerance = 0
    # BUA Tolerance filter toggle
    enable_bua_tol = st.checkbox("Enable BUA tolerance", value=False, key="enable_bua_tol")
    if enable_bua_tol:
        bua_tolerance = st.number_input("BUA tolerance (± sq ft)", min_value=0, step=1, value=0, key="bua_tolerance")
    else:
        bua_tolerance = 0
    # Plot Size Tolerance filter toggle
    enable_plot_tol = st.checkbox("Enable Plot Size tolerance", value=False, key="enable_plot_tol")
    if enable_plot_tol:
        plot_tolerance = st.number_input("Plot Size tolerance (± sq ft)", min_value=0, step=1, value=0, key="plot_tolerance")
    else:
        plot_tolerance = 0

    sales_recurrence = st.selectbox("Sales Recurrence", options=sales_rec_options, index=sales_rec_options.index(sales_recurrence) if sales_recurrence in sales_rec_options else 0, key="sales_recurrence", placeholder="")

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
        # Ensure subcommunity is a list
        if not isinstance(subcommunity, (list, tuple, set)):
            subcommunity = [subcommunity] if subcommunity else []
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
        # Ensure subcommunity is a list
        if not isinstance(subcommunity, (list, tuple, set)):
            subcommunity = [subcommunity] if subcommunity else []
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

 # --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Live Listings", "Trend & Valuation", "Rentals"])

if st.session_state["active_tab"] == "Dashboard":
    # Dashboard tab content
    # ... existing Dashboard code ...
    pass
elif st.session_state["active_tab"] == "Live Listings":
    # Live Listings tab content
    # ... existing Live Listings code ...
    pass
elif st.session_state["active_tab"] == "Trend & Valuation":
    # Trend & Valuation tab content
    # ... existing Trend & Valuation code ...
    pass
elif st.session_state["active_tab"] == "Rentals":
    # Rentals tab content
    # ... existing Rentals code ...
    pass

