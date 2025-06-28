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
    monthly = df2.set_index('ds')['y'].resample('ME').mean().reset_index()
    monthly['y_avg'] = df2['y'].mean()
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
        cutoff = datetime.today() - timedelta(days=last_n_days)
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
    # Only clear filters if a unit is actually selected (not when clearing)
    unit_number = st.session_state.get("unit_number", "")
    if unit_number:
        # Clear property filters when unit is selected to avoid conflicts
        st.session_state.pop("property_type", None)
        st.session_state.pop("bedrooms", None)
        st.session_state.pop("bua", None)
        st.session_state.pop("plot_size", None)
        st.session_state.pop("layout_type", None)
        # Only clear location filters if not locked AND if they don't match the unit
        if not st.session_state.get("lock_location_filters", False):
            # Check if current location filters match the selected unit
            unit_row = all_transactions[all_transactions['Unit No.'] == unit_number]
            if isinstance(unit_row, pd.DataFrame) and not unit_row.empty:
                unit_info = unit_row.iloc[0]
                current_dev = st.session_state.get("development", "")
                current_comm = st.session_state.get("community", [])
                current_subcomm = st.session_state.get("subcommunity", [])
                
                # Only clear if they don't match
                if current_dev and current_dev != unit_info.get('All Developments', ''):
                    st.session_state.pop("development", None)
                if current_comm and unit_info.get('Community/Building', '') not in current_comm:
                    st.session_state.pop("community", None)
                if current_subcomm and unit_info.get('Sub Community / Building', '') not in current_subcomm:
                    st.session_state.pop("subcommunity", None)

def _on_development_change():
    """Callback function for development selection changes"""
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
    
    # Don't automatically clear community/subcommunity - let user manage them

def _on_community_change():
    """Callback function for community selection changes"""
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
    
    # Don't automatically clear subcommunity - let user manage it

def _on_subcommunity_change():
    """Callback function for subcommunity selection changes"""
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
    # Don't clear unit_number when subcommunity is empty - let user manage it

def _on_bedrooms_change():
    """Callback function for bedrooms selection changes"""
    # Only clear unit number if it doesn't match the selected bedrooms
    unit_number = st.session_state.get("unit_number", "")
    bedrooms = st.session_state.get("bedrooms", "")
    
    if unit_number and bedrooms:
        unit_row = all_transactions[all_transactions['Unit No.'] == unit_number]
        if isinstance(unit_row, pd.DataFrame) and not unit_row.empty:
            unit_beds = str(unit_row.iloc[0].get('Beds', ''))
            if unit_beds != bedrooms:
                st.session_state.pop("unit_number", None)

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
for file in layout_files:
    file_path = os.path.join(layout_dir, file)
    try:
        df_l = pd.read_excel(file_path, sheet_name='Types')
        # Clean column names immediately
        df_l.columns = df_l.columns.str.replace('\xa0', '', regex=False).str.strip()
        project_name = os.path.splitext(file)[0]
        if project_name.lower().endswith('_layouts'):
            project_name = project_name[:-len('_layouts')]
        # Keep only Unit No. and Layout Type
        df_l = df_l.loc[:, ['Unit No.', 'Layout Type']]
        df_l['Unit No.'] = df_l['Unit No.'].astype(str).str.strip()
        df_l['Project'] = project_name
        layout_dfs.append(df_l)
    except Exception:
        continue
if layout_dfs:
    layout_map_df = pd.concat(layout_dfs, ignore_index=True).drop_duplicates(subset=['Unit No.'])
    layout_map = dict(zip(layout_map_df['Unit No.'], layout_map_df['Layout Type']))
else:
    layout_map_df = pd.DataFrame(columns=pd.Index(['Unit No.', 'Layout Type', 'Project']))
    layout_map = {}

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
    st.markdown("## 🏠 MegaValuationer")
    st.markdown("### 🧹 Reset & Refresh")
    st.button("🔄 Reset Filters", key="reset_filters")
    st.button("🔄 Refresh All Data", key="refresh_all")
    st.markdown("---")
    st.markdown("### 📂 Transaction Files")
    with st.expander("Select Transaction Files", expanded=False):
        all_txn_files = [f for f in os.listdir(transactions_dir) if f.endswith('.xlsx') and not f.startswith('~$')]
        st.multiselect(
            "Include transaction files:",
            options=all_txn_files,
            key="included_txn_files"
        )
    st.markdown("---")
    st.markdown("### 🔎 Filter Mode")
    filter_mode = st.radio("Select filter mode", ["Unit Selection", "Manual Selection"], key="filter_mode", horizontal=True)

    # --- Property Filters ---
    # Show property info filters only in Manual Selection mode, wrapped in an expander
    if filter_mode == "Manual Selection":
        st.subheader("Property Info")
        with st.expander("🛠️ Manual Property Info", expanded=True):
            property_type = st.session_state.get("property_type", "")
            bedrooms = st.session_state.get("bedrooms", "")
            bua = st.session_state.get("bua", "")
            plot_size = st.session_state.get("plot_size", "")

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

            bua = st.text_input("BUA (sq ft)", value=bua, key="bua")
            plot_size = st.text_input("Plot Size (sq ft)", value=plot_size, key="plot_size")
    else:
        # In Unit Selection mode, show as disabled info fields (do not show property info fields)
        property_type = st.session_state.get("property_type", "")
        bedrooms = st.session_state.get("bedrooms", "")
        bua = st.session_state.get("bua", "")
        plot_size = st.session_state.get("plot_size", "")

    unit_number = st.session_state.get("unit_number", "")
    # Ensure variables are always defined to avoid NameError in layout filtering
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
            unit_df = unit_df[subcommunity_col.isin([current_subcommunity] if isinstance(current_subcommunity, str) else current_subcommunity)]
        
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

            property_type = unit_row['Unit Type']
            bedrooms = str(unit_row['Beds'])
            floor = str(unit_row['Floor Level']) if pd.notna(unit_row['Floor Level']) else ""
            bua = unit_row['Unit Size (sq ft)'] if pd.notna(unit_row['Unit Size (sq ft)']) else ""
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
    if enable_floor_tol:
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

from datetime import datetime, timedelta

# --- Filter Transactions based on Time Period ---
filtered_transactions = all_transactions.copy()

# Apply Time Period Filter
if not filtered_transactions.empty:
    if 'Evidence Date' in filtered_transactions.columns:
        filtered_transactions['Evidence Date'] = pd.to_datetime(filtered_transactions['Evidence Date'], errors='coerce')

        today = datetime.today()

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
    if unit_number and st.session_state.get("enable_bua_tol", False):
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

 # --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Live Listings", "Trend & Valuation", "Other"])

with tab1:
    # st.warning("DASHBOARD TEST MARKER - If you see this, you are in the Dashboard tab!")
    st.title("Real Estate Valuation Dashboard")

    # --- Compact Selected Unit Info Card ---
    if unit_number:
        # Gather info for the card
        layout_val = all_transactions.loc[all_transactions["Unit No."] == unit_number, "Layout Type"].values
        layout_type_val = layout_val[0] if layout_val.size > 0 else ""
        comm_val = community[0] if isinstance(community, list) and community else community
        st.markdown(
            f"""
            <div style='background: #f8f9fa; border-radius: 10px; padding: 12px 18px; margin-bottom: 10px; display: flex; align-items: center; font-size: 1.1em;'>
                <span style='margin-right: 18px;'>📦 <b>{unit_number}</b></span>
                <span style='margin-right: 18px;'>🏙️ {comm_val}</span>
                <span style='margin-right: 18px;'>📐 {layout_type_val}</span>
                <span style='margin-right: 18px;'>🛏️ {bedrooms} Beds</span>
                <span style='margin-right: 18px;'>📏 {bua} sqft</span>
                <span style='margin-right: 18px;'>🌳 {plot_size} sqft</span>
                <span style='margin-right: 18px;'>🛗 {floor}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    # Remove the old verbose Selected Unit Info markdowns

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


with tab2:
    if isinstance(all_listings, pd.DataFrame) and all_listings.shape[0] > 0:
        st.subheader("All Live Listings")
        # Apply sidebar filters to live listings (NO date-based or "Days Listed"/"Listed When"/"Listed Date" filtering here)
        filtered_listings = all_listings.copy()
        if development and 'Development' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Development'] == development]
        if not isinstance(filtered_listings, pd.DataFrame):
            filtered_listings = pd.DataFrame(filtered_listings)
        if community and 'Community' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Community'].isin(community)]
            if not isinstance(filtered_listings, pd.DataFrame):
                filtered_listings = pd.DataFrame(filtered_listings)
        if subcommunity and 'Subcommunity' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Subcommunity'].isin(subcommunity)]
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
        # Remove tolerance filters from live listings - they should only apply to transactions
        if layout_type and 'Layout Type' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Layout Type'].isin(layout_type)]  # type: ignore
            if not isinstance(filtered_listings, pd.DataFrame):
                filtered_listings = pd.DataFrame(filtered_listings)
        if sales_recurrence != "All" and 'Sales Recurrence' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Sales Recurrence'] == sales_recurrence]
            if not isinstance(filtered_listings, pd.DataFrame):
                filtered_listings = pd.DataFrame(filtered_listings)
        # Exclude listings marked as not available
        if 'Availability' in filtered_listings.columns:
            avail_col = filtered_listings['Availability']
            if not isinstance(avail_col, pd.Series):
                avail_col = pd.Series(avail_col)
            filtered_listings = filtered_listings[avail_col.astype(str).str.strip().str.lower() != "not available"]
            if not isinstance(filtered_listings, pd.DataFrame):
                filtered_listings = pd.DataFrame(filtered_listings)

        # Hide certain columns but keep them in the DataFrame (do NOT filter by Days Listed or any date here)
        columns_to_hide = ["Reference Number", "URL", "Source File", "Unit No.", "Unit Number", "Listed When", "Listed when"]
        visible_columns = [c for c in filtered_listings.columns if c not in columns_to_hide] + ["URL"]

        # Show count of live listings
        st.markdown(f"**Showing {filtered_listings.shape[0]} live listings**")

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
            theme='alpine'
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

with tab3:
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

    # Prepare filtered listing DataFrame
listing_df = all_listings.copy() if 'all_listings' in locals() else pd.DataFrame()
if verified_only and 'Verified' in listing_df.columns:
        listing_df = listing_df[listing_df['Verified'].str.lower() == 'yes']
if 'Days Listed' in listing_df.columns:
        # Ensure Days Listed is numeric for pd.to_timedelta
        listing_df['Days Listed'] = pd.to_numeric(listing_df['Days Listed'], errors='coerce')
        # Ensure Days Listed is a pandas Series before calling pd.to_timedelta
        days_listed_col = listing_df['Days Listed']
        if not isinstance(days_listed_col, pd.Series):
            days_listed_col = pd.Series(days_listed_col)
        listing_df['Listing Date'] = datetime.today() - pd.to_timedelta(days_listed_col, unit='D')
        cutoff_listings = datetime.today() - timedelta(days=listings_days)
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
        # Overlay transactions as blue dots
        filtered_transactions = pd.DataFrame(filtered_transactions)
        if not filtered_transactions.empty and 'Evidence Date' in filtered_transactions.columns and 'Price (AED)' in filtered_transactions.columns:
            transaction_df = filtered_transactions.copy()
            transaction_df = pd.DataFrame(transaction_df)
            transaction_df['Evidence Date'] = pd.to_datetime(transaction_df['Evidence Date'], errors='coerce')
            transaction_df = transaction_df.dropna(subset=['Evidence Date', 'Price (AED)'])
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
        # Scenario line (only if adjustment is nonzero)
        if scenario_pct != 0:
            forecast['yhat_scenario'] = forecast['yhat_actual'] * (1 + scenario_pct/100)
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_scenario'],
                mode='lines+markers', name=f"Scenario ({scenario_pct:+}%)",
                line=dict(dash='dot')
            ))
        # Overlay live listings: verified (green) vs non-verified (red)
        if not listing_df.empty and 'Verified' in listing_df.columns:
            ver_df = listing_df[listing_df['Verified'].str.lower() == 'yes']
            nonver_df = listing_df[listing_df['Verified'].str.lower() != 'yes']
        else:
            ver_df = listing_df.copy()
            nonver_df = listing_df.iloc[0:0]

        if not ver_df.empty:
            fig.add_trace(go.Scatter(
                x=ver_df["Listed When"] if "Listed When" in ver_df.columns else ver_df.get("Listing Date", ver_df.index),
                y=ver_df['Price (AED)'],
                mode='markers',
                marker=dict(symbol='diamond', size=8, opacity=0.8, color='green'),
                name='Verified Listings',
                customdata=ver_df["URL"],
                text=ver_df.apply(lambda row: f'{int(row["Days Listed"])} days ago | {row["Layout Type"]}' if pd.notnull(row.get("Days Listed")) and pd.notnull(row.get("Layout Type")) else "", axis=1) if "Days Listed" in ver_df.columns and "Layout Type" in ver_df.columns else "",
                hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<br><a href='%{customdata}' target='_blank'>View Listing</a><extra></extra>"
            ))
        if not nonver_df.empty:
            fig.add_trace(go.Scatter(
                x=nonver_df["Listed When"] if "Listed When" in nonver_df.columns else nonver_df.get("Listing Date", nonver_df.index),
                y=nonver_df['Price (AED)'],
                mode='markers',
                marker=dict(symbol='diamond', size=8, opacity=0.8, color='red'),
                name='Non-verified Listings',
                customdata=nonver_df["URL"],
                text=nonver_df.apply(lambda row: f'{int(row["Days Listed"])} days ago | {row["Layout Type"]}' if pd.notnull(row.get("Days Listed")) and pd.notnull(row.get("Layout Type")) else "", axis=1) if "Days Listed" in nonver_df.columns and "Layout Type" in nonver_df.columns else "",
                hovertemplate="Date: %{x|%b %d, %Y}<br>Price: AED %{y:,.0f}<br>%{text}<br><a href='%{customdata}' target='_blank'>View Listing</a><extra></extra>"
            ))
        fig.update_layout(
            title='Prophet Forecast (Actual Value)',
            xaxis_title='Month', yaxis_title='AED'
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
        """, height=600, scrolling=True)
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




with tab4:
    st.write("Placeholder for other functionality.")

import logging
logger = logging.getLogger(__name__)

def safe_operation(func):
    """Decorator for safe operations with proper error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            st.error(f"Operation failed: {str(e)}")
            return None
    return wrapper

def enhanced_similarity_model(df, numeric_features, categorical_features, weights=None):
    """Enhanced similarity model with feature weighting"""
    # Add feature importance calculation
    # Implement custom distance metrics
    # Add similarity score confidence intervals

def auto_tune_prophet(monthly_df, max_evals=50):
    """Automatically tune Prophet hyperparameters"""
    # Implement Bayesian optimization
    # Add model comparison (Prophet vs other models)
    # Include seasonality detection

def validate_data_quality(df, required_columns):
    """Validate data quality and completeness"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for data types, ranges, outliers
    # Return validation report
    return True

# Add ML pipeline for automated valuation
class ValuationPipeline:
    def __init__(self):
        self.models = {
            'similarity': None,
            'prophet': None,
            'regression': None
        }
    
    def fit_all_models(self, data):
        """Fit all models in parallel"""
        # Implementation

# Add real-time data refresh capabilities
@st.cache_data(ttl=300)  # 5 minutes
def get_live_data():
    """Fetch live data from external APIs"""
    # Implementation

# Track user interactions for improvement
def track_user_behavior(action, details):
    """Track user interactions for analytics"""
    # Implementation

def detect_outliers_iqr(df, column='y', factor=1.5):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def calculate_forecast_metrics(actual, predicted):
    """Calculate multiple forecast accuracy metrics"""
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return {
        'MAPE': mape,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2_score(actual, predicted)
    }

def detect_seasonality(monthly_df):
    """Detect if data has significant seasonality"""
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(monthly_df['y'], period=12)
        seasonal_strength = np.std(decomposition.seasonal) / np.std(decomposition.resid)
        return seasonal_strength > 0.1  # Threshold for significant seasonality
    except ImportError:
        # statsmodels not available, return False
        return False
    except Exception:
        # Other errors, return False
        return False

def compare_prophet_models(monthly_df):
    """Compare different Prophet configurations"""
    models = {
        'Linear': {'growth': 'linear'},
        'Logistic': {'growth': 'logistic', 'cap': monthly_df['y'].max() * 1.2},
        'Custom': {'growth': 'linear', 'changepoint_prior_scale': 0.01}
    }
    
    results = {}
    for name, config in models.items():
        m = Prophet(**config)
        m.fit(monthly_df)
        # Cross-validation
        df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='365 days')
        results[name] = performance_metrics(df_cv)
    
    return results

# ADD: Better uncertainty handling
def calculate_forecast_uncertainty(forecast, confidence_level=0.8):
    """Calculate forecast uncertainty with custom confidence intervals"""
    z_score = norm.ppf((1 + confidence_level) / 2)
    forecast['custom_lower'] = forecast['yhat'] - z_score * forecast['yhat_lower']
    forecast['custom_upper'] = forecast['yhat'] + z_score * forecast['yhat_upper']
    return forecast

# ENHANCE: Better changepoint visualization
def analyze_changepoints(m, monthly_df):
    """Analyze and visualize changepoints"""
    fig = m.plot_components(forecast)
    
    # Add changepoint significance
    changepoint_dates = m.changepoints
    changepoint_effects = m.params['delta'].flatten()
    
    # Filter significant changepoints
    significant_changes = changepoint_effects[np.abs(changepoint_effects) > np.std(changepoint_effects)]
    
    return fig, significant_changes

# ADD: Memory-efficient data processing
@st.cache_data(ttl=3600)
def process_large_dataset(df, chunk_size=10000):
    """Process large datasets in chunks"""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        # Process chunk
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

# IMPROVE: Better error handling for Prophet
@safe_operation
def safe_prophet_fit(monthly_df, **kwargs):
    """Safely fit Prophet model with error handling"""
    try:
        m = Prophet(**kwargs)
        m.fit(monthly_df)
        return m
    except Exception as e:
        st.error(f"Prophet fitting failed: {e}")
        # Fallback to simple model
        return Prophet(growth='linear')

# ADD: Comprehensive data validation
def validate_prophet_data(monthly_df):
    """Validate data for Prophet forecasting"""
    issues = []
    
    if len(monthly_df) < 6:
        issues.append("Insufficient data points (< 6 months)")
    
    if monthly_df['y'].isnull().sum() > 0:
        issues.append("Missing values detected")
    
    if monthly_df['y'].std() == 0:
        issues.append("No variance in target variable")
    
    if monthly_df['y'].min() <= 0:
        issues.append("Non-positive values detected")
    
    return issues

# ISSUE: Mixed frequency usage
# FIX: Ensure consistent frequency

# CURRENT: Simple capacity calculation
default_cap = int(monthly_df['y'].max() * 1.1)

# IMPROVEMENT: More sophisticated capacity estimation
def estimate_capacity(monthly_df, method='percentile'):
    """Estimate capacity for logistic growth"""
    if method == 'percentile':
        return monthly_df['y'].quantile(0.95) * 1.1
    elif method == 'trend':
        # Extrapolate trend
        return monthly_df['y'].iloc[-1] * 1.2
    elif method == 'market':
        # Use market analysis
        return monthly_df['y'].max() * 1.15

# IMPROVE: Better event handling
def create_market_events():
    """Create market-specific events for Prophet"""
    events = pd.DataFrame({
        'holiday': ['Eid Al Fitr', 'Eid Al Adha', 'New Year'],
        'ds': ['2024-04-10', '2024-06-17', '2024-01-01'],
        'lower_window': [-7, -7, -14],
        'upper_window': [7, 7, 7]
    })
    return events

# Place the stub for tune_prophet_hyperparameters here, before any use
if 'tune_prophet_hyperparameters' not in globals():
    def tune_prophet_hyperparameters(monthly_df):
        return None, None

# Move this outside the expander so it's always defined
monthly_df = get_monthly_df(all_transactions, prophet_last_n_days)


