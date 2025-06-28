# --- Unified Data Refresh Function ---
def refresh_all_data():
    load_all_transactions.clear()
    load_all_listings.clear()
    # Layout map is not cached with @cache_data but can be handled if needed

import streamlit as st
import os
from streamlit import cache_data
import pandas as pd
import re
import math
# --- Required for Similarity Matching ---
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Prophet enhancements
from sklearn.metrics import mean_absolute_percentage_error

@st.cache_resource
def load_prophet_model(
    growth, n_changepoints, changepoint_range, changepoint_prior_scale,
    yearly_seasonality, weekly_seasonality, daily_seasonality,
    seasonality_prior_scale, interval_width, cap, events_df, monthly_df
):
    """Cache and return a fitted Prophet model."""
    m = Prophet(
        growth=growth,
        n_changepoints=n_changepoints,
        changepoint_range=changepoint_range,
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_prior_scale=seasonality_prior_scale,
        interval_width=interval_width,
        holidays=events_df if events_df is not None else None
    )
    if growth == "logistic" and cap is not None:
        monthly_df['cap'] = cap
    m.fit(monthly_df)
    return m


import streamlit.components.v1 as components
# AgGrid for interactive tables in tab2
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import numpy as np
import plotly.graph_objects as go


from sklearn.linear_model import LinearRegression
from prophet import Prophet
 
import numpy as np
from datetime import datetime, timedelta

def prepare_prophet_df(df):
    df2 = df.dropna(subset=['Evidence Date', 'Price (AED/sq ft)']).copy()
    df2['ds'] = pd.to_datetime(df2['Evidence Date'])
    df2['y'] = pd.to_numeric(df2['Price (AED/sq ft)'], errors='coerce')
    monthly = df2.set_index('ds')['y'].resample('M').mean().reset_index()
    return monthly



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

# --- Filter change callbacks ---
def _on_development_change():
    # Only clear unit_number if it no longer matches the selected development
    unit_number = st.session_state.get("unit_number", "")
    development = st.session_state.get("development", "")

    if unit_number and development:
        valid_units = all_transactions[
            all_transactions['All Developments'] == development
        ]['Unit No.'].unique()
        if unit_number not in valid_units:
            st.session_state.pop("unit_number", None)

    st.session_state.pop("community", None)
    st.session_state.pop("subcommunity", None)

def _on_community_change():
    # Only clear unit_number if it no longer matches the selected communities
    unit_number = st.session_state.get("unit_number", "")
    community = st.session_state.get("community", [])

    if unit_number and community:
        valid_units = all_transactions[
            all_transactions['Community/Building'].isin(community)
        ]['Unit No.'].unique()
        if unit_number not in valid_units:
            st.session_state.pop("unit_number", None)

    st.session_state.pop("subcommunity", None)

def _on_subcommunity_change():
    # Only clear unit_number if it no longer matches the selected subcommunities
    unit_number = st.session_state.get("unit_number", "")
    subcommunity = st.session_state.get("subcommunity", [])

    if unit_number and subcommunity:
        valid_units = all_transactions[
            all_transactions['Sub Community / Building'].isin(subcommunity)
        ]['Unit No.'].unique()
        if unit_number not in valid_units:
            st.session_state.pop("unit_number", None)
    else:
        st.session_state.pop("unit_number", None)

def _on_bedrooms_change():
    # Clear only the unit number selection when bedrooms changes
    st.session_state["unit_number"] = ""

# --- Page Config ---
st.set_page_config(page_title="Valuation App V2", layout="wide")

# --- Load Transaction Data from DATA/Transactions (cached) ---
transactions_dir = os.path.join(os.path.dirname(__file__), 'DATA', 'Transactions')
with st.spinner("Loading transaction files..."):
    all_transactions = load_all_transactions(transactions_dir)
if not all_transactions.empty:
    # Parse Evidence Date once for filtering
    if 'Evidence Date' in all_transactions.columns:
        all_transactions['Evidence Date'] = pd.to_datetime(all_transactions['Evidence Date'], errors='coerce')
    # Count unique source files actually loaded
    file_count = int(all_transactions['Source File'].nunique())
    st.success(f"Loaded {file_count} transaction file(s).")
else:
    st.warning("No transaction data loaded.")
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
layout_dir = os.path.join(os.path.dirname(__file__), 'DATA', 'Layout Types')
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
    layout_map_df = pd.DataFrame(columns=['Unit No.', 'Layout Type', 'Project'])
    layout_map = {}

# Add Layout Type to transactions
if not all_transactions.empty:
    # Clean unit numbers for strict, case-insensitive match
    all_transactions['Unit No.'] = all_transactions['Unit No.'].astype(str).str.strip().str.upper()
    layout_map_df['Unit No.'] = layout_map_df['Unit No.'].astype(str).str.strip().str.upper()
    layout_map = dict(zip(layout_map_df['Unit No.'], layout_map_df['Layout Type']))
    all_transactions['Layout Type'] = all_transactions['Unit No.'].map(layout_map).fillna('')
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

    # --- Filter Mode ---
    st.subheader("Filter Mode")
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

            prop_type_options = all_transactions['Unit Type'].dropna().unique().tolist()
            property_type = st.selectbox(
                "Property Type",
                options=[""] + sorted(prop_type_options),
                index=0,
                key="property_type"
            )

            bedrooms_options = all_transactions['Beds'].dropna().astype(str).unique().tolist()
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
        unit_row = units_df[units_df["Unit No."] == unit_number]
        if not unit_row.empty:
            selected_development = unit_row["Development"].values[0] if "Development" in unit_row.columns else unit_row["All Developments"].values[0] if "All Developments" in unit_row.columns else ""
            selected_community = unit_row["Community"].values[0] if "Community" in unit_row.columns else unit_row["Community/Building"].values[0] if "Community/Building" in unit_row.columns else ""
            sub_community = unit_row["Subcommunity"].values[0] if "Subcommunity" in unit_row.columns else unit_row["Sub Community / Building"].values[0] if "Sub Community / Building" in unit_row.columns else ""
            selected_sub_communities = [sub_community] if pd.notna(sub_community) else []
            layout = unit_row["Layout Type"].values[0] if "Layout Type" in unit_row.columns else ""
            layout_type_options = sorted(
                units_df[
                    (units_df["Development"].eq(selected_development) if "Development" in units_df.columns else units_df["All Developments"].eq(selected_development))
                    & (units_df["Community"].eq(selected_community) if "Community" in units_df.columns else units_df["Community/Building"].eq(selected_community))
                    & (units_df["Subcommunity"].isin(selected_sub_communities) if "Subcommunity" in units_df.columns else units_df["Sub Community / Building"].isin(selected_sub_communities))
                ]["Layout Type"].dropna().unique()
            )
            selected_layout_type = layout if layout in layout_type_options else None


    # --- Unit Number Filter ---
    st.subheader("Unit Number")
    if not all_transactions.empty:
        unit_df = all_transactions
        if 'development' in locals() and development:
            unit_df = unit_df[unit_df['All Developments'] == development]
        if 'community' in locals() and community:
            unit_df = unit_df[unit_df['Community/Building'].isin(community)]
        if 'subcommunity' in locals() and subcommunity:
            unit_df = unit_df[unit_df['Sub Community / Building'].isin(subcommunity)]
        unit_number_options = sorted(unit_df['Unit No.'].dropna().unique())
    else:
        unit_number_options = []
    current = st.session_state.get("unit_number", "")
    options = [""] + unit_number_options
    default_idx = options.index(current) if current in options else 0
    if filter_mode == "Unit Selection":
        unit_number = st.selectbox(
            "Unit Number",
            options=options,
            index=default_idx,
            key="unit_number",
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

    if community:
        filtered_unit_nos.update(
            all_transactions[all_transactions['Community/Building'].isin(community)]['Unit No.'].dropna().unique()
        )
    if subcommunity:
        filtered_unit_nos.update(
            all_transactions[all_transactions['Sub Community / Building'].isin(subcommunity)]['Unit No.'].dropna().unique()
        )

    if filtered_unit_nos:
        layout_df_filtered = layout_df_filtered[layout_df_filtered['Unit No.'].isin(filtered_unit_nos)]
    else:
        layout_df_filtered = layout_df_filtered[layout_df_filtered['Unit No.'].isin([])]  # no match, disable options

    # If the unit selection logic above found layout_type_options, use them
    if layout_type_options:
        layout_options = layout_type_options
    else:
        layout_options = sorted(layout_df_filtered['Layout Type'].dropna().unique())
    mapped_layout = layout_map.get(unit_number, "") if unit_number else ""
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
        dev_options = sorted(all_transactions['All Developments'].dropna().unique())
        com_options = sorted(all_transactions['Community/Building'].dropna().unique())
        subcom_options = sorted(all_transactions['Sub Community / Building'].dropna().unique())
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
        prop_type_options = sorted(all_transactions['Unit Type'].dropna().unique())
        bedrooms_options = sorted(all_transactions['Beds'].dropna().astype(str).unique())
        sales_rec_options = ['All'] + sorted(all_transactions['Sales Recurrence'].dropna().unique())
    else:
        prop_type_options = bedrooms_options = []
        sales_rec_options = ['All']

    # --- Determine selected values for autofill ---
    if filter_mode == "Unit Selection":
        if unit_number:
            unit_row = all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]

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
        subcom_options = sorted(
            all_transactions[all_transactions['Community/Building'].isin(community)]['Sub Community / Building'].dropna().unique()
        )
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

        elif time_filter_mode == "From Date to Date" and all(date_range):
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
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
        filtered_transactions = filtered_transactions[filtered_transactions['Community/Building'].isin(community)]
    if subcommunity:
        filtered_transactions = filtered_transactions[filtered_transactions['Sub Community / Building'].isin(subcommunity)]
    if property_type:
        filtered_transactions = filtered_transactions[filtered_transactions['Unit Type'] == property_type]
    if bedrooms:
        filtered_transactions = filtered_transactions[filtered_transactions['Beds'].astype(str) == bedrooms]
    # Floor tolerance filter for apartments if enabled
    if property_type == "Apartment" and unit_number and st.session_state.get("enable_floor_tol", False):
        tol = st.session_state.get("floor_tolerance", 0)
        try:
            selected_floor = int(
                all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Floor Level']
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
        filtered_transactions = filtered_transactions[filtered_transactions['Layout Type'].isin(layout_type)]
    # BUA tolerance filter if enabled
    if property_type == "Apartment" and unit_number and st.session_state.get("enable_bua_tol", False):
        tol = st.session_state.get("bua_tolerance", 0)
        if tol > 0:
            try:
                selected_bua = float(
                    all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Unit Size (sq ft)']
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
                    all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Plot Size (sq ft)']
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
        filtered_transactions = filtered_transactions[filtered_transactions['Community/Building'].isin(community)]
    if subcommunity:
        filtered_transactions = filtered_transactions[filtered_transactions['Sub Community / Building'].isin(subcommunity)]
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
        filtered_transactions = filtered_transactions[filtered_transactions['Layout Type'].isin(layout_type)]
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

 # --- Load Live Listings Data from DATA/Listings ---
listings_dir = os.path.join(os.path.dirname(__file__), 'DATA', 'Listings')
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
    st.title("Real Estate Valuation Dashboard")
    st.write("Main dashboard area will be developed here.")

    # Only show Selected Unit Info (left column) and remove valuation summary (right column)
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
    if info_parts:
        st.markdown(" | ".join(info_parts))

    # Try to use selected_unit_data if available for info, else fallback to session state
    if unit_number:
        selected_unit_data = all_transactions[all_transactions['Unit No.'] == unit_number].copy()
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

    st.subheader("Transaction History")
    if isinstance(filtered_transactions, pd.DataFrame) and filtered_transactions.shape[0] > 0:
        columns_to_hide = ["Select Data Points", "Maid", "Study", "Balcony", "Developer Name", "Source", "Comments", "Source File", "View"]
        visible_columns = [col for col in filtered_transactions.columns if col not in columns_to_hide]
        # Hide Sub Community column if no values present
        if 'Sub Community / Building' in filtered_transactions.columns and filtered_transactions['Sub Community / Building'].dropna().empty:
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
        if community and 'Community' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Community'].isin(community)]
        if subcommunity and 'Subcommunity' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Subcommunity'].isin(subcommunity)]
        if property_type and 'Type' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Type'] == property_type]
        if bedrooms and 'Beds' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Beds'].astype(str) == bedrooms]
        # Floor tolerance filter for live listings (optional, not specified in instructions)
        if property_type == "Apartment" and unit_number and 'floor_tolerance' in st.session_state and 'Floor Level' in filtered_listings.columns:
            try:
                selected_floor = int(
                    all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Floor Level']
                )
                tol = st.session_state['floor_tolerance']
                low = selected_floor - tol
                high = selected_floor + tol
                filtered_listings = filtered_listings[
                    (filtered_listings['Floor Level'] >= low) &
                    (filtered_listings['Floor Level'] <= high)
                ]
            except Exception:
                pass
        if layout_type and 'Layout Type' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Layout Type'].isin(layout_type)]
        if sales_recurrence != "All" and 'Sales Recurrence' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Sales Recurrence'] == sales_recurrence]

        # Exclude listings marked as not available
        if 'Availability' in filtered_listings.columns:
            filtered_listings = filtered_listings[filtered_listings['Availability'].astype(str).str.strip().str.lower() != "not available"]

        # Hide certain columns but keep them in the DataFrame (do NOT filter by Days Listed or any date here)
        columns_to_hide = ["Reference Number", "URL", "Source File", "Unit No.", "Unit Number", "Listed When", "Listed when"]
        visible_columns = [c for c in filtered_listings.columns if c not in columns_to_hide] + ["URL"]

        # Show count of live listings
        st.markdown(f"**Showing {filtered_listings.shape[0]} live listings**")

        # Use AgGrid for clickable selection
        gb = GridOptionsBuilder.from_dataframe(filtered_listings[visible_columns])
        gb.configure_selection('single', use_checkbox=False, rowMultiSelectWithClick=False)
        grid_options = gb.build()

        grid_response = AgGrid(
            filtered_listings[visible_columns],
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
    # --- Selected Unit Info ---
    info_parts = []
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

    # --- Most Similar Past Deals Section ---
    with st.expander("Most Similar Past Deals", expanded=False):
        if unit_number:
            # Prepare feature DataFrame from all_transactions
            sim_df = all_transactions.copy()
            # Define feature columns
            numeric_features = ["Unit Size (sq ft)", "Plot Size (sq ft)", "Beds", "Floor Level"]
            categorical_features = []
            # Identify available categorical columns
            if "Layout Type" in sim_df.columns:
                categorical_features.append("Layout Type")
            if "All Developments" in sim_df.columns:
                categorical_features.append("All Developments")
            if "Community/Building" in sim_df.columns:
                categorical_features.append("Community/Building")
            if "Sub Community / Building" in sim_df.columns:
                categorical_features.append("Sub Community / Building")
            # Coerce numeric feature columns to floats, converting invalid entries (e.g., "-") to NaN
            for col in numeric_features:
                if col in sim_df.columns:
                    sim_df[col] = pd.to_numeric(sim_df[col], errors='coerce')
            # Drop rows where any numeric feature is missing or invalid
            sim_df = sim_df.dropna(subset=[col for col in numeric_features if col in sim_df.columns])
            # For categorical, fill missing with a placeholder
            for col in categorical_features:
                sim_df[col] = sim_df[col].fillna("UNKNOWN")
            # Build preprocessing pipeline
            transformers = []
            if numeric_features:
                transformers.append(("num", StandardScaler(), numeric_features))
            if categorical_features:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))
            preprocessor = ColumnTransformer(transformers)
            # Fit transformer on sim_df
            X = preprocessor.fit_transform(sim_df)
            # Fit KNN model (include one extra neighbor to exclude the unit itself)
            knn = NearestNeighbors(n_neighbors=6, metric="euclidean")
            knn.fit(X)
            # Prepare target vector
            target_row = sim_df[sim_df["Unit No."] == unit_number]
            if target_row.empty:
                st.warning("Selected unit not found in historical transactions for similarity matching.")
            else:
                # Preprocess target
                X_target = preprocessor.transform(target_row)
                distances, indices = knn.kneighbors(X_target)
                # Exclude the first neighbor if it's the unit itself
                idxs = indices[0].tolist()
                # Remove same-unit match if present
                idxs = [i for i in idxs if sim_df.iloc[i]["Unit No."] != unit_number]
                # Take top 5 similar
                top_idxs = idxs[:5]
                similar_df = sim_df.iloc[top_idxs].copy()
                # Optionally, convert price column to numeric for display if needed
                if "Price (AED/sq ft)" in sim_df.columns:
                    similar_df["Price (AED/sq ft)"] = pd.to_numeric(similar_df["Price (AED/sq ft)"], errors='coerce')
                # Display key columns in a table
                display_cols = ["Evidence Date", "Price (AED/sq ft)", "Unit Size (sq ft)", "Plot Size (sq ft)", "Beds", "Floor Level"]
                # Add categorical columns if available
                for col in categorical_features:
                    display_cols.append(col)
                # Ensure columns exist
                display_cols = [col for col in display_cols if col in similar_df.columns]
                # Format date column
                if "Evidence Date" in display_cols:
                    similar_df["Evidence Date"] = pd.to_datetime(similar_df["Evidence Date"]).dt.strftime("%Y-%m-%d")
                st.markdown("Top 5 most similar past transactions:")
                st.table(similar_df[display_cols])
        else:
            st.info("Select a unit to see similar past deals.")
    with st.expander("Prophet Time Frame", expanded=False):
        prophet_last_n_days = st.number_input(
            "Prophet: Last N Days",
            min_value=30, max_value=3650, step=30, value=1095, key="prophet_last_n_days"
        )
        # Apply local time filter for Prophet
        prophet_df = all_transactions.copy()
        if 'Evidence Date' in prophet_df.columns:
            prophet_df['Evidence Date'] = pd.to_datetime(prophet_df['Evidence Date'], errors='coerce')
            cutoff = datetime.today() - timedelta(days=prophet_last_n_days)
            prophet_df = prophet_df[prophet_df['Evidence Date'] >= cutoff]
    # Prepare data for Prophet
    monthly_df = prepare_prophet_df(prophet_df)
    # Outlier trimming
    trim_pct = st.checkbox("Trim top/bottom 5% of data", value=True,
                           help="Remove extreme monthly values before fitting.")
    if trim_pct:
        low, high = monthly_df['y'].quantile(0.05), monthly_df['y'].quantile(0.95)
        monthly_df = monthly_df[(monthly_df['y'] >= low) & (monthly_df['y'] <= high)]
    # Historical accuracy badge
    if len(monthly_df) >= 12:
        holdout = monthly_df.iloc[-6:]
        train = monthly_df.iloc[:-6]
        prophet_args = dict(
            growth='linear',  # will be replaced below with actual args
            n_changepoints=25,
            changepoint_range=0.8,
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_prior_scale=10.0,
            interval_width=0.8
        )
        # Try to get from session state if available
        try:
            prophet_args['growth'] = growth
            prophet_args['n_changepoints'] = n_changepoints
            prophet_args['changepoint_range'] = changepoint_range
            prophet_args['changepoint_prior_scale'] = changepoint_prior_scale
            prophet_args['yearly_seasonality'] = yearly_seasonality
            prophet_args['weekly_seasonality'] = weekly_seasonality
            prophet_args['daily_seasonality'] = daily_seasonality
            prophet_args['seasonality_prior_scale'] = seasonality_prior_scale
            prophet_args['interval_width'] = interval_width
        except Exception:
            pass
        m_temp = Prophet(**prophet_args)
        m_temp.fit(train)
        f = m_temp.predict(holdout[['ds']])
        mape = mean_absolute_percentage_error(holdout['y'], f['yhat']) * 100
        st.markdown(f"**Out‑of‑Sample MAPE (last 6 months): {mape:.1f}%**")
    # Events annotation
    events_file = st.file_uploader("Upload Events CSV", type=["csv"], key="events_csv")
    if events_file:
        events_df = pd.read_csv(events_file, parse_dates=['ds'])
    else:
        events_df = None

    # ─── Live Listings Integration Controls ───
    st.subheader("Live Listings Filters")
    verified_only = st.checkbox(
        "Use Only Verified Listings",
        value=True,
        help="Filter to listings marked as verified."
    )
    listings_days = st.number_input(
        "Recent Listings: Last N Days",
        min_value=1, max_value=365, step=1, value=30,
        help="Include only listings not older than N days."
    )

    # Prepare filtered listing DataFrame
    listing_df = all_listings.copy() if 'all_listings' in locals() else pd.DataFrame()
    if verified_only and 'Verified' in listing_df.columns:
        listing_df = listing_df[listing_df['Verified'].str.lower() == 'yes']
    if 'Days Listed' in listing_df.columns:
        listing_df['Listing Date'] = datetime.today() - pd.to_timedelta(listing_df['Days Listed'], unit='D')
        cutoff_listings = datetime.today() - timedelta(days=listings_days)
        listing_df = listing_df[listing_df['Listing Date'] >= cutoff_listings]

    # Sync listings context to selected unit
    mask = pd.Series(True, index=listing_df.index)
    # Development match
    if 'Development' in listing_df.columns and development:
        mask &= listing_df['Development'] == development
    # Community match
    if 'Community' in listing_df.columns and community:
        comm_list = community if isinstance(community, list) else [community]
        mask &= listing_df['Community'].isin(comm_list)
    # Subcommunity match
    subcol = 'Subcommunity' if 'Subcommunity' in listing_df.columns else 'Sub Community / Building'
    if subcol in listing_df.columns and subcommunity:
        if isinstance(subcommunity, (list, tuple)):
            mask &= listing_df[subcol].isin(subcommunity)
        else:
            mask &= listing_df[subcol] == subcommunity
    listing_df = listing_df[mask]
    if layout_type:
        listing_df = listing_df[listing_df['Layout Type'].isin(layout_type)]

    # Compute median listing price per sqft and AED
    if not listing_df.empty and 'Price (AED)' in listing_df.columns and 'BUA' in listing_df.columns:
        listing_df['Price_per_sqft'] = listing_df['Price (AED)'] / listing_df['BUA']
        live_med_sqft = listing_df['Price_per_sqft'].median()
        median_price = live_med_sqft * unit_bua if unit_bua else None
    else:
        median_price = None

    # Display total number of live listings
    total_listings = listing_df.shape[0]
    st.markdown(f"**Total Live Listings:** {total_listings}")
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
    if len(monthly_df) < 6:
        st.info(f"Need at least 6 months of data for forecasting—got {len(monthly_df)}.")
    elif len(monthly_df) < 12:
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
        st.table(display_df)
    else:
        # Full-history Prophet forecast
        # Get or fit cached Prophet model
        m = load_prophet_model(
            growth, n_changepoints, changepoint_range, changepoint_prior_scale,
            yearly_seasonality, weekly_seasonality, daily_seasonality,
            seasonality_prior_scale, interval_width, cap, events_df, monthly_df
        )
        future = m.make_future_dataframe(periods=6, freq='M')
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
                x=ver_df['Listing Date'],
                y=ver_df['Price (AED)'],
                mode='markers',
                marker=dict(symbol='diamond', size=8, opacity=0.8, color='green'),
                name='Verified Listings',
                customdata=ver_df['URL'],
                hovertemplate="Date: %{x|%b %Y}<br>Price: AED %{y:,.0f}<extra></extra>"
            ))
        if not nonver_df.empty:
            fig.add_trace(go.Scatter(
                x=nonver_df['Listing Date'],
                y=nonver_df['Price (AED)'],
                mode='markers',
                marker=dict(symbol='diamond', size=8, opacity=0.8, color='red'),
                name='Non-verified Listings',
                customdata=nonver_df['URL'],
                hovertemplate="Date: %{x|%b %Y}<br>Price: AED %{y:,.0f}<extra></extra>"
            ))
        fig.update_layout(
            title='Prophet Forecast (Actual Value)',
            xaxis_title='Month', yaxis_title='AED'
        )
        # Render interactive Plotly chart with clickable listings
        html_str = fig.to_html(include_plotlyjs='cdn')
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
        with st.expander("Changepoints & Slope Changes", expanded=False):
            cp_dates = m.changepoints
            cp_delta = m.params['delta'].flatten()
            cp_df = pd.DataFrame({"Date": cp_dates, "Slope Change": cp_delta})
            cp_df['Date'] = cp_df['Date'].dt.strftime('%B %Y')
            st.table(cp_df)
        # Table
        fc = forecast[['ds','yhat_actual','yhat_lower_actual','yhat_upper_actual']].tail(6).copy()
        fc['Month'] = fc['ds'].dt.strftime('%B %Y')
        display_df = fc[['Month','yhat_actual','yhat_lower_actual','yhat_upper_actual']].rename(columns={
            'yhat_actual':'Forecast Value','yhat_lower_actual':'Lower CI','yhat_upper_actual':'Upper CI'
        })
        st.table(display_df)
        # Download data as CSV only
        st.download_button(
            "Download Data as CSV",
            display_df.to_csv(index=False).encode(),
            file_name="forecast.csv"
        )




with tab4:
    st.write("Placeholder for other functionality.")


