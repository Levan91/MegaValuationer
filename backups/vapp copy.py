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


import streamlit.components.v1 as components
# AgGrid for interactive tables in tab2
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

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
    # --- Compose selected unit info for display above header ---
    # Try to get selected unit fields for display, including development and community
    selected_unit_number = unit_number if unit_number else ""
    if unit_number and not all_transactions.empty:
        try:
            unit_row = all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]
            selected_development = unit_row['All Developments'] if 'All Developments' in unit_row else ""
            selected_community = unit_row['Community/Building'] if 'Community/Building' in unit_row else ""
            selected_subcommunity = unit_row['Sub Community / Building'] if 'Sub Community / Building' in unit_row else ""
            selected_layout_type = unit_row['Layout Type'] if 'Layout Type' in unit_row else ""
            selected_beds = str(unit_row['Beds']) if 'Beds' in unit_row and pd.notna(unit_row['Beds']) else ""
            selected_bua = unit_row['Unit Size (sq ft)'] if 'Unit Size (sq ft)' in unit_row and pd.notna(unit_row['Unit Size (sq ft)']) else ""
            selected_plot = unit_row['Plot Size (sq ft)'] if 'Plot Size (sq ft)' in unit_row and pd.notna(unit_row['Plot Size (sq ft)']) else ""
            selected_floor = unit_row['Floor Level'] if 'Floor Level' in unit_row and pd.notna(unit_row['Floor Level']) else ""
        except Exception:
            selected_development = development
            # If community is a list, join or take first
            if isinstance(community, list):
                selected_community = community[0] if len(community) > 0 else ""
            else:
                selected_community = community if community else ""
            selected_subcommunity = subcommunity[0] if isinstance(subcommunity, list) and len(subcommunity) > 0 else (subcommunity if subcommunity else "")
            selected_layout_type = layout_type[0] if isinstance(layout_type, list) and len(layout_type) > 0 else ""
            selected_beds = bedrooms
            selected_bua = bua
            selected_plot = plot_size
            selected_floor = ""
    else:
        selected_development = development
        if isinstance(community, list):
            selected_community = community[0] if len(community) > 0 else ""
        else:
            selected_community = community if community else ""
        selected_subcommunity = subcommunity[0] if isinstance(subcommunity, list) and len(subcommunity) > 0 else (subcommunity if subcommunity else "")
        selected_layout_type = layout_type[0] if isinstance(layout_type, list) and len(layout_type) > 0 else ""
        selected_beds = bedrooms
        selected_bua = bua
        selected_plot = plot_size
        selected_floor = ""

    st.markdown(
        f"📦 {selected_unit_number} | 🏙️ {selected_development}, {selected_community}, {selected_subcommunity} | 🧱 {selected_layout_type} | 🛏️ {selected_beds} Beds | 📐 {selected_bua} sqft | 🌳 Plot {selected_plot} sqft | 🏢 {selected_floor}",
        unsafe_allow_html=True
    )
    st.header("Trend & Valuation")
    # --- Transaction-Based Valuation Summary ---
    with st.expander("🔍 Valuation Analysis Summary", expanded=True):
        # Prepare transaction comparables
        tx = filtered_transactions.copy()
        if 'Evidence Date' in tx.columns:
            tx['Evidence Date'] = pd.to_datetime(tx['Evidence Date'], dayfirst=True, errors='coerce')
        tx = tx.sort_values('Evidence Date', ascending=False)

        # Initialize valuation_metrics for this tab scope
        valuation_metrics = {}

        if isinstance(tx, pd.DataFrame) and tx.shape[0] > 0:
            # Use up to the last 5 transactions (but at least 1 if available)
            n_comps = min(5, tx.shape[0])
            comps = tx.head(n_comps)
            # Ensure comparables price-per-sqft is numeric
            comps['Price (AED/sq ft)'] = pd.to_numeric(comps['Price (AED/sq ft)'], errors='coerce')

            # Average price per sqft (raw)
            avg_ppsqft = comps['Price (AED/sq ft)'].mean()
            st.markdown(f"- **Last {n_comps} Transactions Avg (AED/sq ft):** {avg_ppsqft:.2f}")

            # Determine BUA of selected unit
            unit_bua = None
            if unit_number:
                row = all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]
                unit_bua = row.get('Unit Size (sq ft)', None)
            elif bua:
                try:
                    unit_bua = float(bua)
                except:
                    unit_bua = None

            # Total value based on resale avg
            if unit_bua:
                total_val = avg_ppsqft * unit_bua
                st.markdown(f"- **Total Value (Resale Avg × BUA):** {total_val:,.0f} AED")
            else:
                total_val = None

            # --- Growth Rate Calculation (Trendline-based, Linear Regression) ---
            import numpy as np
            from sklearn.linear_model import LinearRegression
            def calculate_trendline_growth_rate(df, price_col='Price (AED/sq ft)', date_col='Evidence Date'):
                import pandas as pd
                df = df.dropna(subset=[price_col, date_col]).copy()
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                if df.shape[0] < 2:
                    return None, None
                # Compute number of full months between each date and the earliest date
                df['Months'] = (df[date_col].dt.year - df[date_col].min().year) * 12 + \
                               (df[date_col].dt.month - df[date_col].min().month)
                df['Months'] = df['Months'].astype(float)
                X = df[['Months']]
                y = df[price_col]
                model = LinearRegression().fit(X, y)
                monthly_growth = model.coef_[0] / y.mean() if y.mean() != 0 else 0
                yearly_growth = (1 + monthly_growth)**12 - 1
                return monthly_growth * 100, yearly_growth * 100

            monthly_growth, yearly_growth = calculate_trendline_growth_rate(filtered_transactions)
            def fmt_growth2(val):
                if val is None:
                    return "N/A"
                sign = "+" if val > 0 else ""
                return f"{sign}{val:.2f}%"
            if monthly_growth is not None and yearly_growth is not None:
                st.markdown(f"- 📉 Growth Rate: {fmt_growth2(yearly_growth)} yearly | {fmt_growth2(monthly_growth)} monthly")
                trend_label = (
                    "Rising" if yearly_growth > 0 else
                    "Falling" if yearly_growth < 0 else
                    "Flat"
                )
            else:
                st.markdown("- 📉 Growth Rate: N/A (not enough data for trendline)")
                trend_label = "N/A"

            # Prepare initial valuation_metrics dictionary (to be updated later)
            valuation_metrics = {
                "last_3_avg": f"{avg_ppsqft:.0f}",
                "total_value": f"{total_val:,.0f}" if unit_bua else "N/A",
                "trend_direction": trend_label,
                # The following will be updated in later logic if available
                "forecasted_price": "N/A",
                "valuation_range": "N/A",
                # Add raw values for later use
                "ForecastedPrice": avg_ppsqft if avg_ppsqft is not None else 0,
                "Last3Avg": avg_ppsqft if avg_ppsqft is not None else 0,
            }
        else:
            st.info("No transaction data available.")
            # Ensure valuation_metrics is at least empty
            valuation_metrics = {}



    # --- Price Trend & Forecast Chart ---
    tx_chart = filtered_transactions.copy()
    # Determine BUA for actual price calculation
    unit_bua = None
    if unit_number:
        try:
            unit_bua = float(all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Unit Size (sq ft)'])
        except:
            unit_bua = None
    elif bua:
        try:
            unit_bua = float(bua)
        except:
            unit_bua = None

    monthly_growth_series = None  # For monthly growth chart

    if 'Evidence Date' in tx_chart.columns:
        tx_chart['Evidence Date'] = pd.to_datetime(tx_chart['Evidence Date'], errors='coerce')
        # Ensure price per sqft is numeric
        tx_chart['Price (AED/sq ft)'] = pd.to_numeric(tx_chart['Price (AED/sq ft)'], errors='coerce')
        # Monthly aggregates including transaction count
        df_monthly = (
            tx_chart.set_index('Evidence Date')
                      .resample('M')['Price (AED/sq ft)']
                      .agg(['mean', 'median', 'count'])
                      .rename(columns={'mean': 'Average', 'median': 'Median', 'count': 'Txns'})
        )
        # 3-month rolling average
        df_monthly['3M Rolling Avg'] = df_monthly['Average'].rolling(3).mean()

        # Compute monthly growth rate (as % change)
        if df_monthly.shape[0] >= 2:
            growth_pct = df_monthly['Average'].pct_change() * 100
            # Prepare months as string (e.g., Jan, Feb, ...)
            months = df_monthly.index.strftime('%b %Y')
            monthly_growth_series = {
                "Month": months[1:],  # skip first NaN growth
                "GrowthRate": growth_pct.iloc[1:].values
            }

        # Convert per-sqft stats to actual price
        if unit_bua:
            df_monthly['Average Price'] = df_monthly['Average'] * unit_bua
            df_monthly['Median Price'] = df_monthly['Median'] * unit_bua
            df_monthly['3M Rolling Price'] = df_monthly['3M Rolling Avg'] * unit_bua
        else:
            st.info("BUA required to calculate actual prices.")
            st.stop()

        if df_monthly.shape[0] >= 3:
            # --- Forecast via Linear Regression on Actual Price ---
            # Reset index and drop months without actual price
            df_reg = df_monthly.reset_index()
            df_reg = df_reg.dropna(subset=['Average Price']).reset_index(drop=True)
            # Recompute month index
            df_reg['MonthIndex'] = np.arange(df_reg.shape[0])
            X = df_reg[['MonthIndex']]
            y = df_reg['Average Price']
            # Fit regression
            model = LinearRegression().fit(X, y)
            # Prepare future indices and dates
            last_idx = df_reg['MonthIndex'].iloc[-1]
            future_indices = np.arange(last_idx + 1, last_idx + 7)
            last_date = df_reg['Evidence Date'].iloc[-1]
            future_months = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=6, freq='M')
            # Predict
            forecast_values = model.predict(future_indices.reshape(-1, 1))
            forecast_df = pd.DataFrame({
                'Forecast Month': future_months,
                'Forecast Price (AED)': [round(v, 0) for v in forecast_values]
            })

            # --- Two-Way Pricing Comparison & Suggested Range (after forecast computed) ---
            if isinstance(tx, pd.DataFrame) and tx.shape[0] > 0 and unit_bua and 'forecast_df' in locals() and not forecast_df.empty:
                # Recent transactions for resale avg
                n_comps = min(5, tx.shape[0])
                comps = tx.head(n_comps)
                avg_ppsqft = comps['Price (AED/sq ft)'].mean()
                st.markdown(f"- **Last {n_comps} Transactions Avg:** {avg_ppsqft:.0f} AED/sqft")
                conservative_ppsqft = avg_ppsqft
                conservative_total = conservative_ppsqft * unit_bua
                # Forecasted total for current month
                high_total = forecast_df['Forecast Price (AED)'].iloc[0]
                high_ppsqft = high_total / unit_bua
                suggested_ppsqft = high_ppsqft
                # Display
                st.markdown("### 💸 Pricing Comparison")
                st.markdown(f"- **Conservative Price (Resale Avg):** {conservative_ppsqft:.0f} AED/sqft → Total: {conservative_total:,.0f} AED")
                st.markdown(f"- **Forecasted Price (Current Month):** {high_ppsqft:.0f} AED/sqft → Total: {high_total:,.0f} AED")

                # --- Live Listings Median Suggested Price ---
                live = all_listings.copy() if 'all_listings' in globals() else pd.DataFrame()
                # Apply sidebar filters to live listings
                if not live.empty:
                    if development and 'Development' in live.columns:
                        live = live[live['Development'] == development]
                    if community and 'Community' in live.columns:
                        live = live[live['Community'].isin(community)]
                    if subcommunity and 'Subcommunity' in live.columns:
                        live = live[live['Subcommunity'].isin(subcommunity)]
                    if property_type and 'Unit Type' in live.columns:
                        live = live[live['Unit Type'] == property_type]
                    if bedrooms and 'Beds' in live.columns:
                        live = live[live['Beds'].astype(str) == bedrooms]
                    # Floor tolerance filter for live listings in Trend & Valuation
                    if property_type == "Apartment" and unit_number and 'floor_tolerance' in st.session_state and 'Floor Level' in live.columns:
                        try:
                            selected_floor = int(
                                all_transactions[all_transactions['Unit No.'] == unit_number].iloc[0]['Floor Level']
                            )
                            tol = st.session_state['floor_tolerance']
                            low = selected_floor - tol
                            high = selected_floor + tol
                            live = live[
                                (live['Floor Level'] >= low) &
                                (live['Floor Level'] <= high)
                            ]
                        except Exception:
                            pass
                    if layout_type and 'Layout Type' in live.columns:
                        live = live[live['Layout Type'].isin(layout_type)]
                    if sales_recurrence != "All" and 'Sales Recurrence' in live.columns:
                        live = live[live['Sales Recurrence'] == sales_recurrence]
                    # Exclude listings marked as not available
                    if 'Availability' in live.columns:
                        live = live[live['Availability'].astype(str).str.strip().str.lower() != "not available"]
                # Filter verified and recent via Days Listed
                if 'Verified' in live.columns:
                    live = live[live['Verified'].astype(str).str.strip().str.lower() == "yes"]
                if 'Days Listed' in live.columns:
                    live['Days Listed'] = pd.to_numeric(live['Days Listed'], errors='coerce')
                    live = live[live['Days Listed'].notna() & (live['Days Listed'] <= 30)]
                # Compute price-per-sqft using listing BUA
                live_count = 0
                if not live.empty:
                    # Ensure numeric price and BUA
                    live['Price (AED)'] = pd.to_numeric(live['Price (AED)'], errors='coerce')
                    live['BUA'] = pd.to_numeric(live['BUA'], errors='coerce')
                    # Drop rows missing price or BUA
                    live = live.dropna(subset=['Price (AED)', 'BUA'])
                    # Ignore listings with BUA outside ±20% of the selected unit's BUA
                    if unit_bua:
                        try:
                            selected_bua = float(unit_bua)
                            lower = selected_bua * 0.8
                            upper = selected_bua * 1.2
                            before_count = live.shape[0]
                            live = live[(live['BUA'] >= lower) & (live['BUA'] <= upper)]
                            dropped = before_count - live.shape[0]
                            if dropped > 0:
                                st.markdown(f"_Ignored {dropped} listing(s) with BUA outside ±20% range of selected unit._")
                        except:
                            pass
                    if not live.empty:
                        live['Price_per_sqft'] = live['Price (AED)'] / live['BUA']
                        live = live.dropna(subset=['Price_per_sqft'])
                        live_count = live.shape[0]
                if live_count >= 3:
                    # Remove outliers based on IQR on price_per_sqft
                    # Save pre-IQR DataFrame
                    live_before_iqr = live.copy()
                    # Compute Q1 and Q3
                    q1 = live['Price_per_sqft'].quantile(0.25)
                    q3 = live['Price_per_sqft'].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    # Create IQR mask
                    iqr_mask = (live_before_iqr['Price_per_sqft'] >= lower_bound) & (live_before_iqr['Price_per_sqft'] <= upper_bound)
                    before_iqr_count = live.shape[0]
                    live = live_before_iqr[iqr_mask]
                    after_iqr_count = live.shape[0]
                    dropped = before_iqr_count - after_iqr_count
                    if dropped > 0:
                        st.markdown(f"_Removed {dropped} outlier listing(s) based on price-per-sqft IQR before averaging._")
                    # Recompute live_count after outlier removal
                    live_count = live.shape[0]
                    if live_count >= 1:
                        live_avg_ppsqft = live['Price_per_sqft'].median()
                        live_avg_total = live_avg_ppsqft * unit_bua
                        st.markdown(f"- **Live Listings Average (IQR-cleaned):** {live_avg_ppsqft:.0f} AED/sqft (based on {live_count} listings) → Total: {live_avg_total:,.0f} AED")
                        st.markdown("_This average excludes listings with extreme price-per-sqft values based on IQR filtering._")

                        # --- Suggested Listing Price Range ---
                        # Use conservative_ppsqft, live_avg_ppsqft, and unit_bua for range
                        if unit_bua:
                            conservative_price_per_sqft = conservative_ppsqft  # already defined above
                            live_price_per_sqft = live_avg_ppsqft  # just calculated
                            total_min = round(conservative_price_per_sqft * unit_bua)
                            total_max = round(live_price_per_sqft * unit_bua)
                            st.markdown("#### 💡 Suggested Listing Price Range")
                            st.markdown(f"- **AED/sqft:** {conservative_price_per_sqft:.0f} – {live_price_per_sqft:.0f}")
                            st.markdown(f"- **Total AED:** {total_min:,} – {total_max:,}")
                        # Display included listings in an expander, hiding unwanted columns, with AgGrid
                        with st.expander("Listings included in IQR-cleaned average (Trend & Valuation)", expanded=False):
                            df_inc = live.reset_index(drop=True)
                            if not df_inc.empty:
                                gb_inc = GridOptionsBuilder.from_dataframe(df_inc)
                                # Configure selection
                                gb_inc.configure_selection('single', use_checkbox=False, rowMultiSelectWithClick=False)
                                # Hide unwanted columns
                                for col in ["Reference Number", "URL", "Unit No.", "Unit Number", "Listed When", "Source File"]:
                                    if col in df_inc.columns:
                                        gb_inc.configure_column(col, hide=True)
                                grid_options_inc = gb_inc.build()
                                grid_response_inc = AgGrid(
                                    df_inc,
                                    gridOptions=grid_options_inc,
                                    enable_enterprise_modules=False,
                                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                                    theme='alpine'
                                )
                                selected_inc = grid_response_inc['selected_rows']
                                selected_url = None
                                if isinstance(selected_inc, list) and selected_inc:
                                    selected_url = selected_inc[0].get("URL")
                                elif isinstance(selected_inc, pd.DataFrame) and not selected_inc.empty and "URL" in selected_inc.columns:
                                    selected_url = selected_inc.iloc[0]["URL"]
                                # Removed setting preview URL in session state for included
                            else:
                                st.write("No listings to display.")
                        # Display excluded listings in an expander, hiding unwanted columns, with AgGrid
                        with st.expander("Listings excluded as outliers (Trend & Valuation)", expanded=False):
                            df_ex = live_before_iqr[~iqr_mask].reset_index(drop=True)
                            if not df_ex.empty:
                                gb_ex = GridOptionsBuilder.from_dataframe(df_ex)
                                # Configure selection
                                gb_ex.configure_selection('single', use_checkbox=False, rowMultiSelectWithClick=False)
                                # Hide unwanted columns
                                for col in ["Reference Number", "URL", "Unit No.", "Unit Number", "Listed When", "Source File"]:
                                    if col in df_ex.columns:
                                        gb_ex.configure_column(col, hide=True)
                                grid_options_ex = gb_ex.build()
                                grid_response_ex = AgGrid(
                                    df_ex,
                                    gridOptions=grid_options_ex,
                                    enable_enterprise_modules=False,
                                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                                    theme='alpine'
                                )
                                selected_ex = grid_response_ex['selected_rows']
                                selected_url_ex = None
                                if isinstance(selected_ex, list) and selected_ex:
                                    selected_url_ex = selected_ex[0].get("URL")
                                elif isinstance(selected_ex, pd.DataFrame) and not selected_ex.empty and "URL" in selected_ex.columns:
                                    selected_url_ex = selected_ex.iloc[0]["URL"]
                                # Removed setting preview URL in session state for excluded
                            else:
                                st.write("No excluded listings to display.")
                    else:
                        st.markdown(f"- **Live Listings Average:** No valid listings remain after outlier removal; skipping average suggestion")
                else:
                    st.markdown(f"- **Live Listings Average:** Not enough valid listings (found {live_count}); skipping average suggestion")

                st.markdown("### 💸 Suggested Valuation Range")
                st.markdown(f"- **{conservative_ppsqft:.0f} – {high_ppsqft:.0f} AED/sqft**")
                st.markdown(f"- **Total Value Range:** {conservative_total:,.0f} – {high_total:,.0f} AED")
                # --- Update valuation_metrics in session_state with forecast and range ---
                # Update valuation_metrics with forecast and price range (new keys)
                valuation_metrics.update({
                    "forecasted_price": f"{high_ppsqft:.0f}",
                    "valuation_range": f"{conservative_ppsqft:.0f} – {high_ppsqft:.0f} AED/sqft"
                })
            else:
                st.info("Insufficient data for pricing comparison or forecast.")

            # Build Plotly figure using actual prices, include Txns in hover
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_monthly.index, y=df_monthly['Average Price'],
                mode='lines+markers', name='Average Price',
                line=dict(color='royalblue'),
                customdata=df_monthly['Txns'],
                hovertemplate='Average Price: %{y:.0f} AED<br>Txns: %{customdata}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=df_monthly.index, y=df_monthly['Median Price'],
                mode='lines', name='Median Price',
                line=dict(color='green', dash='dot'),
                hovertemplate='Median Price: %{y:.0f} AED<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=df_monthly.index, y=df_monthly['3M Rolling Price'],
                mode='lines', name='3M Rolling Price',
                line=dict(color='orange', dash='dash'),
                hovertemplate='3M Rolling Price: %{y:.0f} AED<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Forecast Month'], y=forecast_df['Forecast Price (AED)'],
                mode='markers+lines', name='Forecast Price',
                marker=dict(symbol='x', color='firebrick'),
                line=dict(dash='dot')
            ))
            # Shaded forecast window
            fig.add_vrect(
                x0=forecast_df['Forecast Month'].min(),
                x1=forecast_df['Forecast Month'].max(),
                fillcolor='grey', opacity=0.2, line_width=0
            )
            fig.update_layout(
                title='📈 Price Trend & Forecast',
                height=520,
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table: only actual price
            forecast_df['Forecast Month'] = forecast_df['Forecast Month'].dt.strftime('%b %Y')
            st.table(forecast_df)
        else:
            st.info("Need at least 3 months of data for trend & forecast.")
    else:
        st.info("No date column for trend analysis.")


    # --- Always mirror valuation_metrics in session_state after all calculations ---
    st.session_state["valuation_metrics"] = valuation_metrics


with tab4:
    st.write("Placeholder for other functionality.")


