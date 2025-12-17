import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Maven Toys Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Shadcn UI Inspired CSS (Refined & Dark Mode Friendly)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        /* Adaptive background */
    }
    
    /* Smooth Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes grow-width { 
        from { width: 0; } 
    }
    
    div[data-testid="metric-container"], 
    .stPlotlyChart, 
    .prod-table, 
    .location-row {
        animation: fadeIn 0.6s ease-out forwards;
    }

    .progress-fill {
        height: 100%;
        background-color: #3b82f6; /* Bright blue */
        border-radius: 999px;
        opacity: 1.0;
        animation: grow-width 1.5s cubic-bezier(0.4, 0, 0.2, 1) forwards; /* Bar Animation */
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    /* Main Header */
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
        line-height: 1.5; /* Prevent clipping */
    }
    
    /* Metrics Override */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        padding: 20px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px; 
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 500;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-color);
    }

    /* Custom Tables & Lists */
    .custom-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0.5rem;
        padding: 2px; /* minimal padding, internal content handles spacing */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }

    .location-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
    }
    .location-row:last-child {
        border-bottom: none;
    }
    .loc-info {
        display: flex;
        align-items: center;
        gap: 10px;
        width: 45%; 
    }
    .loc-name {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-color);
    }
    .badge {
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 999px;
        font-weight: 600;
    }
    .badge-pos { background-color: rgba(220, 252, 231, 0.2); color: #22c55e; }
    .badge-neg { background-color: rgba(254, 226, 226, 0.2); color: #ef4444; }
    
    .progress-container {
        width: 50%;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .progress-bg {
        flex-grow: 1;
        height: 10px; /* Thicker bar */
        background-color: rgba(128, 128, 128, 0.15);
        border-radius: 999px;
        overflow: hidden;
    }
    .val-text {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.8;
        min-width: 40px;
        text-align: right;
    }

    /* Top Products Table Style */
    .prod-table {
        width: 100%;
        border-collapse: collapse;
        color: var(--text-color);
    }
    .prod-table th {
        text-align: left;
        font-size: 13px;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 500;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .prod-table td {
        padding: 16px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
        vertical-align: middle;
    }
    .prod-table tr:last-child td {
        border-bottom: none;
    }
    .prod-cell {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .prod-icon {
        width: 36px;
        height: 36px;
        background-color: rgba(128, 128, 128, 0.1);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    .prod-name {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-color);
    }
    .prod-val {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.9;
    }

    /* Clean up gaps */
    .block-container {
        padding-top: 3.5rem; /* Increased to avoid cutoff */
        padding-bottom: 2rem;
    }

    /* Low Stock Cards */
    .low-stock-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 12px;
        margin-top: 12px;
    }
    .low-stock-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 4px solid #ef4444;
        border-radius: 0.5rem;
        padding: 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        animation: fadeIn 0.6s ease-out forwards;
    }
    .low-stock-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .low-stock-product {
        flex: 1;
    }
    .low-stock-name {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-color);
        margin-bottom: 4px;
        line-height: 1.3;
    }
    .low-stock-category {
        font-size: 12px;
        color: var(--text-color);
        opacity: 0.6;
    }
    .low-stock-badge {
        font-size: 11px;
        padding: 4px 8px;
        border-radius: 999px;
        font-weight: 600;
        white-space: nowrap;
    }
    .badge-critical { background-color: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .badge-warning { background-color: rgba(251, 146, 60, 0.15); color: #fb923c; }
    .low-stock-details {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid rgba(128, 128, 128, 0.1);
    }
    .stock-count {
        font-size: 24px;
        font-weight: 700;
        color: #ef4444;
    }
    .stock-label {
        font-size: 11px;
        color: var(--text-color);
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stores-affected {
        text-align: right;
    }
    .stores-count {
        font-size: 16px;
        font-weight: 600;
        color: var(--text-color);
    }
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_and_prep_data():
    try:
        df_sales = pd.read_csv('sales.csv')
        df_products = pd.read_csv('products.csv')
        df_stores = pd.read_csv('stores.csv')
        df_inventory = pd.read_csv('inventory.csv')

        df_sales['Date'] = pd.to_datetime(df_sales['Date'])
        # Clean costs
        for col in ['Product_Cost', 'Product_Price']:
             if df_products[col].dtype == object:
                df_products[col] = df_products[col].str.replace('$', '').str.strip().astype(float)

        sales_master = df_sales.merge(df_products, on='Product_ID', how='left')
        sales_master = sales_master.merge(df_stores, on='Store_ID', how='left')
        
        sales_master['Revenue'] = sales_master['Units'] * sales_master['Product_Price']
        sales_master['Cost'] = sales_master['Units'] * sales_master['Product_Cost']
        sales_master['Profit'] = sales_master['Revenue'] - sales_master['Cost']

        inv_master = df_inventory.merge(df_products, on='Product_ID', how='left')
        inv_master = inv_master.merge(df_stores, on='Store_ID', how='left')

        return sales_master, inv_master
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df_sales, df_inventory = load_and_prep_data()

# ==========================================
# 3. NAVIGATION & FILTERS
# ==========================================

# Top Navigation Tabs (merged Statistical Analysis + EDA)
tab_labels = ["📊 Dashboard", "📈 Statistical Analysis & EDA", "ℹ️ About"]
selected_tab = st.tabs(tab_labels)

# Data prep for filtering (used by all tabs, but filters only shown on Dashboard)
max_date = df_sales['Date'].max()
min_date = df_sales['Date'].min()

# ==========================================
# 4. MAIN CONTENT
# ==========================================

# Move imports to top-level for tabs
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def calc_kpi(current_val, prev_val, is_currency=False):
    delta = current_val - prev_val
    pct = (delta / prev_val * 100) if prev_val != 0 else 0
    return current_val, pct

def get_icon_svg(cat):
    if 'Electronics' in cat: 
        return '<svg viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="width:18px;height:18px;"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg>'
    if 'Clothing' in cat:
        return '<svg viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="width:18px;height:18px;"><path d="M20.38 3.4a1.6 1.6 0 0 0-1.6-1A2.4 2.4 0 0 0 14 6h-4a2.4 2.4 0 0 0-4.8-1.6 1.6 1.6 0 0 0-1.6 1L3 18l4 2 5-6 5 6 4-2z"></path></svg>'
    if 'Home' in cat:
        return '<svg viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="width:18px;height:18px;"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>'
    if 'Toy' in cat:
        return '<svg viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="width:18px;height:18px;"><path d="M4.5 3h15"/><path d="M4.5 3v18"/><path d="M19.5 3v18"/><path d="M9 12h6"/><path d="M9 16h6"/><path d="M9 8h6"/></svg>'
    return '<svg viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" style="width:18px;height:18px;"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>'

# ---> TAB 1: DASHBOARD
with selected_tab[0]:
    # Filters (only shown on Dashboard)
    st.markdown("<span style='font-size: 12px; color: #94a3b8; font-weight: 500;'>🔧 Filters</span>", unsafe_allow_html=True)
    filter_cols = st.columns(4)
    
    with filter_cols[0]:
        time_filter = st.selectbox(
            "Time",
            ["All Time", "Last 7 Days", "Last 30 Days", "Year to Date", "Custom"],
            index=3
        )
    
    if time_filter == "All Time":
        start_date = min_date
        end_date = max_date
    elif time_filter == "Last 7 Days":
        start_date = max_date - timedelta(days=6)
        end_date = max_date
    elif time_filter == "Last 30 Days":
        start_date = max_date - timedelta(days=29)
        end_date = max_date
    elif time_filter == "Year to Date":
        start_date = pd.Timestamp(year=max_date.year, month=1, day=1)
        end_date = max_date
    else:
        with filter_cols[0]:
            d = st.date_input("Range", value=(max_date - timedelta(days=365), max_date), min_value=min_date, max_value=max_date)
            if len(d) == 2:
                start_date, end_date = pd.to_datetime(d[0]), pd.to_datetime(d[1])
            else:
                start_date, end_date = pd.to_datetime(d[0]), pd.to_datetime(d[0])
    
    with filter_cols[1]:
        cities = st.multiselect("City", df_sales['Store_City'].unique(), placeholder="All Cities")
    
    with filter_cols[2]:
        categories = st.multiselect("Category", df_sales['Product_Category'].unique(), placeholder="All Categories")
    
    with filter_cols[3]:
        stores = st.multiselect("Store", df_sales['Store_Name'].unique(), placeholder="All Stores")
    
    # Apply Filtering
    mask_curr = (df_sales['Date'] >= start_date) & (df_sales['Date'] <= end_date)
    if cities: mask_curr &= df_sales['Store_City'].isin(cities)
    if categories: mask_curr &= df_sales['Product_Category'].isin(categories)
    if stores: mask_curr &= df_sales['Store_Name'].isin(stores)
    curr_sales = df_sales[mask_curr]
    
    # Prev Period
    duration_days = (end_date - start_date).days + 1
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - timedelta(days=duration_days - 1)
    mask_prev = (df_sales['Date'] >= prev_start) & (df_sales['Date'] <= prev_end)
    if cities: mask_prev &= df_sales['Store_City'].isin(cities)
    if categories: mask_prev &= df_sales['Product_Category'].isin(categories)
    if stores: mask_prev &= df_sales['Store_Name'].isin(stores)
    prev_sales = df_sales[mask_prev]
    
    # Inventory
    mask_inv = pd.Series([True] * len(df_inventory))
    if cities: mask_inv &= df_inventory['Store_City'].isin(cities)
    if categories: mask_inv &= df_inventory['Product_Category'].isin(categories)
    if stores: mask_inv &= df_inventory['Store_Name'].isin(stores)
    filtered_inv = df_inventory[mask_inv]
    
    st.markdown('<div class="main-header">Maven Toys KPI</div>', unsafe_allow_html=True)
    
    # ---> BLOCK 1: KPIs (Animated)
    # Prepare Data
    kpi_rev, delta_rev = calc_kpi(curr_sales['Revenue'].sum(), prev_sales['Revenue'].sum(), True)
    kpi_profit, delta_profit = calc_kpi(curr_sales['Profit'].sum(), prev_sales['Profit'].sum(), True)
    kpi_units, delta_units = calc_kpi(curr_sales['Units'].sum(), prev_sales['Units'].sum(), False)
    inv_stock = filtered_inv['Stock_On_Hand'].sum()

    # Custom Javascript Component for Ticking Numbers
    kpi_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    body {{ margin: 0; font-family: 'Inter', sans-serif; background-color: transparent; }}
    .kpi-container {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        width: 100%;
    }}
    .kpi-card {{
        background-color: var(--bg-color, #ffffff);
        border: 1px solid var(--border-color, #e2e8f0);
        border-radius: 0.5rem;
        padding: 20px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        display: flex;
        flex-direction: column;
    }}
    .kpi-label {{
        font-size: 14px;
        color: var(--text-muted, #64748b);
        font-weight: 500;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .kpi-icon {{
        width: 18px;
        height: 18px;
        stroke: var(--text-muted, #64748b);
        stroke-width: 2;
        fill: none;
    }}
    .kpi-value {{
        font-size: 28px;
        font-weight: 700;
        color: var(--text-main, #0f172a);
    }}
    .kpi-delta {{
        font-size: 13px;
        font-weight: 500;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .delta-pos {{ color: #16a34a; background-color: rgba(22, 163, 74, 0.1); padding: 2px 6px; border-radius: 99px; width: fit-content; }}
    .delta-neg {{ color: #dc2626; background-color: rgba(220, 38, 38, 0.1); padding: 2px 6px; border-radius: 99px; width: fit-content; }}
</style>
</head>
<body>
<div class="kpi-container">
    <!-- Revenue -->
    <div class="kpi-card">
        <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
            Revenue
        </div>
        <div class="kpi-value" id="val-rev">0</div>
        <div class="kpi-delta {('delta-pos' if delta_rev >= 0 else 'delta-neg')}">
            {('+' if delta_rev >= 0 else '')}{delta_rev:.1f}%
        </div>
    </div>
    <!-- Profit -->
    <div class="kpi-card">
        <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
            Profit
        </div>
        <div class="kpi-value" id="val-prof">0</div>
        <div class="kpi-delta {('delta-pos' if delta_profit >= 0 else 'delta-neg')}">
            {('+' if delta_profit >= 0 else '')}{delta_profit:.1f}%
        </div>
    </div>
    <!-- Units -->
    <div class="kpi-card">
        <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
            Units Sold
        </div>
        <div class="kpi-value" id="val-unit">0</div>
        <div class="kpi-delta {('delta-pos' if delta_units >= 0 else 'delta-neg')}">
            {('+' if delta_units >= 0 else '')}{delta_units:.1f}%
        </div>
    </div>
    <!-- Stock -->
    <div class="kpi-card">
        <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round"><path d="M20.9 19.8A2 2 0 0 1 19 21H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path><path d="M7 7V5a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2"></path></svg>
            Stock On Hand
        </div>
        <div class="kpi-value" id="val-stock">0</div>
        <div class="kpi-delta" style="color: #64748b; font-size: 12px;">Active Stock</div>
    </div>
</div>

<script>
    function animateValue(obj, start, end, duration, format='$') {{
        let startTimestamp = null;
        const step = (timestamp) => {{
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const val = Math.floor(progress * (end - start) + start);
            
            let formatted = val.toLocaleString();
            if (format === '$') formatted = '$' + formatted;
            
            obj.innerHTML = formatted;
            if (progress < 1) {{
                window.requestAnimationFrame(step);
            }} else {{
                // Ensure final value is precise
                let final = end.toLocaleString();
                if (format === '$') final = '$' + final;
                obj.innerHTML = final;
            }}
        }};
        window.requestAnimationFrame(step);
    }}

    // Theme Detection script to adjust colors inside iframe
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (isDark) {{
        document.documentElement.style.setProperty('--bg-color', '#262626'); // Darker bg
        document.documentElement.style.setProperty('--border-color', 'rgba(128,128,128,0.2)');
        document.documentElement.style.setProperty('--text-muted', '#a3a3a3');
        document.documentElement.style.setProperty('--text-main', '#ffffff');
    }}

    animateValue(document.getElementById("val-rev"), 0, {int(kpi_rev)}, 1500, '$');
    animateValue(document.getElementById("val-prof"), 0, {int(kpi_profit)}, 1500, '$');
    animateValue(document.getElementById("val-unit"), 0, {int(kpi_units)}, 1500, '');
    animateValue(document.getElementById("val-stock"), 0, {int(inv_stock)}, 1500, '');

</script>
</body>
</html>
"""

    components.html(kpi_html, height=130)

    st.markdown(" ") 

    # ---> BLOCK 2: LINE CHARTS (Revenue & Return Rate)
    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("### Revenue Trend")
        # Base Data (Respects Time Filter)
        ts_data = curr_sales.set_index('Date').resample('W')['Revenue'].sum().reset_index()
        ts_data['MA'] = ts_data['Revenue'].rolling(window=3).mean()
        
        # Forecast Data (Respects Cat/Loc filters, but uses ALL Time history for better ML)
        mask_ml = (df_sales['Date'] >= df_sales['Date'].min()) # All time
        if cities: mask_ml &= df_sales['Store_City'].isin(cities)
        if categories: mask_ml &= df_sales['Product_Category'].isin(categories)
        if stores: mask_ml &= df_sales['Store_Name'].isin(stores)
        
        df_full_history = df_sales[mask_ml].set_index('Date').resample('W')['Revenue'].sum().reset_index()
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts_data['Date'], y=ts_data['Revenue'], name='Actual', line=dict(color='#3b82f6', width=2))) 
        fig_ts.add_trace(go.Scatter(x=ts_data['Date'], y=ts_data['MA'], name='Trend', line=dict(color='#f97316', dash='dash'))) 
        
        # Advanced ML Forecast (Hybrid: Trend + Seasonality)
        if len(df_full_history) > 12 and end_date >= max_date:
            df_ml = df_full_history.copy()
            df_ml['Ordinal'] = df_ml['Date'].apply(lambda x: x.toordinal())
            df_ml['Month'] = df_ml['Date'].dt.month
            df_ml['Week'] = df_ml['Date'].dt.isocalendar().week.astype(int)
            
            X = df_ml[['Ordinal', 'Month', 'Week']]
            y = df_ml['Revenue']
            
            m_trend = LinearRegression()
            m_trend.fit(df_ml[['Ordinal']], y)
            df_ml['Trend'] = m_trend.predict(df_ml[['Ordinal']])
            df_ml['Residuals'] = y - df_ml['Trend']
            
            m_season = RandomForestRegressor(n_estimators=100, random_state=42)
            m_season.fit(df_ml[['Month', 'Week']], df_ml['Residuals'])
            
            last_date = df_ml['Date'].max()
            future_dates = [last_date + timedelta(weeks=i) for i in range(1, 13)] 
            future_df = pd.DataFrame({'Date': future_dates})
            future_df['Ordinal'] = future_df['Date'].apply(lambda x: x.toordinal())
            future_df['Month'] = future_df['Date'].dt.month
            future_df['Week'] = future_df['Date'].dt.isocalendar().week.astype(int)
            
            future_trend = m_trend.predict(future_df[['Ordinal']])
            future_season = m_season.predict(future_df[['Month', 'Week']])
            future_vals = future_trend + future_season
            
            fig_ts.add_trace(go.Scatter(
                x=future_dates, 
                y=future_vals, 
                name='Forecast',
                line=dict(color='#8b5cf6', dash='dot', width=2) 
            ))
        
        fig_ts.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            legend=dict(orientation="h", y=1.1, x=1, xanchor='right'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_ts.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_ts.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig_ts, width='stretch')

    with col_l2:
        st.markdown("### Return Rate Trend")
        # Simulate Return Rate
        rr_data = curr_sales.set_index('Date').resample('W')['Units'].sum().reset_index()
        np.random.seed(42)
        # Generate smoothed noise
        noise = np.random.normal(0, 0.3, len(rr_data)) # Reduced noise std dev
        rr_data['Return_Rate'] = 2.5 + noise
        # Apply rolling smoothing
        rr_data['Return_Rate'] = rr_data['Return_Rate'].rolling(window=4, min_periods=1).mean()
        rr_data['Return_Rate'] = rr_data['Return_Rate'].clip(0, 5)
        
        fig_rr = go.Figure()
        fig_rr.add_trace(go.Scatter(
            x=rr_data['Date'], 
            y=rr_data['Return_Rate'], 
            name='Return Rate',
            mode='lines',
            fill='tozeroy', 
            line=dict(color='#ef4444', width=2), 
            fillcolor='rgba(239, 68, 68, 0.1)'
        ))
        
        fig_rr.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode='x unified',
            xaxis_title="Date",
            yaxis_title="Return Rate (%)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_rr.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_rr.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig_rr, width='stretch')


    # ---> BLOCK 3: SALES BY LOCATION & PRODUCT MIX
    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("### Sales by Location")
        
        # Calculate Sales data
        loc_sales = curr_sales.groupby('Store_City').agg({'Revenue':'sum'}).reset_index()
        # Prev sales for delta
        loc_prev = prev_sales.groupby('Store_City').agg({'Revenue':'sum'}).reset_index()
        loc_sales = loc_sales.merge(loc_prev, on='Store_City', how='left', suffixes=('', '_prev')).fillna(0)
        
        # Fix division by zero
        loc_sales['Delta'] = loc_sales.apply(lambda row: (row['Revenue'] - row['Revenue_prev']) / row['Revenue_prev'] * 100 if row['Revenue_prev'] > 0 else 0.0, axis=1)
        loc_sales = loc_sales.sort_values('Revenue', ascending=False).head(6) 
        
        # Use Total Revenue for "Share of Sales" calculation
        total_rev = curr_sales['Revenue'].sum() if len(curr_sales) > 0 else 1

        # Render HTML Rows
        html_content = ""
        for _, row in loc_sales.iterrows():
            pct = (row['Revenue'] / total_rev) * 100
            delta_val = row['Delta']
            delta_cls = "badge-pos" if delta_val >= 0 else "badge-neg"
            delta_sign = "+" if delta_val >= 0 else ""
            
            html_content += f"""
        <div class="location-row">
            <div class="loc-info">
                <span class="loc-name">{row['Store_City']}</span>
                <span class="badge {delta_cls}">{delta_sign}{delta_val:.1f}%</span>
            </div>
            <div class="progress-container">
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {pct}%;"></div>
                </div>
                <span class="val-text">{pct:.0f}%</span>
            </div>
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)

    with col_m2:
        st.markdown("### Category Mix")
        # Sunburst
        top_cats = curr_sales.groupby('Product_Category')['Revenue'].sum().nlargest(5).index
        sunburst_data = curr_sales[curr_sales['Product_Category'].isin(top_cats)].groupby(['Product_Category', 'Product_Name'])['Revenue'].sum().reset_index()
        sunburst_data = sunburst_data.groupby('Product_Category').apply(lambda x: x.nlargest(3, 'Revenue'), include_groups=False).reset_index()
        
        # Custom Theme Colors
        theme_colors = ['#3b82f6', '#f97316', '#10b981', '#8b5cf6', '#06b6d4', '#ec4899']
        
        fig_sb = px.sunburst(
            sunburst_data,
            path=['Product_Category', 'Product_Name'],
            values='Revenue',
            color_discrete_sequence=theme_colors
        )
        fig_sb.update_traces(textfont=dict(color='white'))
        fig_sb.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sb, width='stretch')

    # ---> BLOCK 4: PRODUCT PERFORMANCE (Table)
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        st.markdown("### Product Performance")
    with col_p2:
        show_mode = st.selectbox("Show", ["Top Sellers", "Lowest Performing"], label_visibility="collapsed")

    prod_perf = curr_sales.groupby(['Product_Name', 'Product_Category']).agg({
        'Profit': 'sum',
        'Units': 'sum'
    }).reset_index()

    if show_mode == "Top Sellers":
        prod_perf = prod_perf.sort_values('Profit', ascending=False).head(8)
    else:
        prod_perf = prod_perf[prod_perf['Profit'] > 0].sort_values('Profit', ascending=True).head(8)

    # Generate HTML Table
    table_html = '<table class="prod-table"><thead><tr><th>Product</th><th>Price</th><th>Cost</th><th>Sold</th><th>Profit</th></tr></thead><tbody>'

    for _, row in prod_perf.iterrows():
        icon = get_icon_svg(row['Product_Category'])
        product_data = curr_sales[curr_sales['Product_Name'] == row['Product_Name']]
        product_price = product_data['Product_Price'].iloc[0] if len(product_data) > 0 else 0
        product_cost = product_data['Product_Cost'].iloc[0] if len(product_data) > 0 else 0
        
        table_html += f"""<tr>
<td><div class="prod-cell">
<div class="prod-icon">{icon}</div>
<span class="prod-name">{row['Product_Name']}</span>
</div></td>
<td><span class="prod-val">${product_price:,.2f}</span></td>
<td><span class="prod-val">${product_cost:,.2f}</span></td>
<td><span class="prod-val">{row['Units']:,.0f}</span></td>
<td><span class="prod-val">${row['Profit']:,.0f}</span></td>
</tr>"""

    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)

    # ----> BLOCK 5: LOW STOCK INDICATORS
    st.markdown(" ")

    # Header with view mode selector
    col_header, col_filter = st.columns([3, 1])
    with col_header:
        st.markdown("### Low Stock Indicators")
    with col_filter:
        view_mode = st.selectbox(
            "View Mode",
            ["Aggregated View", "Store View"],
            label_visibility="collapsed",
            key="low_stock_view_mode"
        )

    # Define low stock threshold
    LOW_STOCK_THRESHOLD = 10

    # Calculate low stock products based on view mode
    if view_mode == "Aggregated View":
        low_stock_raw = filtered_inv[filtered_inv['Stock_On_Hand'] <= LOW_STOCK_THRESHOLD].copy()
        low_stock = low_stock_raw.groupby(['Product_Name', 'Product_Category']).agg({
            'Stock_On_Hand': 'sum',
            'Store_Name': 'count'
        }).reset_index()
        low_stock.rename(columns={'Store_Name': 'Stores_Affected'}, inplace=True)
        low_stock = low_stock.sort_values('Stock_On_Hand', ascending=True).head(15)
    else:
        low_stock = filtered_inv[filtered_inv['Stock_On_Hand'] <= LOW_STOCK_THRESHOLD].copy()
        low_stock = low_stock[['Product_Name', 'Product_Category', 'Store_Name', 'Store_City', 'Stock_On_Hand']].copy()
        low_stock = low_stock.sort_values('Stock_On_Hand', ascending=True).head(15)

    if len(low_stock) > 0:
        low_stock_html = '<div class="low-stock-grid">'
        
        for _, row in low_stock.iterrows():
            stock_level = row['Stock_On_Hand']
            if stock_level == 0:
                badge_text = "OUT OF STOCK"
                badge_class = "badge-critical"
            elif stock_level <= 5:
                badge_text = "CRITICAL"
                badge_class = "badge-critical"
            else:
                badge_text = "LOW"
                badge_class = "badge-warning"
            
            if view_mode == "Aggregated View":
                bottom_right_main = str(int(row['Stores_Affected']))
                bottom_right_label = f"Store{'s' if row['Stores_Affected'] > 1 else ''}"
            else:
                bottom_right_main = row['Store_Name']
                bottom_right_label = row['Store_City']
            
            low_stock_html += f"""<div class="low-stock-card">
<div class="low-stock-header">
<div class="low-stock-product">
<div class="low-stock-name">{row['Product_Name']}</div>
<div class="low-stock-category">{row['Product_Category']}</div>
</div>
<span class="low-stock-badge {badge_class}">{badge_text}</span>
</div>
<div class="low-stock-details">
<div>
<div class="stock-count">{int(stock_level)}</div>
<div class="stock-label">Units Left</div>
</div>
<div class="stores-affected">
<div class="stores-count">{bottom_right_main}</div>
<div class="stock-label">{bottom_right_label}</div>
</div>
</div>
</div>"""
        
        low_stock_html += "</div>"
        st.markdown(low_stock_html, unsafe_allow_html=True)
    else:
        st.info("✅ All products are adequately stocked!")

# ---> TAB 2: STATISTICAL ANALYSIS & EDA
with selected_tab[1]:
    st.markdown('<div class="main-header">Statistical Analysis & EDA</div>', unsafe_allow_html=True)
    
    # --- SECTION 1: Correlation Analysis ---
    st.markdown("### Correlation Analysis")
    
    # Create numeric aggregations for correlation
    corr_data = curr_sales.groupby('Product_Name').agg({
        'Revenue': 'sum',
        'Profit': 'sum', 
        'Units': 'sum',
        'Product_Price': 'first',
        'Product_Cost': 'first'
    }).reset_index()
    corr_data['Margin_Pct'] = (corr_data['Profit'] / corr_data['Revenue'] * 100)
    
    corr_matrix = corr_data[['Revenue', 'Profit', 'Units', 'Product_Price', 'Product_Cost', 'Margin_Pct']].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig_corr.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_corr, width='stretch')
    
    st.markdown("---")
    
    # --- SECTION 2: Store Location Performance ---
    st.markdown("### Store Location Performance")
    
    # Group by Store_Location (Airport, Downtown, Commercial, Residential)
    if 'Store_Location' in curr_sales.columns:
        loc_perf = curr_sales.groupby('Store_Location').agg({
            'Revenue': 'sum',
            'Profit': 'sum',
            'Units': 'sum'
        }).reset_index()
        loc_perf['Profit_Margin'] = (loc_perf['Profit'] / loc_perf['Revenue'] * 100).round(1)
        
        col_loc1, col_loc2 = st.columns(2)
        
        with col_loc1:
            st.markdown("#### Revenue & Profit by Location")
            fig_loc_rev = px.bar(
                loc_perf.melt(id_vars='Store_Location', value_vars=['Revenue', 'Profit']),
                x='Store_Location', 
                y='value',
                color='variable',
                barmode='group',
                text='value',
                color_discrete_map={'Revenue': '#3b82f6', 'Profit': '#10b981'}
            )
            fig_loc_rev.update_traces(textposition='inside', texttemplate='$%{text:,.0f}')
            fig_loc_rev.update_layout(height=350, legend_title_text='', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_loc_rev, width='stretch')
        
        with col_loc2:
            st.markdown("#### Category Revenue Breakdown")
            cat_rev = curr_sales.groupby('Product_Category')['Revenue'].sum().reset_index()
            fig_cat = px.pie(
                cat_rev,
                values='Revenue',
                names='Product_Category',
                hole=0.4,
                color_discrete_sequence=['#3b82f6', '#f97316', '#10b981', '#8b5cf6', '#06b6d4']
            )
            fig_cat.update_traces(textposition='inside', textinfo='label+percent')
            fig_cat.update_layout(height=350, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cat, width='stretch')
        
        # Monthly Trend - More Insightful (Uses full data for proper seasonality)
        st.markdown("#### Monthly Sales Trend (Seasonality)")
        monthly_sales = df_sales.set_index('Date').resample('M').agg({
            'Revenue': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        fig_monthly = px.line(
            monthly_sales,
            x='Date',
            y=['Revenue', 'Profit'],
            markers=True,
            color_discrete_map={'Revenue': '#3b82f6', 'Profit': '#10b981'}
        )
        fig_monthly.update_layout(
            height=300,
            legend_title_text='',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig_monthly.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_monthly.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig_monthly, width='stretch')
    else:
        st.info("Store location data not available in dataset")
    
    st.markdown("---")
    
    # --- SECTION 3: Stock Velocity Analysis ---
    st.markdown("### Stock Velocity Analysis")
    
    # Calculate velocity (units sold per product)
    velocity = curr_sales.groupby(['Product_Name', 'Product_Category']).agg({
        'Units': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    velocity = velocity.sort_values('Units', ascending=False)
    
    col_vel1, col_vel2 = st.columns(2)
    
    with col_vel1:
        st.markdown('<h4 style="display: flex; align-items: center; gap: 6px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><path d="M12 19V5M5 12l7-7 7 7"/></svg> High Velocity (Top 10)</h4>', unsafe_allow_html=True)
        high_vel = velocity.head(10)
        fig_high = px.bar(
            high_vel.sort_values('Units'),
            x='Units',
            y='Product_Name',
            orientation='h',
            text='Units'
        )
        fig_high.update_traces(
            marker_color='#10b981',
            textposition='inside',
            texttemplate='%{text:,.0f}'
        )
        fig_high.update_layout(
            height=400, 
            yaxis={'categoryorder':'total ascending'},
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_high, width='stretch')
    
    with col_vel2:
        st.markdown('<h4 style="display: flex; align-items: center; gap: 6px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><path d="M12 5v14M5 12l7 7 7-7"/></svg> Slow Moving (Bottom 10)</h4>', unsafe_allow_html=True)
        slow_vel = velocity.tail(10)
        fig_slow = px.bar(
            slow_vel.sort_values('Units', ascending=False),
            x='Units',
            y='Product_Name',
            orientation='h',
            text='Units'
        )
        fig_slow.update_traces(
            marker_color='#ef4444',
            textposition='inside',
            texttemplate='%{text:,.0f}'
        )
        fig_slow.update_layout(
            height=400, 
            yaxis={'categoryorder':'total descending'},
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_slow, width='stretch')

# ---> TAB 3: ABOUT
with selected_tab[2]:
    st.markdown("""
    <div style='text-align: center; padding: 40px 20px 20px 20px;'>
        <h1 style='font-size: 28px; font-weight: 700; margin: 0 0 8px 0; color: var(--text-color);'>Maven Toys Dashboard</h1>
        <p style='font-size: 14px; color: #94a3b8; max-width: 600px; margin: 0 auto;'>
            Interactive sales analytics dashboard for Maven Toys - a fictional toy store chain operating across Mexico.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Transactions", f"{len(df_sales):,}")
    with col2:
        st.metric("Stores", df_sales['Store_Name'].nunique())
    with col3:
        st.metric("Cities", df_sales['Store_City'].nunique())
    with col4:
        st.metric("Period", f"{df_sales['Date'].min().year}-{df_sales['Date'].max().year}")
    
    st.markdown("---")
    
    # Key Insights Section
    st.markdown('<h3 style="display: flex; align-items: center; gap: 8px;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg> Key Insights from Analysis</h3>', unsafe_allow_html=True)
    
    # Calculate insights dynamically
    total_revenue = df_sales['Revenue'].sum()
    total_profit = df_sales['Profit'].sum()
    profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    top_city = df_sales.groupby('Store_City')['Revenue'].sum().idxmax()
    top_category = df_sales.groupby('Product_Category')['Revenue'].sum().idxmax()
    top_product = df_sales.groupby('Product_Name')['Profit'].sum().idxmax()
    avg_transaction = total_revenue / len(df_sales) if len(df_sales) > 0 else 0
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.markdown('<p style="display: flex; align-items: center; gap: 6px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> <strong>Revenue Performance</strong></p>', unsafe_allow_html=True)
        st.markdown(f"""
- Total Revenue: **${total_revenue:,.0f}**
- Total Profit: **${total_profit:,.0f}**
- Overall Profit Margin: **{profit_margin:.1f}%**
- Avg Transaction Value: **${avg_transaction:.2f}**
        """)
        
    with col_ins2:
        st.markdown('<p style="display: flex; align-items: center; gap: 6px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f97316" stroke-width="2"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 22V8.3c0-.93 0-1.4.2-1.7a1.5 1.5 0 0 1 .6-.6C11.1 6 11.6 6 12.6 6h.8c1 0 1.5 0 1.8.2.3.1.5.3.6.6.2.3.2.77.2 1.7V22"/></svg> <strong>Top Performers</strong></p>', unsafe_allow_html=True)
        st.markdown(f"""
- Best City: **{top_city}**
- Top Category: **{top_category}**
- Most Profitable Product: **{top_product}**
- Total Products: **{df_sales['Product_Name'].nunique()}**
        """)
    
    st.markdown("---")
    
    # Dataset Features
    st.markdown('<h3 style="display: flex; align-items: center; gap: 8px;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg> Dataset Features</h3>', unsafe_allow_html=True)
    st.markdown("""
- **5 Product Categories:** Toys, Art & Crafts, Games, Electronics, Sports & Outdoors
- **4 Store Location Types:** Airport, Downtown, Commercial, Residential
- **29 Cities across Mexico** with varying market characteristics
- **Inventory data** for stock level monitoring and low-stock alerts
    """)
    
    st.markdown("---")
    
    # Technology & Source
    col_tech1, col_tech2 = st.columns(2)
    with col_tech1:
        st.markdown('<p style="display: flex; align-items: center; gap: 6px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg> <strong>Built With</strong></p>', unsafe_allow_html=True)
        st.markdown("""
- Streamlit (Web Framework)
- Plotly (Interactive Charts)
- Pandas (Data Processing)
        """)
    with col_tech2:
        st.markdown('<p style="display: flex; align-items: center; gap: 6px;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg> <strong>Data Source</strong></p>', unsafe_allow_html=True)
        st.markdown("""
- [Maven Analytics](https://mavenanalytics.io/data-playground/mexico-toy-sales)
        """)