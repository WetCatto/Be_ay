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
    page_title="Executive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Shadcn UI Inspired CSS (Refined)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc; /* Slate-50 */
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Main Header */
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Metrics Override */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 20px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px; 
        color: #64748b; 
        font-weight: 500;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
    }

    /* Custom Tables & Lists */
    .custom-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
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
        border-bottom: 1px solid #f1f5f9;
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
        color: #334155;
    }
    .badge {
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 999px;
        font-weight: 600;
    }
    .badge-pos { background-color: #dcfce7; color: #166534; }
    .badge-neg { background-color: #fee2e2; color: #991b1b; }
    
    .progress-container {
        width: 50%;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .progress-bg {
        flex-grow: 1;
        height: 10px; /* Thicker bar */
        background-color: #f1f5f9;
        border-radius: 999px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background-color: #0f172a;
        border-radius: 999px;
    }
    .val-text {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        min-width: 40px;
        text-align: right;
    }

    /* Top Products Table Style */
    .prod-table {
        width: 100%;
        border-collapse: collapse;
    }
    .prod-table th {
        text-align: left;
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
        padding: 12px 16px;
        border-bottom: 1px solid #e2e8f0;
    }
    .prod-table td {
        padding: 16px;
        border-bottom: 1px solid #f1f5f9;
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
        background-color: #f1f5f9;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    .prod-name {
        font-size: 14px;
        font-weight: 600;
        color: #0f172a;
    }
    .prod-val {
        font-size: 14px;
        font-weight: 600;
        color: #334155;
    }
    .prod-meta {
        font-size: 13px;
        color: #64748b;
    }

    /* Clean up gaps */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
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
        df_products['Product_Cost'] = df_products['Product_Cost'].str.replace('$', '').str.strip().astype(float)
        df_products['Product_Price'] = df_products['Product_Price'].str.replace('$', '').str.strip().astype(float)

        sales_master = df_sales.merge(df_products, on='Product_ID', how='left')
        sales_master = sales_master.merge(df_stores, on='Store_ID', how='left')
        
        sales_master['Revenue'] = sales_master['Units'] * sales_master['Product_Price']
        sales_master['Cost'] = sales_master['Units'] * sales_master['Product_Cost']
        sales_master['Profit'] = sales_master['Revenue'] - sales_master['Cost']
        sales_master['Margin_Percent'] = (sales_master['Profit'] / sales_master['Revenue']).fillna(0) * 100

        inv_master = df_inventory.merge(df_products, on='Product_ID', how='left')
        inv_master = inv_master.merge(df_stores, on='Store_ID', how='left')

        return sales_master, inv_master
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df_sales, df_inventory = load_and_prep_data()

# ==========================================
# 3. SIDEBAR & FILTERS
# ==========================================
with st.sidebar:
    st.markdown("### 🛠️ Controls")
    
    # 1. Quick Filters
    st.markdown("**Time Period**")
    time_filter = st.radio(
        "Select Range", 
        ["Last 7 Days", "Last 30 Days", "Year to Date", "Custom"],
        label_visibility="collapsed",
        horizontal=True
    )
    
    max_date = df_sales['Date'].max()
    min_date = df_sales['Date'].min()
    
    if time_filter == "Last 7 Days":
        start_date = max_date - timedelta(days=6)
        end_date = max_date
    elif time_filter == "Last 30 Days":
        start_date = max_date - timedelta(days=29)
        end_date = max_date
    elif time_filter == "Year to Date":
        start_date = pd.Timestamp(year=max_date.year, month=1, day=1)
        end_date = max_date
    else:
        # Custom
        d = st.date_input("Custom Range", value=(max_date - timedelta(days=365), max_date), min_value=min_date, max_value=max_date)
        if len(d) == 2:
            start_date, end_date = pd.to_datetime(d[0]), pd.to_datetime(d[1])
        else:
            start_date, end_date = pd.to_datetime(d[0]), pd.to_datetime(d[0])

    st.markdown("---")
    
    cities = st.multiselect("Cities", df_sales['Store_City'].unique())
    categories = st.multiselect("Categories", df_sales['Product_Category'].unique())
    stores = st.multiselect("Stores", df_sales['Store_Name'].unique())

    st.markdown("---")
    st.info(f"Analyzing from **{start_date.date()}** to **{end_date.date()}**")

# Start Filtering
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

# ==========================================
# 4. MAIN CONTENT
# ==========================================
st.markdown('<div class="main-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)

# ---> BLOCK 1: KPIs
def calc_kpi(current_val, prev_val, is_currency=False):
    delta = current_val - prev_val
    pct = (delta / prev_val * 100) if prev_val != 0 else 0
    fmt = "${:,.0f}" if is_currency else "{:,.0f}"
    return fmt.format(current_val), f"{pct:+.1f}%"

curr_rev = curr_sales['Revenue'].sum()
curr_profit = curr_sales['Profit'].sum()
curr_units = curr_sales['Units'].sum()
stock_on_hand = filtered_inv['Stock_On_Hand'].sum()

# Prev metrics
prev_rev = prev_sales['Revenue'].sum()
prev_profit = prev_sales['Profit'].sum()
prev_units = prev_sales['Units'].sum()

kpi_rev, delta_rev = calc_kpi(curr_rev, prev_rev, True)
kpi_profit, delta_profit = calc_kpi(curr_profit, prev_profit, True)
kpi_units, delta_units = calc_kpi(curr_units, prev_units, False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", kpi_rev, delta_rev)
c2.metric("Profit", kpi_profit, delta_profit)
c3.metric("Units Sold", kpi_units, delta_units)
c4.metric("Stock On Hand", f"{stock_on_hand:,.0f}", help="Current physical stock")

st.markdown(" ") 

# ---> BLOCK 2: LINE CHARTS (Revenue & Return Rate)
col_l1, col_l2 = st.columns(2)

with col_l1:
    st.markdown("### Revenue Trend")
    ts_data = curr_sales.set_index('Date').resample('W')['Revenue'].sum().reset_index()
    ts_data['MA'] = ts_data['Revenue'].rolling(window=3).mean()
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts_data['Date'], y=ts_data['Revenue'], name='Actual', line=dict(color='#0f172a', width=2)))
    fig_ts.add_trace(go.Scatter(x=ts_data['Date'], y=ts_data['MA'], name='Forecast', line=dict(color='#f97316', dash='dash'))) 
    
    fig_ts.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        legend=dict(orientation="h", y=1.1, x=1, xanchor='right'),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    st.plotly_chart(fig_ts, use_container_width=True)

with col_l2:
    st.markdown("### Return Rate Trend")
    # Simulate Return Rate
    rr_data = curr_sales.set_index('Date').resample('W')['Units'].sum().reset_index()
    np.random.seed(42)
    rr_data['Return_Rate'] = 2.5 + np.random.normal(0, 0.5, len(rr_data)) # Simulated
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
        plot_bgcolor='white', paper_bgcolor='white'
    )
    st.plotly_chart(fig_rr, use_container_width=True)


# ---> BLOCK 3: SALES BY LOCATION & PRODUCT MIX
col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    st.markdown("### Sales by Location")
    
    # Calculate Sales data
    loc_sales = curr_sales.groupby('Store_City').agg({'Revenue':'sum'}).reset_index()
    # Prev sales for delta
    loc_prev = prev_sales.groupby('Store_City').agg({'Revenue':'sum'}).reset_index()
    loc_sales = loc_sales.merge(loc_prev, on='Store_City', how='left', suffixes=('', '_prev')).fillna(0)
    
    loc_sales['Delta'] = (loc_sales['Revenue'] - loc_sales['Revenue_prev']) / loc_sales['Revenue_prev'] * 100
    loc_sales = loc_sales.sort_values('Revenue', ascending=False).head(6) 
    
    max_rev = loc_sales['Revenue'].max() if len(loc_sales) > 0 else 1

    # Render HTML Rows
    # Increased styling for larger appearance
    html_content = ""
    for _, row in loc_sales.iterrows():
        pct = (row['Revenue'] / max_rev) * 100
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
    sunburst_data = sunburst_data.groupby('Product_Category').apply(lambda x: x.nlargest(3, 'Revenue')).reset_index(drop=True)
    
    fig_sb = px.sunburst(
        sunburst_data,
        path=['Product_Category', 'Product_Name'],
        values='Revenue',
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_sb.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0)) # Increased height to fill space
    st.plotly_chart(fig_sb, use_container_width=True)

# ---> BLOCK 4: TOP PRODUCTS DETAILS (Table)
st.markdown("### Best Selling Products")

prod_perf = curr_sales.groupby(['Product_Name', 'Product_Category']).agg({
    'Revenue': 'sum',
    'Units': 'sum'
}).reset_index()
prod_perf = prod_perf.sort_values('Revenue', ascending=False).head(8)

def get_icon(cat):
    if 'Electronics' in cat: return '💻'
    if 'Clothing' in cat: return '👕'
    if 'Home' in cat: return '🏠'
    if 'Toy' in cat: return '🧸'
    return '📦'

# Generate HTML Table
# Headers matching the data types in user's snippet (Price, Sold count)
table_html = '<table class="prod-table"><thead><tr><th>Product</th><th>Price</th><th>Sold</th></tr></thead><tbody>'

for _, row in prod_perf.iterrows():
    icon = get_icon(row['Product_Category'])
    # Calc average price for the Price column
    price_val = row['Revenue'] / row['Units'] if row['Units'] > 0 else 0
    
    # IMPORTANT: No indentation in HTML string to prevent markdown code block rendering
    table_html += f"""<tr>
<td><div class="prod-cell">
<div class="prod-icon">{icon}</div>
<span class="prod-name">{row['Product_Name']}</span>
</div></td>
<td><span class="prod-val">${price_val:,.2f}</span></td>
<td><span class="prod-val">{row['Units']:,.0f}</span></td>
</tr>"""

table_html += "</tbody></table>"

st.markdown(table_html, unsafe_allow_html=True)