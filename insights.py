import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(
    page_title="Strategic Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Analysis Report
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    .report-card { background-color: white; padding: 24px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 24px; }
    h1, h2, h3 { color: #0f172a; }
    .metric-label { font-size: 14px; color: #64748b; }
    .metric-value { font-size: 28px; font-weight: 700; color: #0f172a; }
    .insight-box { background-color: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 16px; border-radius: 4px; color: #0c4a6e; font-size: 15px; margin-top: 12px; }
</style>
""", unsafe_allow_html=True)

# 1. DATA LOADING
@st.cache_data
def load_data():
    df_sales = pd.read_csv('sales.csv')
    df_products = pd.read_csv('products.csv')
    df_stores = pd.read_csv('stores.csv')
    df_inventory = pd.read_csv('inventory.csv')

    df_sales['Date'] = pd.to_datetime(df_sales['Date'])
    
    # Cleaning Costs
    for col in ['Product_Cost', 'Product_Price']:
        df_products[col] = df_products[col].str.replace('$', '').str.strip().astype(float)

    # Master Merges
    sales_master = df_sales.merge(df_products, on='Product_ID', how='left')
    sales_master = sales_master.merge(df_stores, on='Store_ID', how='left')
    
    # KPIs
    sales_master['Revenue'] = sales_master['Units'] * sales_master['Product_Price']
    sales_master['Cost'] = sales_master['Units'] * sales_master['Product_Cost']
    sales_master['Profit'] = sales_master['Revenue'] - sales_master['Cost']

    # Inventory Master
    inv_master = df_inventory.merge(df_products, on='Product_ID', how='left')
    inv_master = inv_master.merge(df_stores, on='Store_ID', how='left')
    inv_master['Inv_Value'] = inv_master['Stock_On_Hand'] * inv_master['Product_Cost']
    
    return sales_master, inv_master

df_sales, df_inv = load_data()

st.title("📈 Strategic Business Insights")
st.markdown("Deep dive analysis answering key business questions.")

# ==============================================================================
# Q1: Which product categories drive the biggest profits? Is this the same across store locations?
# ==============================================================================
st.markdown("---")
st.header("1. Profit Drivers: Categories & Locations")

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("##### Total Profit by Category")
    cat_profit = df_sales.groupby('Product_Category')['Profit'].sum().reset_index().sort_values('Profit', ascending=True)
    
    fig_cat = px.bar(cat_profit, x='Profit', y='Product_Category', orientation='h', text_auto='.2s', color='Profit', color_continuous_scale='Blues')
    fig_cat.update_layout(height=400, xaxis_title="Total Profit ($)", yaxis_title="")
    st.plotly_chart(fig_cat, use_container_width=True)

with c2:
    st.markdown("##### Top Category by Store Location")
    # Find top category for each store
    store_cat = df_sales.groupby(['Store_City', 'Product_Category'])['Profit'].sum().reset_index()
    # Sort and pick top 1
    top_store = store_cat.sort_values(['Store_City', 'Profit'], ascending=[True, False]).groupby('Store_City').head(1)
    
    fig_map = px.bar(top_store, x='Store_City', y='Profit', color='Product_Category', text='Product_Category', title="Dominant Category per City")
    fig_map.update_layout(height=400)
    st.plotly_chart(fig_map, use_container_width=True)

# Insight Text
top_cat_name = cat_profit.iloc[-1]['Product_Category']
st.markdown(f"""
<div class='insight-box'>
    <b>💡 Insight:</b> <b>{top_cat_name}</b> is the primary profit driver globally. 
    However, location data shows variation. Check if larger cities favor different categories compared to smaller ones.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Q2: Can you find any seasonal trends or patterns in the sales data?
# ==============================================================================
st.markdown("---")
st.header("2. Seasonal Trends & Patterns")

# Aggregate by Month
df_sales['YearMonth'] = df_sales['Date'].dt.to_period('M').astype(str)
monthly_sales = df_sales.groupby('YearMonth')['Revenue'].sum().reset_index()

fig_trend = px.line(monthly_sales, x='YearMonth', y='Revenue', markers=True, title="Monthly Revenue Trend")
fig_trend.update_layout(height=350, xaxis_title="Month", yaxis_title="Revenue ($)")
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("""
<div class='insight-box'>
    <b>💡 Insight:</b> Look for recurring peaks. Toy sales typically spike in <b>November/December</b> (Holiday Season) and potentially in summer months. 
    Flat periods indicate opportunities for mid-year promotions.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Q3: Are we losing any sales with products being out of stock at certain locations?
# ==============================================================================
st.markdown("---")
st.header("3. Stockout Impact Analysis")

# 1. Identify Out of Stock Items (Stock_On_Hand == 0)
oos_items = df_inv[df_inv['Stock_On_Hand'] == 0].copy()

# 2. Calculate "Velocity" (Avg Daily Units Sold) for these items per store
# We need Sales history for (Store, Product) pairs.
# Group last 90 days sales
max_date = df_sales['Date'].max()
mask_90 = df_sales['Date'] >= (max_date - timedelta(days=90))
recent_sales = df_sales[mask_90].groupby(['Store_ID', 'Product_ID'])['Units'].sum().reset_index()
recent_sales['Daily_Velocity'] = recent_sales['Units'] / 90.0

# Merge Velocity into OOS Items
oos_impact = oos_items.merge(recent_sales, on=['Store_ID', 'Product_ID'], how='left')
oos_impact['Daily_Lost_Revenue'] = oos_impact['Daily_Velocity'] * oos_impact['Product_Price']
oos_impact = oos_impact.dropna(subset=['Daily_Lost_Revenue']) # Remove items that never sold anyway

# Top Lost Revenue Opportunities
total_lost_daily = oos_impact['Daily_Lost_Revenue'].sum()
top_misses = oos_impact.groupby('Product_Name')['Daily_Lost_Revenue'].sum().reset_index().sort_values('Daily_Lost_Revenue', ascending=False).head(5)

c3, c4 = st.columns([1, 2])
with c3:
    st.metric("Est. Daily Lost Revenue", f"${total_lost_daily:,.0f}", help="Based on 90-day sales velocity of currently OOS items")
    st.markdown(f"**{len(oos_impact)}** SKUs are currently OOS at specific locations despite having demand.")

with c4:
    st.markdown("**Top Products Causing Lost Sales (Daily Risk)**")
    st.dataframe(
        top_misses.rename(columns={'Product_Name':'Product', 'Daily_Lost_Revenue':'Daily Loss ($)'})
        .style.format({'Daily Loss ($)': "${:,.2f}"}),
        use_container_width=True,
        hide_index=True
    )

st.markdown("""
<div class='insight-box'>
    <b>💡 Insight:</b> OOS isn't just "0 stock"—it's 0 stock on <i>high demand</i> items. 
    Prioritize restocking the products listed above to capture lost revenue immediately.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Q4: How much money is tied up in inventory at the toy stores?
# ==============================================================================
st.markdown("---")
st.header("4. Capital Tied Up in Inventory")

# As established, we'll treat all stores as "Toy Stores" contextually.
# Calculate Global Inventory Value
total_inv_value = df_inv['Inv_Value'].sum()

# Calculate "Days of Inventory On Hand" (DOH)
# Global Avg Daily COGS
last_30_sales = df_sales[df_sales['Date'] >= (max_date - timedelta(days=30))]
daily_cogs = last_30_sales['Cost'].sum() / 30.0
doh = total_inv_value / daily_cogs if daily_cogs > 0 else 0

c5, c6 = st.columns(2)
c5.metric("Total Inventory Value (Cost)", f"${total_inv_value/1e6:,.1f}M", help="Capital tied up in stock")
c6.metric("Days of Inventory (DOH)", f"{doh:.1f} Days", help="How long stock will last at current sales rate")

st.markdown("#### Inventory Value by Category")
cat_inv = df_inv.groupby('Product_Category')['Inv_Value'].sum().reset_index().sort_values('Inv_Value', ascending=False)
fig_tree = px.treemap(cat_inv, path=['Product_Category'], values='Inv_Value', title="Where is the money tied up?", color='Inv_Value', color_continuous_scale='Greens')
fig_tree.update_layout(height=350)
st.plotly_chart(fig_tree, use_container_width=True)

st.markdown("""
<div class='insight-box'>
    <b>💡 Insight:</b> A high DOH with high capital tied up (e.g., in Arts & Crafts) might indicate overstocking or slow-moving items. 
    Compare this with the Profit Drivers in Section 1 to ensure capital is backing high-margin items.
</div>
""", unsafe_allow_html=True)
