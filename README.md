# Maven Toys Sales & Inventory Dashboard

## Project Overview
This repository contains a comprehensive data analytics application built for **Maven Toys**, a fictional toy retailer operating in Mexico. The project utilizes **Streamlit** to deliver two distinct interfaces:
1.  **Operational Dashboard (app.py):** A real-time monitor of sales performance, inventory levels, and product trends.
2.  **Strategic Insights Report (insights.py):** A deep-dive analysis answering critical business questions regarding profit drivers, seasonality, and stockout impacts.

The application processes transaction-level data to visualize revenue, profit margins, and inventory efficiency across 50+ stores and thousands of products.

## Key Features

### 1. Main Dashboard (app.py)
* **KPI Monitor:** Real-time tracking of Revenue, Profit, Units Sold, and Stock on Hand with week-over-week percentage changes.
* **Sales Forecasting:** Uses Machine Learning (Random Forest + Linear Regression) to project future revenue trends based on historical seasonality.
* **Interactive Filters:** Drill down by Time Period, City, Store, and Product Category.
* **Inventory Alerts:** Visual indicators for "Low Stock" and "Critical Stock" items to prevent lost sales.
* **Statistical Analysis:**
    * Correlation matrices (Price vs. Sales vs. Profit).
    * Store location performance (Airport vs. Downtown vs. Residential).
    * Product velocity analysis (Fast vs. Slow-moving SKUs).

### 2. Strategic Insights (insights.py)
* **Profit Drivers:** Identifies which categories and locations contribute most to the bottom line.
* **Seasonality:** Visualizes monthly revenue trends to identify peak shopping periods.
* **Stockout Impact:** Calculates estimated daily lost revenue due to out-of-stock items on high-demand products.
* **Capital Analysis:** Assessment of capital tied up in inventory (Days of Inventory On Hand).

## File Structure

| File Name | Description |
| :--- | :--- |
| **app.py** | The main Streamlit application containing the operational dashboard and EDA tools. |
| **insights.py** | A separate Streamlit report focused on answering specific business questions. |
| **requirements.txt** | List of Python dependencies required to run the project. |
| **sales.csv** | Transactional sales data (Date, Store, Product, Units). |
| **inventory.csv** | Current stock levels per store and product. |
| **products.csv** | Product metadata (Name, Category, Cost, Price). |
| **stores.csv** | Store metadata (Name, City, Location Type). |

## Installation & Usage

### Prerequisites
* Python 3.8+
* Pip (Python Package Installer)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/maven-toys-dashboard.git](https://github.com/yourusername/maven-toys-dashboard.git)
    cd maven-toys-dashboard
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To launch the **Main Dashboard**:
```bash
streamlit run app.py
```

Technologies Used
Web Framework: Streamlit

Data Processing: Pandas, NumPy

Visualization: Plotly Express, Plotly Graph Objects

Machine Learning: Scikit-learn (RandomForestRegressor, LinearRegression) for sales forecasting.
