# ============================================================
# Experiment 2: ETL Process - Extract, Transform, Load
# Course: Data Mining (241AI003)
# Aditya University
# ============================================================

import pandas as pd
import numpy as np
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  EXPERIMENT 2: ETL PROCESS DEMONSTRATION")
print("=" * 60)

# ============================================================
# STEP 1: EXTRACT
# ============================================================
print("\n[STEP 1] EXTRACT - Loading data from multiple sources")
print("-" * 50)

# Source 1: Sales Data (CSV-like)
sales_csv = """TransactionID,CustomerID,Product,Amount,Date,Status
1001,C01,Laptop,75000,2024-01-10,Completed
1002,C02,Phone,25000,2024-01-11,Completed
1003,C03,Tablet,15000,2024-01-12,Pending
1004,C01,Laptop,75000,2024-01-10,Completed
1005,C04,Phone,25000,2024-01-13,Cancelled
1006,C05,Headphones,5000,2024-01-14,Completed
1007,,Camera,30000,2024-01-15,Completed
1008,C06,Tablet,15000,2024-01-16,Completed
"""

# Source 2: Customer Data
customer_csv = """CustomerID,Name,City,Age
C01,Alice,Mumbai,28
C02,Bob,Delhi,35
C03,Carol,Chennai,22
C04,Dave,Bangalore,45
C05,Eve,Hyderabad,31
C06,Frank,Pune,27
"""

df_sales = pd.read_csv(StringIO(sales_csv))
df_customers = pd.read_csv(StringIO(customer_csv))

print(f"[Source 1] Sales Data Extracted: {df_sales.shape[0]} records, {df_sales.shape[1]} attributes")
print(df_sales)

print(f"\n[Source 2] Customer Data Extracted: {df_customers.shape[0]} records, {df_customers.shape[1]} attributes")
print(df_customers)

# ============================================================
# STEP 2: TRANSFORM
# ============================================================
print("\n[STEP 2] TRANSFORM - Cleaning and transforming data")
print("-" * 50)

# 2a. Remove duplicates
before = len(df_sales)
df_sales = df_sales.drop_duplicates()
after = len(df_sales)
print(f"[2a] Duplicates removed: {before - after} records | Remaining: {after}")

# 2b. Handle missing values
print(f"\n[2b] Missing Values Before:\n{df_sales.isnull().sum()}")
df_sales['CustomerID'] = df_sales['CustomerID'].fillna('UNKNOWN')
print(f"Missing values filled with 'UNKNOWN'")

# 2c. Filter only Completed transactions (Business Rule)
df_sales = df_sales[df_sales['Status'] == 'Completed']
print(f"\n[2c] After applying business rule (Completed only): {len(df_sales)} records")

# 2d. Data type conversion
df_sales['Date'] = pd.to_datetime(df_sales['Date'])
df_sales['Amount'] = df_sales['Amount'].astype(float)
print(f"\n[2d] Data types converted:")
print(df_sales.dtypes)

# 2e. Aggregation
agg_data = df_sales.groupby('Product').agg(
    Total_Sales=('Amount', 'sum'),
    Avg_Sales=('Amount', 'mean'),
    Count=('TransactionID', 'count')
).reset_index()
print(f"\n[2e] Aggregated Data by Product:")
print(agg_data)

# 2f. Merge / Join
df_merged = pd.merge(df_sales, df_customers, on='CustomerID', how='left')
print(f"\n[2f] Merged Dataset (Sales + Customer Info):")
print(df_merged[['TransactionID', 'CustomerID', 'Name', 'Product', 'Amount', 'City', 'Date']])

# ============================================================
# STEP 3: LOAD
# ============================================================
print("\n[STEP 3] LOAD - Saving transformed data to data warehouse")
print("-" * 50)

# Fact Table
fact_table = df_merged[['TransactionID', 'CustomerID', 'Product', 'Amount', 'Date']].copy()
fact_table.to_csv('fact_sales.csv', index=False)
print("[Fact Table] fact_sales.csv saved successfully!")
print(fact_table)

# Dimension Table - Customer
dim_customer = df_customers.copy()
dim_customer.to_csv('dim_customer.csv', index=False)
print("\n[Dimension Table] dim_customer.csv saved successfully!")
print(dim_customer)

# Aggregated Summary
agg_data.to_csv('agg_product_sales.csv', index=False)
print("\n[Aggregated Table] agg_product_sales.csv saved successfully!")
print(agg_data)

print("\n" + "=" * 60)
print("  ETL PROCESS COMPLETED SUCCESSFULLY")
print("  Files created: fact_sales.csv, dim_customer.csv, agg_product_sales.csv")
print("=" * 60)
