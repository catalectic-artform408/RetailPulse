# %% [markdown]
# # 01 — Data Loading & Cleaning
# Load the 9 Olist CSV files into a SQLite database, fix data types,
# audit quality issues, and document every cleaning decision.

# %%
import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "olist_data"
DB_PATH = PROJECT_ROOT / "data" / "olist.db"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"

os.makedirs(DB_PATH.parent, exist_ok=True)

# %% [markdown]
# ## 1. Load Raw CSVs

# %%
# CSV file → table name mapping
FILE_TABLE_MAP = {
    "olist_orders_dataset.csv":             "orders",
    "olist_customers_dataset.csv":          "customers",
    "olist_order_items_dataset.csv":        "order_items",
    "olist_order_payments_dataset.csv":     "order_payments",
    "olist_order_reviews_dataset.csv":      "order_reviews",
    "olist_products_dataset.csv":           "products",
    "olist_sellers_dataset.csv":            "sellers",
    "olist_geolocation_dataset.csv":        "geolocation",
    "product_category_name_translation.csv":"category_translation",
}

# Read all CSVs into a dict of DataFrames
dfs = {}
for csv_file, table_name in FILE_TABLE_MAP.items():
    path = DATA_DIR / csv_file
    dfs[table_name] = pd.read_csv(path)
    print(f"✓ {table_name:25s} → {len(dfs[table_name]):>8,} rows, {dfs[table_name].shape[1]} cols")

# %% [markdown]
# ## 2. Data Type Fixes
# Parse datetime columns and enforce correct types.

# %%
# --- Orders: parse all timestamp columns ---
datetime_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for col in datetime_cols:
    dfs["orders"][col] = pd.to_datetime(dfs["orders"][col], errors="coerce")

# --- Reviews: parse date columns ---
dfs["order_reviews"]["review_creation_date"] = pd.to_datetime(
    dfs["order_reviews"]["review_creation_date"], errors="coerce"
)
dfs["order_reviews"]["review_answer_timestamp"] = pd.to_datetime(
    dfs["order_reviews"]["review_answer_timestamp"], errors="coerce"
)

# --- Order Items: parse shipping_limit_date ---
dfs["order_items"]["shipping_limit_date"] = pd.to_datetime(
    dfs["order_items"]["shipping_limit_date"], errors="coerce"
)

# --- Numeric columns: ensure correct types ---
dfs["order_items"]["price"] = pd.to_numeric(dfs["order_items"]["price"], errors="coerce")
dfs["order_items"]["freight_value"] = pd.to_numeric(dfs["order_items"]["freight_value"], errors="coerce")
dfs["order_payments"]["payment_value"] = pd.to_numeric(dfs["order_payments"]["payment_value"], errors="coerce")

print("✓ All datetime and numeric columns parsed")

# %% [markdown]
# ## 3. Data Quality Audit
# Document every issue found — this is what impresses interviewers.

# %%
print("=" * 70)
print("DATA QUALITY AUDIT REPORT")
print("=" * 70)

# --- Issue 1: Null delivery timestamps ---
orders = dfs["orders"]
null_carrier = orders["order_delivered_carrier_date"].isna().sum()
null_customer = orders["order_delivered_customer_date"].isna().sum()
total_orders = len(orders)
print(f"\n🔍 Issue 1: Null Delivery Timestamps")
print(f"   Delivered to carrier  : {null_carrier:,} nulls ({null_carrier/total_orders*100:.1f}%)")
print(f"   Delivered to customer : {null_customer:,} nulls ({null_customer/total_orders*100:.1f}%)")
print(f"   → Decision: Keep nulls (orders may be in-transit or cancelled). "
      f"Filter when computing delivery metrics.")

# --- Issue 2: Missing review comments ---
reviews = dfs["order_reviews"]
null_comments = reviews["review_comment_message"].isna().sum()
print(f"\n🔍 Issue 2: Missing Review Comments")
print(f"   Null comment messages : {null_comments:,} ({null_comments/len(reviews)*100:.1f}%)")
print(f"   → Decision: Keep nulls. Use review_score as primary metric; "
      f"comments are supplementary for NLP analysis.")

# --- Issue 3: Zero-value payments ---
payments = dfs["order_payments"]
zero_pay = (payments["payment_value"] == 0).sum()
print(f"\n🔍 Issue 3: Zero-value Payments")
print(f"   payment_value = 0     : {zero_pay:,} rows ({zero_pay/len(payments)*100:.2f}%)")
print(f"   → Decision: Keep these — they represent voucher-only payments "
      f"or initial installment records. They are valid business data.")

# --- Issue 4: Delivery before purchase (anomalous dates) ---
delivered = orders.dropna(subset=["order_delivered_customer_date"])
bad_dates = (delivered["order_delivered_customer_date"] < delivered["order_purchase_timestamp"]).sum()
print(f"\n🔍 Issue 4: Delivery Date Before Purchase Date")
print(f"   Anomalous records     : {bad_dates:,}")
if bad_dates > 0:
    print(f"   → Decision: Flag these rows but keep them. They likely represent "
          f"data entry errors. Exclude from delivery time calculations.")

# --- Issue 5: Multiple payment rows per order ---
multi_pay = payments.groupby("order_id").size()
multi_pay_count = (multi_pay > 1).sum()
print(f"\n🔍 Issue 5: Multiple Payment Rows Per Order")
print(f"   Orders with >1 payment: {multi_pay_count:,} ({multi_pay_count/total_orders*100:.1f}%)")
print(f"   Max payments per order: {multi_pay.max()}")
print(f"   → Decision: Normal — represents split payments (voucher + credit card). "
      f"Aggregate by order_id when computing total payment.")

# --- Issue 6: Duplicate geolocation entries ---
geo = dfs["geolocation"]
geo_dupes = geo.duplicated(subset=["geolocation_zip_code_prefix"]).sum()
print(f"\n🔍 Issue 6: Duplicate Geolocation Entries")
print(f"   Duplicate zip prefixes: {geo_dupes:,}")
print(f"   → Decision: Deduplicate by taking the first occurrence per zip prefix.")

# Deduplicate geolocation
dfs["geolocation"] = dfs["geolocation"].drop_duplicates(
    subset=["geolocation_zip_code_prefix"], keep="first"
).reset_index(drop=True)

# --- Issue 7: Null product categories ---
null_cats = dfs["products"]["product_category_name"].isna().sum()
print(f"\n🔍 Issue 7: Null Product Categories")
print(f"   Products without category: {null_cats:,}")
print(f"   → Decision: Fill with 'unknown' to prevent join failures.")

dfs["products"]["product_category_name"] = dfs["products"]["product_category_name"].fillna("unknown")

# --- Summary ---
print(f"\n{'=' * 70}")
print(f"CLEANING SUMMARY")
print(f"{'=' * 70}")
print(f"  • Parsed 5 datetime columns in orders, 2 in reviews, 1 in order_items")
print(f"  • Deduplicated geolocation table: {len(geo):,} → {len(dfs['geolocation']):,} rows")
print(f"  • Filled {null_cats} null product categories with 'unknown'")
print(f"  • Flagged {bad_dates} anomalous delivery-before-purchase records")
print(f"  • Retained {zero_pay} zero-value payment rows (valid business data)")
print(f"  • All other nulls preserved with documented reasoning")

# %% [markdown]
# ## 4. Load into SQLite Database

# %%
# Remove existing DB for a clean load
if DB_PATH.exists():
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)

# Execute schema DDL
with open(SCHEMA_PATH, "r") as f:
    conn.executescript(f.read())
print("✓ Schema created")

# Load DataFrames into tables
for table_name, df in dfs.items():
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"  ✓ {table_name:25s} → {row_count:>8,} rows loaded")

conn.close()
print(f"\n✓ Database saved to: {DB_PATH}")

# %% [markdown]
# ## 5. Verification

# %%
conn = sqlite3.connect(DB_PATH)
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()
print(f"\n✓ {len(tables)} tables in database:")
for t in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
    print(f"  {t[0]:30s} {count:>8,} rows")
conn.close()
