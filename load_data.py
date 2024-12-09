import pandas as pd
from sqlalchemy import create_engine, text
import os

# Database connection details
DB_USER = "root"
DB_PASS = "admin" 
DB_NAME = "retaildb"
DB_HOST = "127.0.0.1"  # Use localhost when using Cloud SQL Auth Proxy
DB_PORT = 3306

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Paths to CSV files
HOUSEHOLDS_CSV = "data/households.csv"
PRODUCTS_CSV = "data/products.csv"
TRANSACTIONS_CSV = "data/transactions.csv"

def clean_households(filepath):
    df = pd.read_csv(filepath)
    original_columns = df.columns.tolist()
    print(f"Original Households Columns: {original_columns}")

    # Standardize column names: lowercase and replace spaces with underscores
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    standardized_columns = df.columns.tolist()
    print(f"Standardized Households Columns: {standardized_columns}")

    # Rename columns to match database schema
    rename_map = {
        'l': 'loyalty_flag',
        'marital': 'marital_status',
        'homeowner': 'homeowner_desc',
        'hh_size': 'hshd_size'
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure all required columns are present
    required_columns = ['hshd_num', 'loyalty_flag', 'age_range', 'marital_status',
                        'income_range', 'homeowner_desc', 'hshd_composition',
                        'hshd_size', 'children']
    for col in required_columns:
        if col not in df.columns:
            print(f"'{col}' column not found in households. Adding a placeholder column.")
            if col in ['hshd_size', 'children']:
                df[col] = 0
            else:
                df[col] = "Unknown"

    # Remove duplicates based on primary key
    df = df.drop_duplicates(subset=['hshd_num'], keep='first')

    # Clean column data types
    df['hshd_size'] = pd.to_numeric(df['hshd_size'], errors='coerce').fillna(0).astype(int)
    df['children'] = pd.to_numeric(df['children'], errors='coerce').fillna(0).astype(int)

    # Optional: Fill empty strings with NULL
    df.replace({'': None}, inplace=True)

    # Log non-null counts for each column
    print("Households Data Non-Null Counts:")
    print(df.notnull().sum())

    return df

def clean_products(filepath):
    df = pd.read_csv(filepath)
    original_columns = df.columns.tolist()
    print(f"Original Products Columns: {original_columns}")

    # Standardize column names: lowercase and replace spaces with underscores
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    standardized_columns = df.columns.tolist()
    print(f"Standardized Products Columns: {standardized_columns}")

    # Rename columns to match database schema
    rename_map = {
        'brand_ty': 'brand_type'
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure all required columns are present
    required_columns = ['product_num', 'department', 'commodity', 'brand_type', 'natural_organic_flag']
    for col in required_columns:
        if col not in df.columns:
            print(f"'{col}' column not found in products. Adding a placeholder column.")
            if col == 'natural_organic_flag':
                df[col] = False
            else:
                df[col] = "Unknown"

    # Remove duplicates based on primary key
    df = df.drop_duplicates(subset=['product_num'], keep='first')

    # Optional: Fill empty strings with NULL
    df.replace({'': None}, inplace=True)

    # Log non-null counts for each column
    print("Products Data Non-Null Counts:")
    print(df.notnull().sum())

    return df

def clean_transactions(filepath):
    df = pd.read_csv(filepath)
    original_columns = df.columns.tolist()
    print(f"Original Transactions Columns: {original_columns}")

    # Standardize column names: lowercase and replace spaces with underscores
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    standardized_columns = df.columns.tolist()
    print(f"Standardized Transactions Columns: {standardized_columns}")

    # Rename columns to match database schema
    rename_map = {
        'purchase_': 'date',
        'store_r': 'store_region'
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure all required columns are present
    required_columns = ['hshd_num', 'basket_num', 'date', 'product_num', 'spend', 'units',
                        'store_region', 'week_num', 'year']
    for col in required_columns:
        if col not in df.columns:
            print(f"'{col}' column not found in transactions. Adding a placeholder column.")
            if col in ['spend']:
                df[col] = 0.0
            elif col in ['units', 'basket_num', 'week_num', 'year']:
                df[col] = 0
            elif col == 'date':
                df[col] = pd.NaT
            else:
                df[col] = "Unknown"

    # Remove duplicates based on primary key (assuming combination is unique)
    df = df.drop_duplicates(subset=['hshd_num', 'basket_num', 'date', 'product_num'], keep='first')

    # Clean column data types
    # Allow pandas to infer the date format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0.0)
    df['units'] = pd.to_numeric(df['units'], errors='coerce').fillna(0).astype(int)
    df['week_num'] = pd.to_numeric(df['week_num'], errors='coerce').fillna(0).astype(int)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    # Optional: Fill empty strings with NULL
    df.replace({'': None}, inplace=True)

    # Log non-null counts for each column
    print("Transactions Data Non-Null Counts:")
    print(df.notnull().sum())

    # Log sample data for debugging
    print("Sample Transactions Data:")
    print(df[['hshd_num', 'basket_num', 'date', 'product_num', 'spend', 'units', 'store_region', 'week_num', 'year']].head())

    return df

def load_csv_data():
    # Clean data
    households_df = clean_households(HOUSEHOLDS_CSV)
    products_df = clean_products(PRODUCTS_CSV)
    transactions_df = clean_transactions(TRANSACTIONS_CSV)

    with engine.connect() as conn:
        try:
            # Disable foreign key checks to allow truncating tables
            conn.execute(text("SET FOREIGN_KEY_CHECKS=0;"))

            # Truncate tables in child-to-parent order
            print("Truncating Transactions table...")
            conn.execute(text("TRUNCATE TABLE Transactions;"))
            print("Truncating Products table...")
            conn.execute(text("TRUNCATE TABLE Products;"))
            print("Truncating Households table...")
            conn.execute(text("TRUNCATE TABLE Households;"))

            # Re-enable foreign key checks
            conn.execute(text("SET FOREIGN_KEY_CHECKS=1;"))

            # Load data in parent-to-child order
            print("Loading Households data...")
            households_df.to_sql('Households', con=engine, if_exists='append', index=False, method='multi', chunksize=1000)
            print("Households data loaded successfully.")

            print("Loading Products data...")
            products_df.to_sql('Products', con=engine, if_exists='append', index=False, method='multi', chunksize=1000)
            print("Products data loaded successfully.")

            print("Loading Transactions data...")
            transactions_df.to_sql('Transactions', con=engine, if_exists='append', index=False, method='multi', chunksize=1000)
            print("Transactions data loaded successfully.")

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    print("All data loaded successfully into MySQL on GCP.")

if __name__ == "__main__":
    # Ensure the script is run from the project's root directory
    load_csv_data()
