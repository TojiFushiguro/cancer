# data_cleaning.py
import pandas as pd

def clean_households(filepath: str) -> pd.DataFrame:
    """Clean and standardize households data."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={
        'L': 'LOYALTY_FLAG',
        'MARITAL': 'MARITAL_STATUS',
        'HOMEOWNER': 'HOMEOWNER_DESC',
        'HH_SIZE': 'HSHD_SIZE'
    })
    required = ['HSHD_NUM', 'LOYALTY_FLAG', 'AGE_RANGE', 'MARITAL_STATUS',
                'INCOME_RANGE', 'HOMEOWNER_DESC', 'HSHD_COMPOSITION',
                'HSHD_SIZE', 'CHILDREN']
    for col in required:
        if col not in df.columns:
            df[col] = None
    df['HSHD_NUM'] = pd.to_numeric(df['HSHD_NUM'], errors='coerce')
    df = df.dropna(subset=['HSHD_NUM']).drop_duplicates('HSHD_NUM')
    return df[required]

def clean_products(filepath: str) -> pd.DataFrame:
    """Clean and standardize products data."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={
        'BRAND_TY': 'BRAND_TYPE',
        'NATURAL_ORGANIC': 'NATURAL_ORGANIC_FLAG'
    })
    required = ['PRODUCT_NUM', 'DEPARTMENT', 'COMMODITY', 'BRAND_TYPE', 'NATURAL_ORGANIC_FLAG']
    for col in required:
        if col not in df.columns:
            df[col] = None
    df['PRODUCT_NUM'] = pd.to_numeric(df['PRODUCT_NUM'], errors='coerce')
    df = df.dropna(subset=['PRODUCT_NUM']).drop_duplicates('PRODUCT_NUM')
    return df[required]

def clean_transactions(filepath: str) -> pd.DataFrame:
    """Clean and standardize transactions data."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={
        'PURCHASE_': 'DATE',
        'STORE_R': 'STORE_REGION'
    })
    required = ['HSHD_NUM', 'BASKET_NUM', 'DATE', 'PRODUCT_NUM',
                'SPEND', 'UNITS', 'STORE_REGION', 'WEEK_NUM', 'YEAR']
    for col in required:
        if col not in df.columns:
            df[col] = None
    df['HSHD_NUM'] = pd.to_numeric(df['HSHD_NUM'], errors='coerce')
    df['BASKET_NUM'] = pd.to_numeric(df['BASKET_NUM'], errors='coerce')
    df['PRODUCT_NUM'] = pd.to_numeric(df['PRODUCT_NUM'], errors='coerce')
    df['SPEND'] = pd.to_numeric(df['SPEND'], errors='coerce').fillna(0)
    df['UNITS'] = pd.to_numeric(df['UNITS'], errors='coerce').fillna(0)
    df['WEEK_NUM'] = pd.to_numeric(df['WEEK_NUM'], errors='coerce').fillna(0)
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce').fillna(0)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['HSHD_NUM', 'BASKET_NUM', 'PRODUCT_NUM', 'DATE'])
    return df[required]
