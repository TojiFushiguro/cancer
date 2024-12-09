# ml_utils.py

import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from db_utils import engine
import warnings
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from contextlib import contextmanager

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

@lru_cache(maxsize=32)
def basket_analysis(cache_key=None):
    """
    Optimized market basket analysis with caching.
    Args:
        cache_key: String to force cache invalidation (e.g., current date)
    """
    query = text("""
        WITH RECURSIVE 
        filtered_transactions AS (
            SELECT /*+ INDEX(t idx_trans_basket) */
                BASKET_NUM, 
                PRODUCT_NUM
            FROM Transactions t
            WHERE DATE >= CURDATE() - INTERVAL 90 DAY
        ),
        product_pairs AS (
            SELECT 
                t1.BASKET_NUM,
                t1.PRODUCT_NUM as product1,
                t2.PRODUCT_NUM as product2
            FROM filtered_transactions t1
            INNER JOIN filtered_transactions t2 
                ON t1.BASKET_NUM = t2.BASKET_NUM
                AND t1.PRODUCT_NUM < t2.PRODUCT_NUM
        ),
        pair_counts AS (
            SELECT 
                product1,
                product2,
                COUNT(*) as frequency
            FROM product_pairs
            GROUP BY product1, product2
            HAVING frequency > 10
        )
        SELECT 
            pc.product1,
            pc.product2,
            pc.frequency,
            p1.DEPARTMENT as dept1,
            p2.DEPARTMENT as dept2,
            p1.COMMODITY as comm1,
            p2.COMMODITY as comm2
        FROM pair_counts pc
        JOIN Products p1 ON pc.product1 = p1.PRODUCT_NUM
        JOIN Products p2 ON pc.product2 = p2.PRODUCT_NUM
        ORDER BY pc.frequency DESC
        LIMIT 20
    """)
    
    try:
        logger.info("Executing optimized basket analysis query...")
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn)
            
        # Calculate support and confidence
        total_baskets_query = text("""
            SELECT /*+ INDEX(t idx_trans_date) */
                COUNT(DISTINCT BASKET_NUM) as total 
            FROM Transactions t
            WHERE DATE >= CURDATE() - INTERVAL 90 DAY
        """)
        
        with get_db_connection() as conn:
            total_baskets = conn.execute(total_baskets_query).fetchone()[0]
            
        df['support'] = df['frequency'] / total_baskets
        df['pair_label'] = df.apply(lambda x: f"{x['comm1']} & {x['comm2']}", axis=1)
        
        logger.info(f"Basket Analysis: Found {len(df)} significant product pairs.")
        return df
        
    except Exception as e:
        logger.error(f"Basket Analysis failed: {e}")
        return pd.DataFrame()

@lru_cache(maxsize=32)
def churn_prediction(cache_key=None):
    """
    Optimized churn prediction with caching.
    Args:
        cache_key: String to force cache invalidation (e.g., current date)
    """
    warnings.filterwarnings('ignore')
    
    query = text("""
        WITH customer_metrics AS (
            SELECT /*+ INDEX(t idx_trans_hshd) */
                h.HSHD_NUM,
                h.HSHD_SIZE,
                h.CHILDREN,
                h.INCOME_RANGE,
                MAX(t.DATE) as last_purchase,
                COUNT(DISTINCT t.BASKET_NUM) as total_baskets,
                SUM(t.SPEND) as total_spend,
                AVG(t.SPEND) as avg_spend,
                STDDEV(t.SPEND) as spend_stddev,
                COUNT(DISTINCT p.DEPARTMENT) as unique_departments,
                SUM(CASE WHEN p.BRAND_TYPE = 'Private' THEN t.SPEND ELSE 0 END) as private_brand_spend,
                COUNT(DISTINCT CASE WHEN p.NATURAL_ORGANIC_FLAG = 1 THEN t.PRODUCT_NUM END) as organic_products
            FROM Households h
            JOIN Transactions t ON h.HSHD_NUM = t.HSHD_NUM
            JOIN Products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
            WHERE t.DATE >= CURDATE() - INTERVAL 365 DAY
            GROUP BY h.HSHD_NUM, h.HSHD_SIZE, h.CHILDREN, h.INCOME_RANGE
        )
        SELECT * FROM customer_metrics
    """)
    
    try:
        logger.info("Executing optimized churn prediction query...")
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty:
            logger.warning("Churn Prediction: No data available.")
            return "N/A", "N/A", {}

        # Feature engineering
        df['last_purchase'] = pd.to_datetime(df['last_purchase'])
        last_date = df['last_purchase'].max()
        df['days_since_purchase'] = (last_date - df['last_purchase']).dt.days
        df['purchase_frequency'] = df['total_baskets'] / np.maximum(df['days_since_purchase'], 1)
        df['spend_per_day'] = df['total_spend'] / np.maximum(df['days_since_purchase'], 1)
        df['organic_ratio'] = df['organic_products'] / np.maximum(df['total_baskets'], 1)
        df['private_brand_ratio'] = df['private_brand_spend'] / np.maximum(df['total_spend'], 1)
        
        # Handle categorical variables
        le = LabelEncoder()
        df['INCOME_RANGE_encoded'] = le.fit_transform(df['INCOME_RANGE'])
        
        # Define churn (no purchase in last 90 days)
        df['churned'] = (df['days_since_purchase'] > 90).astype(int)
        
        # Prepare features
        features = [
            'total_baskets', 'total_spend', 'avg_spend', 'spend_stddev',
            'unique_departments', 'purchase_frequency', 'spend_per_day',
            'HSHD_SIZE', 'CHILDREN', 'INCOME_RANGE_encoded',
            'organic_ratio', 'private_brand_ratio'
        ]
        
        X = df[features].fillna(0)
        y = df['churned']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        if len(set(y)) < 2:
            logger.warning("Churn Prediction: Insufficient class variation.")
            return "N/A", "N/A", {}
            
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        churn_rate = y.mean()
        
        # Get feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        
        logger.info(f"Churn Prediction: Accuracy={accuracy:.2f}, Churn Rate={churn_rate:.2f}")
        return accuracy, churn_rate, feature_importance
        
    except Exception as e:
        logger.error(f"Churn Prediction failed: {e}")
        return "N/A", "N/A", {}

@lru_cache(maxsize=32)
def clv_prediction(cache_key=None):
    """
    Optimized Customer Lifetime Value prediction with caching.
    Args:
        cache_key: String to force cache invalidation (e.g., current date)
    """
    warnings.filterwarnings('ignore')
    
    query = text("""
        WITH customer_features AS (
            SELECT /*+ INDEX(t idx_trans_hshd) */
                h.HSHD_NUM,
                h.HSHD_SIZE,
                h.CHILDREN,
                h.INCOME_RANGE,
                h.MARITAL_STATUS,
                h.HOMEOWNER_DESC,
                COUNT(DISTINCT t.BASKET_NUM) as total_baskets,
                SUM(t.SPEND) as total_spend,
                AVG(t.SPEND) as avg_spend,
                COUNT(DISTINCT p.DEPARTMENT) as unique_departments,
                SUM(CASE WHEN p.BRAND_TYPE = 'Private' THEN t.SPEND ELSE 0 END) as private_brand_spend,
                COUNT(DISTINCT CASE WHEN p.NATURAL_ORGANIC_FLAG = 1 THEN t.PRODUCT_NUM END) as organic_products,
                COUNT(DISTINCT t.STORE_REGION) as unique_stores
            FROM Households h
            JOIN Transactions t ON h.HSHD_NUM = t.HSHD_NUM
            JOIN Products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
            WHERE t.DATE >= CURDATE() - INTERVAL 365 DAY
            GROUP BY 
                h.HSHD_NUM, h.HSHD_SIZE, h.CHILDREN, h.INCOME_RANGE,
                h.MARITAL_STATUS, h.HOMEOWNER_DESC
        )
        SELECT * FROM customer_features
    """)
    
    try:
        logger.info("Executing optimized CLV prediction query...")
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty:
            logger.warning("CLV Prediction: No data available.")
            return "N/A", "N/A", None
            
        # Feature engineering
        le = LabelEncoder()
        df['INCOME_RANGE_encoded'] = le.fit_transform(df['INCOME_RANGE'])
        df['MARITAL_STATUS_encoded'] = le.fit_transform(df['MARITAL_STATUS'])
        df['HOMEOWNER_DESC_encoded'] = le.fit_transform(df['HOMEOWNER_DESC'])
        
        df['private_brand_ratio'] = df['private_brand_spend'] / np.maximum(df['total_spend'], 1)
        df['organic_ratio'] = df['organic_products'] / np.maximum(df['total_baskets'], 1)
        df['avg_basket_size'] = df['total_spend'] / np.maximum(df['total_baskets'], 1)
        
        # Define CLV
        df['CLV'] = df['total_spend']
        
        # Prepare features
        features = [
            'HSHD_SIZE', 'CHILDREN', 'INCOME_RANGE_encoded',
            'MARITAL_STATUS_encoded', 'HOMEOWNER_DESC_encoded',
            'total_baskets', 'avg_spend', 'unique_departments',
            'private_brand_ratio', 'organic_ratio', 'avg_basket_size',
            'unique_stores'
        ]
        
        X = df[features].fillna(0)
        y = df['CLV']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Calculate metrics
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Get feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        
        logger.info(f"CLV Prediction: R2={r2:.2f}, MAE=${mae:.2f}")
        return r2, mae, feature_importance

    except Exception as e:
        logger.error(f"CLV Prediction failed: {e}")
        return "N/A", "N/A", None

def get_latest_cache_key():
    """Generate a cache key based on the current hour"""
    return datetime.now().strftime("%Y-%m-%d-%H")

def clear_all_caches():
    """Clear all function caches"""
    basket_analysis.cache_clear()
    churn_prediction.cache_clear()
    clv_prediction.cache_clear()

def perform_basket_analysis():
    """
    Perform market basket analysis using Gradient Boosting to identify product combinations
    and cross-selling opportunities.
    """
    try:
        # Get transaction data
        query = text("""
            SELECT 
                t.BASKET_NUM,
                t.PRODUCT_NUM,
                p.DEPARTMENT,
                p.COMMODITY,
                SUM(t.SPEND) as total_spend,
                COUNT(*) as purchase_frequency
            FROM Transactions t
            JOIN Products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
            GROUP BY t.BASKET_NUM, t.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty:
            logger.warning("No transaction data available for basket analysis")
            return None
            
        # Create basket matrix
        basket_matrix = pd.pivot_table(
            df,
            index='BASKET_NUM',
            columns=['DEPARTMENT', 'COMMODITY'],
            values='purchase_frequency',
            aggfunc='sum',
            fill_value=0
        )
        
        # Convert to binary (bought/not bought)
        basket_matrix = (basket_matrix > 0).astype(int)
        
        # Find frequent itemsets using Apriori
        frequent_itemsets = apriori(
            basket_matrix, 
            min_support=0.01,  # Items appearing in at least 1% of baskets
            use_colnames=True
        )
        
        # Generate association rules
        rules = association_rules(
            frequent_itemsets, 
            metric="lift",
            min_threshold=1.0  # Only rules with lift > 1 (positive correlation)
        )
        
        # Sort rules by lift and confidence
        rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
        
        # Prepare recommendations
        top_recommendations = []
        for _, rule in rules.head(10).iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            lift = rule['lift']
            confidence = rule['confidence']
            
            rec = {
                'antecedents': [str(x) for x in antecedents],
                'consequents': [str(x) for x in consequents],
                'lift': float(lift),
                'confidence': float(confidence)
            }
            top_recommendations.append(rec)
        
        # Calculate support for product pairs
        product_pairs = []
        for _, rule in rules.head(20).iterrows():
            pair = {
                'product1': str(list(rule['antecedents'])[0]),
                'product2': str(list(rule['consequents'])[0]),
                'support': float(rule['support']),
                'confidence': float(rule['confidence'])
            }
            product_pairs.append(pair)
            
        return {
            'recommendations': top_recommendations,
            'product_pairs': product_pairs
        }
        
    except Exception as e:
        logger.error(f"Basket analysis failed: {str(e)}")
        return None

def predict_customer_churn():
    """
    Predict customer churn using Random Forest Classifier.
    """
    try:
        # Get customer behavior data
        query = text("""
            WITH customer_metrics AS (
                SELECT 
                    h.HSHD_NUM,
                    h.HSHD_SIZE,
                    h.CHILDREN,
                    MAX(t.DATE) as last_purchase,
                    COUNT(DISTINCT t.BASKET_NUM) as total_baskets,
                    AVG(t.SPEND) as avg_spend,
                    SUM(t.SPEND) as total_spend,
                    COUNT(DISTINCT p.DEPARTMENT) as unique_departments,
                    MAX(t.DATE) - MIN(t.DATE) as customer_tenure
                FROM Households h
                JOIN Transactions t ON h.HSHD_NUM = t.HSHD_NUM
                JOIN Products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                GROUP BY h.HSHD_NUM, h.HSHD_SIZE, h.CHILDREN
            )
            SELECT * FROM customer_metrics
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        if df.empty:
            logger.warning("No customer data available for churn prediction")
            return None
            
        # Feature engineering
        df['last_purchase'] = pd.to_datetime(df['last_purchase'])
        current_date = df['last_purchase'].max()
        df['days_since_purchase'] = (current_date - df['last_purchase']).dt.days
        
        # Define churn (no purchase in last 90 days)
        df['churned'] = (df['days_since_purchase'] > 90).astype(int)
        
        # Calculate additional features
        df['purchase_frequency'] = df['total_baskets'] / df['customer_tenure']
        df['avg_basket_size'] = df['total_spend'] / df['total_baskets']
        
        # Prepare features
        features = [
            'HSHD_SIZE', 'CHILDREN', 'total_baskets', 'avg_spend',
            'unique_departments', 'purchase_frequency', 'avg_basket_size',
            'days_since_purchase'
        ]
        
        X = df[features].fillna(0)
        y = df['churned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Get feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        
        # Identify high-risk customers
        df['churn_probability'] = 0.0
        all_scaled = scaler.transform(X)
        df['churn_probability'] = model.predict_proba(all_scaled)[:, 1]
        
        high_risk_customers = df[df['churn_probability'] > 0.7].shape[0]
        churn_rate = df['churned'].mean()
        
        return {
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'churn_rate': float(churn_rate),
                'high_risk_customers': int(high_risk_customers)
            },
            'feature_importance': feature_importance,
            'customer_risk': {
                'high_risk': int(df[df['churn_probability'] > 0.7].shape[0]),
                'medium_risk': int(df[(df['churn_probability'] > 0.4) & (df['churn_probability'] <= 0.7)].shape[0]),
                'low_risk': int(df[df['churn_probability'] <= 0.4].shape[0])
            }
        }
        
    except Exception as e:
        logger.error(f"Churn prediction failed: {str(e)}")
        return None



