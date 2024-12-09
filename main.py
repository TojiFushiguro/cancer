# main.py

import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from sqlalchemy import text, create_engine
import logging
from werkzeug.utils import secure_filename
from data_cleaning import clean_households, clean_products, clean_transactions
from google.cloud import storage
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from ml_utils import basket_analysis, churn_prediction, clv_prediction
from functools import lru_cache
from contextlib import contextmanager
from ml_utils import perform_basket_analysis, predict_customer_churn


# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')

# File upload configurations
ALLOWED_EXTENSIONS = {'csv'}
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'group21-retail-data')

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Database configuration with connection pooling
DB_USER = os.environ.get('DB_USER', 'root')
DB_PASS = os.environ.get('DB_PASS', 'admin')
DB_NAME = os.environ.get('DB_NAME', 'retaildb')
INSTANCE_CONNECTION_NAME = os.environ.get('INSTANCE_CONNECTION_NAME', 'group21cloud:us-central1:retail-instance')

# Create engine with connection pooling
engine = create_engine(
    f'mysql+pymysql://{DB_USER}:{DB_PASS}@/{DB_NAME}?unix_socket=/cloudsql/{INSTANCE_CONNECTION_NAME}',
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    execution_options={"timeout": 30}
)

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Display the login page"""
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    """Handle user login."""
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')

    if username and password and email:
        session['username'] = username
        logger.info(f"User '{username}' logged in successfully with email '{email}'.")
        return redirect(url_for('search_page'))
    else:
        logger.warning("Login attempt with missing credentials.")
        return "Please provide username, password, and email.", 400

@app.route('/logout')
def logout():
    """Handle user logout by clearing the session."""
    username = session.pop('username', None)
    if username:
        logger.info(f"User '{username}' logged out.")
    return redirect(url_for('index'))

@lru_cache(maxsize=128)
def get_cached_search_results(hshd_num, timestamp):
    """Cache search results for 5 minutes"""
    query = text("""
        SELECT /*+ INDEX(t idx_trans_hshd) */
            h.HSHD_NUM, t.BASKET_NUM, t.DATE, t.PRODUCT_NUM, 
            p.DEPARTMENT, p.COMMODITY, t.SPEND, t.UNITS, 
            t.STORE_REGION, t.WEEK_NUM, t.YEAR
        FROM Transactions t
        JOIN Products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        JOIN Households h ON t.HSHD_NUM = h.HSHD_NUM
        WHERE h.HSHD_NUM = :hshd_num
        ORDER BY h.HSHD_NUM, t.BASKET_NUM, t.DATE, t.PRODUCT_NUM, 
                 p.DEPARTMENT, p.COMMODITY
    """)
    
    with get_db_connection() as conn:
        results = conn.execute(query, {'hshd_num': hshd_num}).fetchall()
    return results

@app.route('/search', methods=['GET', 'POST'])
def search_page():
    """Handle search functionality for household numbers."""
    if 'username' not in session:
        logger.warning("Unauthorized access attempt to search page.")
        return redirect(url_for('index'))
    
    results = None
    if request.method == 'POST':
        hshd_num = request.form.get('hshd_num')
        if hshd_num:
            try:
                hshd_num_int = int(hshd_num)
                # Use cached results with 5-minute timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
                timestamp = timestamp[:-1] + '0'  # Round to nearest 5 minutes
                results = get_cached_search_results(hshd_num_int, timestamp)
                logger.info(f"Fetched {len(results)} records for HSHD_NUM={hshd_num_int}")
            except ValueError:
                logger.error(f"Invalid HSHD_NUM input: {hshd_num}")
                return "Invalid Household Number.", 400
            except Exception as e:
                logger.error(f"Database query failed for HSHD_NUM={hshd_num}: {e}")
                return "An error occurred while fetching data.", 500
    return render_template('search.html', results=results)

@lru_cache(maxsize=128)
def get_cached_dashboard_data(date_key):
    """Cache dashboard data for 1 hour"""
    with get_db_connection() as conn:
        # Basic metrics with optimization hints
        metrics_query = text("""
            SELECT /*+ INDEX(t idx_trans_date) */
                COUNT(DISTINCT t.BASKET_NUM) as total_baskets,
                COUNT(DISTINCT t.HSHD_NUM) as total_households,
                ROUND(AVG(t.SPEND), 2) as avg_transaction,
                ROUND(SUM(t.SPEND), 2) as total_revenue,
                MIN(t.DATE) as start_date,
                MAX(t.DATE) as end_date
            FROM Transactions t
            WHERE t.DATE >= CURDATE() - INTERVAL 365 DAY
        """)

        # Household size impact with partitioning
        household_query = text("""
            WITH household_metrics AS (
                SELECT /*+ INDEX(t idx_trans_hshd) */
                    h.HSHD_SIZE,
                    SUM(t.SPEND) as total_spend,
                    COUNT(DISTINCT t.BASKET_NUM) as basket_count
                FROM Households h
                JOIN Transactions t ON h.HSHD_NUM = t.HSHD_NUM
                WHERE t.DATE >= CURDATE() - INTERVAL 365 DAY
                GROUP BY h.HSHD_SIZE
            )
            SELECT 
                HSHD_SIZE,
                ROUND(AVG(total_spend), 2) as avg_spend,
                SUM(basket_count) as total_baskets
            FROM household_metrics
            GROUP BY HSHD_SIZE
        """)

        # Brand preferences with materialized results
        brand_query = text("""
            WITH brand_metrics AS (
                SELECT /*+ INDEX(p idx_products_brand) */
                    p.BRAND_TYPE,
                    p.NATURAL_ORGANIC_FLAG,
                    SUM(t.SPEND) as total_spend,
                    COUNT(DISTINCT t.BASKET_NUM) as basket_count
                FROM Transactions t
                JOIN Products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                WHERE t.DATE >= CURDATE() - INTERVAL 365 DAY
                GROUP BY p.BRAND_TYPE, p.NATURAL_ORGANIC_FLAG
            )
            SELECT * FROM brand_metrics
        """)

        # Department performance
        dept_query = text("""
            SELECT /*+ INDEX(p idx_products_dept) */
                p.DEPARTMENT,
                EXTRACT(MONTH FROM t.DATE) as month,
                SUM(t.SPEND) as total_spend
            FROM Transactions t
            JOIN Products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
            WHERE t.DATE >= CURDATE() - INTERVAL 365 DAY
            GROUP BY p.DEPARTMENT, EXTRACT(MONTH FROM t.DATE)
            ORDER BY p.DEPARTMENT, month
        """)

        results = {
            'metrics': conn.execute(metrics_query).fetchone(),
            'household': conn.execute(household_query).fetchall(),
            'brand': conn.execute(brand_query).fetchall(),
            'department': conn.execute(dept_query).fetchall()
        }

        # Get ML insights
        basket_results = basket_analysis()
        churn_accuracy, churn_rate, _ = churn_prediction()
        clv_r2, clv_mae, _ = clv_prediction()

        # Combine all results
        dashboard_data = {
            'metrics': {
                'total_baskets': results['metrics'].total_baskets,
                'total_households': results['metrics'].total_households,
                'avg_transaction': float(results['metrics'].avg_transaction),
                'total_revenue': float(results['metrics'].total_revenue),
                'date_range': f"{results['metrics'].start_date.strftime('%Y-%m-%d')} to {results['metrics'].end_date.strftime('%Y-%m-%d')}",
                'churn_rate': churn_rate,
                'churn_accuracy': churn_accuracy,
                'clv_accuracy': clv_r2
            },
            'charts': {
                'household_sizes': [str(r.HSHD_SIZE) for r in results['household']],
                'household_spend': [float(r.avg_spend) for r in results['household']],
                'brand_types': [r.BRAND_TYPE for r in results['brand']],
                'brand_spend': [float(r.total_spend) for r in results['brand']],
                'departments': list(set(r.DEPARTMENT for r in results['department'])),
                'dept_months': list(range(1, 13)),
                'dept_spend': [[float(r.total_spend) for r in results['department'] 
                              if r.DEPARTMENT == dept and r.month == month]
                              for dept in set(r.DEPARTMENT for r in results['department'])
                              for month in range(1, 13)],
                'basket_pairs': [f"{r['product1']}-{r['product2']}" for _, r in basket_results.iterrows()],
                'pair_frequency': [int(r['frequency']) for _, r in basket_results.iterrows()]
            }
        }
        
        return dashboard_data

@app.route('/dashboard')
def dashboard_page():
    """Display the enhanced dashboard with comprehensive retail analytics."""
    if 'username' not in session:
        logger.warning("Unauthorized access attempt to dashboard.")
        return redirect(url_for('index'))

    try:
        with get_db_connection() as conn:
            # Basic metrics query
            metrics_query = text("""
                SELECT 
                    COUNT(DISTINCT t.BASKET_NUM) as total_baskets,
                    COUNT(DISTINCT t.HSHD_NUM) as total_households,
                    IFNULL(ROUND(AVG(t.SPEND), 2), 0.00) as avg_transaction,
                    IFNULL(ROUND(SUM(t.SPEND), 2), 0.00) as total_revenue,
                    DATE(MIN(t.DATE)) as start_date,
                    DATE(MAX(t.DATE)) as end_date
                FROM Transactions t
            """)

            # Weekly sales trend
            weekly_query = text("""
                SELECT 
                    CONCAT('Week ', t.WEEK_NUM, ', ', t.YEAR) as week_label,
                    IFNULL(ROUND(SUM(t.SPEND), 2), 0.00) as weekly_sales
                FROM Transactions t
                GROUP BY t.WEEK_NUM, t.YEAR
                ORDER BY t.YEAR, t.WEEK_NUM
            """)

            # Department sales
            department_query = text("""
                SELECT 
                    IFNULL(p.DEPARTMENT, 'Unknown') as department,
                    IFNULL(ROUND(SUM(t.SPEND), 2), 0.00) as total_sales
                FROM Transactions t
                LEFT JOIN Products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
                GROUP BY p.DEPARTMENT
                ORDER BY total_sales DESC
            """)

            # Region distribution
            region_query = text("""
                SELECT 
                    IFNULL(t.STORE_REGION, 'Unknown') as region,
                    IFNULL(ROUND(SUM(t.SPEND), 2), 0.00) as total_sales
                FROM Transactions t
                GROUP BY t.STORE_REGION
                ORDER BY total_sales DESC
            """)

            try:
                metrics_result = conn.execute(metrics_query).fetchone()
                weekly_results = conn.execute(weekly_query).fetchall()
                department_results = conn.execute(department_query).fetchall()
                region_results = conn.execute(region_query).fetchall()
            except Exception as e:
                logger.error(f"Database query error: {str(e)}")
                return render_template('dashboard.html', 
                                    metrics={'total_baskets': 0, 
                                            'total_households': 0,
                                            'avg_transaction': 0.0,
                                            'total_revenue': 0.0,
                                            'date_range': 'No data available'},
                                    charts={})

            def safe_float(value, default=0.0):
                if value is None:
                    return default
                try:
                    return float(value)
                except:
                    return default

            def safe_int(value, default=0):
                if value is None:
                    return default
                try:
                    return int(value)
                except:
                    return default

            # Prepare metrics
            metrics = {
                'total_baskets': safe_int(getattr(metrics_result, 'total_baskets', 0)),
                'total_households': safe_int(getattr(metrics_result, 'total_households', 0)),
                'avg_transaction': safe_float(getattr(metrics_result, 'avg_transaction', 0)),
                'total_revenue': safe_float(getattr(metrics_result, 'total_revenue', 0)),
                'date_range': 'No data available'
            }

            if metrics_result and metrics_result.start_date and metrics_result.end_date:
                metrics['date_range'] = f"{metrics_result.start_date} to {metrics_result.end_date}"

            # Prepare charts data
            charts = {
                # Weekly trends
                'weekly_dates': [str(row.week_label) for row in weekly_results] if weekly_results else [],
                'weekly_sales': [safe_float(row.weekly_sales) for row in weekly_results] if weekly_results else [],

                # Department distribution
                'department_labels': [str(row.department) for row in department_results] if department_results else [],
                'department_values': [safe_float(row.total_sales) for row in department_results] if department_results else [],

                # Region distribution
                'region_labels': [str(row.region) for row in region_results] if region_results else [],
                'region_values': [safe_float(row.total_sales) for row in region_results] if region_results else [],

                # Additional required fields with defaults
                'top_products_labels': [],
                'top_products_values': [],
                'basket_pairs': [],
                'pair_frequency': []
            }

            return render_template('dashboard.html', metrics=metrics, charts=charts)

    except Exception as e:
        logger.error(f"Dashboard generation error: {str(e)}")
        # Return empty dashboard
        return render_template('dashboard.html', 
                             metrics={'total_baskets': 0, 
                                     'total_households': 0,
                                     'avg_transaction': 0.0,
                                     'total_revenue': 0.0,
                                     'date_range': 'Error loading data'},
                             charts={'weekly_dates': [],
                                    'weekly_sales': [],
                                    'department_labels': [],
                                    'department_values': [],
                                    'region_labels': [],
                                    'region_values': [],
                                    'top_products_labels': [],
                                    'top_products_values': [],
                                    'basket_pairs': [],
                                    'pair_frequency': []})
@app.route('/analytics')
def analytics_page():
    """Display ML insights and analytics."""
    if 'username' not in session:
        logger.warning("Unauthorized access attempt to analytics.")
        return redirect(url_for('index'))

    try:
        logger.info("Starting analytics page generation")
        
        # Get ML insights
        logger.info("Starting basket analysis")
        basket_insights = perform_basket_analysis()
        logger.info("Basket analysis completed")
        
        logger.info("Starting churn prediction")
        churn_insights = predict_customer_churn()
        logger.info("Churn prediction completed")

        # Prepare default structure
        analytics_data = {
            'basket_analysis': {
                'recommendations': [],
                'product_pairs': []
            },
            'churn_prediction': {
                'metrics': {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'churn_rate': 0.0
                },
                'risk_levels': {
                    'high_risk': 0,
                    'medium_risk': 0,
                    'low_risk': 0
                },
                'feature_importance': {}
            }
        }

        # Update with actual results if available
        logger.info("Updating basket insights")
        if basket_insights and 'recommendations' in basket_insights:
            analytics_data['basket_analysis']['recommendations'] = basket_insights['recommendations']
            analytics_data['basket_analysis']['product_pairs'] = basket_insights['product_pairs']

        logger.info("Updating churn insights")
        if churn_insights:
            if 'metrics' in churn_insights:
                analytics_data['churn_prediction']['metrics'] = churn_insights['metrics']
            if 'customer_risk' in churn_insights:
                analytics_data['churn_prediction']['risk_levels'] = churn_insights['customer_risk']
            if 'feature_importance' in churn_insights:
                analytics_data['churn_prediction']['feature_importance'] = churn_insights['feature_importance']

        logger.info("Analytics data prepared successfully")
        return render_template('analytics.html', data=analytics_data)

    except Exception as e:
        logger.error(f"Analytics generation error: {str(e)}")
        logger.exception("Full traceback:")
        return render_template('analytics.html', 
                             data={
                                 'basket_analysis': {
                                     'recommendations': [],
                                     'product_pairs': []
                                 },
                                 'churn_prediction': {
                                     'metrics': {
                                         'accuracy': 0.0,
                                         'precision': 0.0,
                                         'recall': 0.0,
                                         'f1_score': 0.0,
                                         'churn_rate': 0.0
                                     },
                                     'risk_levels': {
                                         'high_risk': 0,
                                         'medium_risk': 0,
                                         'low_risk': 0
                                     },
                                     'feature_importance': {}
                                 }
                             })
@app.route('/upload-page')
def upload_page():
    """Display the upload page"""
    if 'username' not in session:
        logger.warning("Unauthorized access attempt to upload page.")
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads using Google Cloud Storage"""
    if 'username' not in session:
        logger.warning("Unauthorized upload attempt.")
        return jsonify({'error': 'Unauthorized'}), 401

    temp_files = {}
    try:
        if not all(x in request.files for x in ['households', 'products', 'transactions']):
            logger.error("Missing required files in upload request")
            return jsonify({'error': 'Missing required files'}), 400

        bucket = storage_client.bucket(BUCKET_NAME)

        # Process each file
        for file_type in ['households', 'products', 'transactions']:
            file = request.files[file_type]
            if file and allowed_file(file.filename):
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    file.save(temp_file.name)
                    temp_files[file_type] = temp_file.name

                    blob = bucket.blob(f"{file_type}/{secure_filename(file.filename)}")
                    blob.upload_from_filename(temp_file.name)
                    logger.info(f"Uploaded {file_type} file to Cloud Storage")

        # Process the files in batches
        logger.info("Starting data cleaning process...")
        households_df = clean_households(temp_files['households'])
        products_df = clean_products(temp_files['products'])
        transactions_df = clean_transactions(temp_files['transactions'])

        # Load the cleaned data into the database
        with get_db_connection() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS=0;"))
            conn.execute(text("TRUNCATE TABLE Transactions;"))
            conn.execute(text("TRUNCATE TABLE Products;"))
            conn.execute(text("TRUNCATE TABLE Households;"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS=1;"))
            
            # Use chunked loading for better memory management
            chunk_size = 5000
            for table, df in [('Households', households_df), 
                            ('Products', products_df), 
                            ('Transactions', transactions_df)]:
                for i in range(0, len(df), chunk_size):
                    chunk = df[i:i + chunk_size]
                    chunk.to_sql(table, con=engine, if_exists='append', index=False)
                    logger.info(f"Loaded chunk {i//chunk_size + 1} of {table}")

        # Clear all caches after data upload
        get_cached_search_results.cache_clear()
        get_cached_dashboard_data.cache_clear()

        # Clean up temporary files
        for temp_file in temp_files.values():
            os.unlink(temp_file)

        return jsonify({'message': 'Files processed and loaded successfully'}), 200

    except Exception as e:
        logger.error(f"Error processing uploads: {str(e)}")
        # Clean up temporary files in case of error
        for temp_file in temp_files.values():
            try:
                os.unlink(temp_file)
            except:
                pass
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    logger.error(f"404 error: {request.url}")
    return "Page not found.", 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logger.error(f"500 error: {str(e)}")
    return "Internal server error.", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))