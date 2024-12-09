# db_utils.py

from sqlalchemy import create_engine
import os

DB_USER = os.environ.get('DB_USER', 'root')
DB_PASS = os.environ.get('DB_PASS', 'admin')  # Replace with your actual password
DB_NAME = os.environ.get('DB_NAME', 'retaildb')
INSTANCE_CONNECTION_NAME = os.environ.get('INSTANCE_CONNECTION_NAME', 'group21cloud:us-central1:retail-instance')  # Replace with your instance details

# Create the SQLAlchemy engine with a connection timeout
engine = create_engine(
    f'mysql+pymysql://{DB_USER}:{DB_PASS}@/{DB_NAME}?unix_socket=/cloudsql/{INSTANCE_CONNECTION_NAME}',
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={'connect_timeout': 10}  # Timeout after 10 seconds
)
