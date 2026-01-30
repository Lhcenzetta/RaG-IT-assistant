from sqlalchemy.engine import create_engine
import os 
from dotenv import load_dotenv
from sqlalchemy.ext.declarative import declarative_base
load_dotenv()
DataBase_URL = f"postgresql+psycopg2://{os.getenv('user')}:{os.getenv('password')}@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('database')}"
engine = create_engine(DataBase_URL)

Base = declarative_base()
