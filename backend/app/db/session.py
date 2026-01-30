
from backend.app.db.database import engine
from sqlalchemy.orm import sessionmaker

session = sessionmaker(bind=engine)

def get_db():
    db = session()
    try:
        yield db
    finally:
        db.close()
