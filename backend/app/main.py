from fastapi import FastAPI
from db.models import Base
from db.database import engine
from api  import auth 
Base.metadata.create_all(engine)
app = FastAPI()