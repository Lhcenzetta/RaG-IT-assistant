from fastapi import FastAPI
from db.models import Base
from db.database import engine

Base.metadata.create_all(engine)
app = FastAPI()