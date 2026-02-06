from fastapi import FastAPI
from backend.app.db.models import Base
from db.database import engine
from api  import auth 


Base.metadata.create_all(engine)

app = FastAPI()


app.include_router(auth.router)