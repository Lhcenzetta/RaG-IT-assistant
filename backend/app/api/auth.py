from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer
from api.Pipline_retriver import Handle_query
from sqlalchemy.orm import Session
from fastapi import APIRouter , Depends, HTTPException, status
from db.models import User, Query
from db import shcema
from datetime import datetime
from db.session import get_db
from passlib.context import CryptContext
import joblib
from langchain_huggingface import HuggingFaceEmbeddings
from jose import jwt, JWTError
import os
import time
from dotenv import load_dotenv

load_dotenv()
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_path = os.getenv("model_path")
algorithme = "HS256"
SECRET_KEY  = os.getenv("SECRET_KEY")
barear_chema = HTTPBearer()
router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_hash_mode_pass(password):
    return pwd_context.hash(password)

def verfiy_hash_passsword(new_password , hashed_password):
    return pwd_context.verify(new_password, hashed_password)

def create_token(paylod):
    return jwt.encode(paylod, SECRET_KEY , algorithm=algorithme)

def decode_token(token):
    return jwt.decode(token , SECRET_KEY , algorithms=algorithme)

def verfiy_token(cre: HTTPAuthorizationCredentials = Depends(barear_chema)):
    token = cre.credentials
    decode = decode_token(token)
    if decode is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='this toke is invalide'
        )
    return decode
        

@router.post("/Signup")
def home(user : shcema.CreateUser , db : Session = Depends(get_db)):
    exist_user = db.query(User).filter(user.email == User.email).first()
    if exist_user:
        raise  HTTPException(status_code=400, detail="THIS USER ALERADY EXIST")
    else:
        new_user = User(
            email = user.email,
            hashedpassword = create_hash_mode_pass(user.hashedpassword),
            is_active = user.is_active,
            created_at = datetime.utcnow()
         )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"Successfully Registred !!!"}


@router.post("/login")
def login(user: shcema.Checkuser , db:Session = Depends(get_db)):
    exit_user = db.query(User).filter(user.email == User.email).first()
    if not exit_user:
        raise HTTPException(status_code=400 , detail="This user doesn't exist ! please login")
    if not verfiy_hash_passsword(user.hashedpassword,exit_user.hashedpassword):
        raise HTTPException(status_code=400 , detail="Password invalid")
    
    paylod = {"email":user.email }
    token  = create_token(paylod)
    return{"token" : token , "token_type" : "bearer"}


@router.post("/query")
async def query(query_user :shcema.Questionner_user ,cre = Depends(verfiy_token), db : Session = Depends(get_db)):
    exist_user = db.query(User).filter(query_user.userid == User.id).first()
    if not exist_user:
        raise HTTPException(status_code=400 , detail="Please check again this user n'exist pas")
    start_time = time.perf_counter()
    answer = Handle_query(query_user.question)
    loaded_model = joblib.load(model_path)
    query_embedding = await embedding.aembed_query(query_user.question)
    cluster_number = int(loaded_model.predict([query_embedding])[0])
    end_time = (time.perf_counter() - start_time) * 1000
    new_query_user = Query(
        userid = exist_user.id,
        question = query_user.question,
        answer = answer,
        cluster = cluster_number,
        latency_ms = end_time,
        created_at = datetime.utcnow()

    )
    db.add(new_query_user)
    db.commit()
    db.refresh(new_query_user)
    return new_query_user




@router.delete("/delete_user/{user_id}")
def delete_user(user_id :int , db:Session = Depends(get_db), cre = Depends(verfiy_token)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404 , detail="User not found")
    db.delete(user)
    db.commit()
    return {"detail" : "User deleted successfully"}