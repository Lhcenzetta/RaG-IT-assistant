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
def query(query_user :shcema.Questionner_user , db : Session = Depends(get_db)):
    exist_user = db.query(User).filter(query_user.userid == User.id).first()
    if not exist_user:
        raise HTTPException(status_code=400 , detail="Please check again this user n'exist pas")
    # answer = Handle_query(query_user.question)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    model_path = os.getenv("model_path")
    loaded_model = joblib.load(model_path)
    query_embedding = embedding.aembed_query(query)
    cluster_number = loaded_model.predict([query_embedding])
    new_query_user = Query(
        userid = exist_user.id,
        question = query_user.question,
        # answer = answer,
        # cluster = cluster_number[0]
    )
    return cluster_number




@router.delete("/delete_user/{user_id}")
def delete_user(user_id :int , db:Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404 , detail="User not found")
    db.delete(user)
    db.commit()
    return {"detail" : "User deleted successfully"}