from sqlalchemy.orm import Session
from fastapi import APIRouter , Depends, HTTPException
from db.models import User
from db import shcema
from datetime import datetime
from db.session import get_db
from passlib.context import CryptContext
router = APIRouter()


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_hash_mode_pass(password):
    return pwd_context.hash(password)

def verfiy_hash_passsword(new_password , hashed_password):
    return pwd_context.verify(new_password, hashed_password)

def create_token(paylod):
    


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
def login(user : shcema.Checkuser , db : Session = Depends(get_db)):
    exit_user = db.query(User).filter(user.email == User.email).first()
    if not exit_user:
        raise HTTPException(status_code=400 , detail="This user doesn't exist ! please login")
    if not verfiy_hash_passsword(user.hashedpassword, User.hashedpassword):
        raise HTTPException(status_code=400 , detail="Password invalid")
    
    paylod = {"email":user.email }
    token  = create_token(paylod)
        

