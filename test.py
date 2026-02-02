from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt ,JWTError
import os 
from dotenv import load_dotenv

bearer_scheme = HTTPBearer()

load_dotenv()
algorithme = "HS256"
secret_key = os.getenv("SECRET_KEY")


def create_token(paylod):
    return jwt.encode(paylod, secret_key ,algorithme)

def decode_token(token):
    try:
        return jwt.decode(token , secret_key , algorithme)
    except JWTError:
        return None

token = create_token({'username' : "lahcen aitzetta"})
print(token)
print(decode_token(token))

def verfiy_token(cred: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = cred.credentials
    decode = decode_token(token)
    if decode is None:
        return "invalide token"
    return decode


