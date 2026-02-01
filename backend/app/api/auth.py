
from fastapi import APIRouter

router = APIRouter()

@router.post("/Signup")
def home():
    return "success"


