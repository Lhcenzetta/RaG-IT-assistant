
from fastapi import APIRouter

router = APIRouter()

@router.get("/homeç")
def home():
    return "success"


