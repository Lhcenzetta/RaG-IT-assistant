from pydantic import BaseModel

class CreateUser(BaseModel):
    email : str
    hashedpassword : str
    is_active : bool
    created_at : str

class Checkuser:
    email : str
    hashedpassword : str