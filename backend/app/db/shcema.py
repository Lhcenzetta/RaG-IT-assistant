from pydantic import BaseModel

class CreateUser(BaseModel):
    email : str
    hashedpassword : str
    is_active : bool