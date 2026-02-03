from pydantic import BaseModel

class CreateUser(BaseModel):
    email : str
    hashedpassword : str
    is_active : bool
    created_at : str

class Checkuser(BaseModel):
    email : str
    hashedpassword : str

class Questionner_user(BaseModel):
    userid : int
    question : str
    class config:
        orm_mode = True