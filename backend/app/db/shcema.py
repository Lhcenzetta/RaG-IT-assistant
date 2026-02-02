from pydantic import BaseModel

class CreateUser(BaseModel):
    email : str
    hashedpassword : str
    is_active : bool
    created_at : str

class Checkuser(BaseModel):
    email : str
    hashedpassword : str

    class config:
        orm_mode = True