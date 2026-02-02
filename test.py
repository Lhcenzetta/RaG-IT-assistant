from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_hash_mode_pass(password):
    return pwd_context.hash(password)

def verfiy_hash_passsword(new_password , hashed_password):
    return pwd_context.verify(new_password, hashed_password)

hash_code = create_hash_mode_pass("1234")
print(hash_code)
print(verfiy_hash_passsword("1234",hash_code))