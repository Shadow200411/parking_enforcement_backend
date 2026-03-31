from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer


#Setting up the hasher and the OAuth2 scheme for Swagger UI
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

#The Secret Key used to sign the badges (need to change later)
SECRET_KEY = "super-secret-police-key"
ALGORITHM = "HS256"

def verify_password(plain_password, hashed_password):
    """
    Cheks if the typed password mathces the database hash.
    """
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    """
    Creates a temoprary digital badge that expires in 8 hours.
    """
    #TO DO: how long will the badge last for?
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=8)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)