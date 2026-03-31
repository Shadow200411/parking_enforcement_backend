from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import jwt

from app.core.database import get_db
from app.models.domain import User
from app.core.security import verify_password, create_access_token, oauth2_scheme, SECRET_KEY, ALGORITHM

router = APIRouter(prefix="/auth",tags=["Authentication"])

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    """Validates credentials and hands out a digital badge."""
    #Find the user in the database.
    stmt = select(User).where(User.fullname == form_data.username)
    result = await db.execute(stmt)
    user = result.scalars().first()
    
    #Check if the user exist and password matches
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
        
    #Create the token
    access_tocken = create_access_token(data={"sub": user.fullname})
    return {"access_token": access_tocken, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    """Checks the tocken and returns the logged-in user.""" 
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
    
    #Make sure the user still exists in the database
    stmt = select(User).where(User.fullname == username)
    result = await db.execute(stmt)
    user = result.scalars().first()
    if user is None:
        raise credentials_exception
    return user
