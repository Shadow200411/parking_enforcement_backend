from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager



from app.api import endpoints, auth
   
from app.services.scheduler import scheduler, start_scheduler

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app : FastAPI):
    print("Starting background scheduler...")
    start_scheduler()
    yield
    print("Stopping background scheduler...")
    scheduler.shutdown()

app = FastAPI(
    title="Parking Enforcement API",
    description="Backend for the AI-powered police car parking enforcement system.",
    version="1.0.0",
    lifespan=lifespan
)


origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "message": "Backend is running."}

app.include_router(auth.router, prefix="/api/v1")
app.include_router(endpoints.router, prefix="/api/v1")
