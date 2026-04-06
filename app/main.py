from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path



from app.api import endpoints, auth
   
from app.services.scheduler import scheduler, start_scheduler
from app.services.ai_inference import init_ocr_engine, shutdown_ocr_engine


@asynccontextmanager
async def lifespan(app : FastAPI):
    print("Starting background scheduler...")
    print("Initializing OCR engine...")
    init_ocr_engine()
    start_scheduler()
    yield
    print("Stopping background scheduler...")
    scheduler.shutdown()
    shutdown_ocr_engine()

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

static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "message": "Backend is running."}

app.include_router(auth.router, prefix="/api/v1")
app.include_router(endpoints.router, prefix="/api/v1")
