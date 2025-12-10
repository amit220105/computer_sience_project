from fastapi import FastAPI
from .routes import router as api_router
from .database import init_db

app = FastAPI(title="Museum Tour API", version="1.0.0", debug=True)
app.include_router(api_router)

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/health")
def health():
    return {"ok": True} 