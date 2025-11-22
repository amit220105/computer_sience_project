from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Tuple
from .routes import router as api_router


app = FastAPI(title= "Museum Tour API", version="1.0.0")
app.include_router(api_router)
class Exhibit(BaseModel):
    id : str
    name : str
    type : Literal["painting", "sculpture"]
    pos : Tuple[float, float]
    score : float = 3.8
    avg_view_time_min : float = 5.0

EXHIBITS: List[Exhibit] = [
    Exhibit(id="E1", name="Sunset",      type="painting",  pos=(10,4),  score=4.1, avg_view_time_min=6.0),
    Exhibit(id="E2", name="Marble Hero", type="sculpture", pos=(5,12),  score=4.3, avg_view_time_min=5.0),
    Exhibit(id="E3", name="Blue Room",   type="painting",  pos=(14,9),  score=3.9, avg_view_time_min=4.5),
]
#@app.get("/ready")
#def ready():
    #from .database import get_session
    #try:
        #with get_session() as session:
            #session.execute("SELECT 1")
        #return {"status": "ready"} 
    #except Exception:
       #raise HTTPException(status_code=503, detail="Service Unavailable")

@app.get("/health")
def health():
    return {"ok": True}       
    
@app.get("/exhibits", response_model=List[Exhibit])
def get_exhibits():
    return EXHIBITS

class PlanRequest(BaseModel):
    time_budget_min : float
    entrance : Tuple[float,float] = (0.0,0.0)
    prefs : dict | None = None

@app.post("/route/plan")
def plan_route(req : PlanRequest):
    import math
    speed = 60.0 # meters per minute
    def dist(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
    pos = req.entrance
    total_time = 0.0
    path = []
    for ex in EXHIBITS:
        walk_min = dist(pos, ex.pos) / speed
        view_min = ex.avg_view_time_min
        if total_time + walk_min + view_min > req.time_budget_min:
            continue
        total_time += walk_min + view_min
        path.append({
            "id": ex.id,
            "name": ex.name,
            "type": ex.type,
            "pos": ex.pos,
            "walk_min": round(walk_min, 2),
            "view_min": view_min,
            "score": ex.score
        })
        pos = ex.pos

    return {
        "path": path,
        "total_time_min": round(total_time, 2),
        "total_score": round(sum(p["score"] for p in path), 2)
    }