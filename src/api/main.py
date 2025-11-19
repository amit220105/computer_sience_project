from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Tuple

app = FastAPI(title= "Museum Tour API", version="1.0.0")
class Exhibit(BaseModel):
    id : int
    name : str
    type : Literal["painting", "sculpture"]
    pos : Tuple[float, float]
    score : float = 3.8
    avg_time_in_min : float = 5.0

EXHIBITS: List[Exhibit] = [
    Exhibit(id="E1", name="Sunset",      type="painting",  pos=(10,4),  score=4.1, avg_view_time_min=6.0),
    Exhibit(id="E2", name="Marble Hero", type="sculpture", pos=(5,12),  score=4.3, avg_view_time_min=5.0),
    Exhibit(id="E3", name="Blue Room",   type="painting",  pos=(14,9),  score=3.9, avg_view_time_min=4.5),
]
@app.get("/ready")
def ready():
    from .database import get_session
    try:
        with get_session() as session:
            session.execute("SELECT 1")
        return {"status": "ready"} 
    except Exception:
       raise HTTPException(status_code=503, detail="Service Unavailable")
    
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
    for exhibit in EXHIBITS:
        walk_time = dist(pos,exhibit.pos) / speed
        view_time = exhibit.avg_time_in_min
        if total_time + walk_time + view_time > req.time_budget_min:
            break
        total_time += walk_time +view_time
        pathappend(exhibit)
        pos = exhibit.pos
        return {"path": path, "total_time_min": total_time, "score": sum(e.score for e in path)}