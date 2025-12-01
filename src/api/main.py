from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Tuple

from sqlmodel import select
from .routes import router as api_router
from .database import get_session
from .models import Exhibit

app = FastAPI(title= "Museum Tour API", version="1.0.0")
app.include_router(api_router)
session = get_session()
 
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
    return session.exec(select(Exhibit)).all()

class PlanRequest(BaseModel):
    time_budget_min : float
    entrance : Tuple[float,float] = (0.0,0.0)
    prefs : dict | None = None

@app.post("/route/plan")
def plan_route(req : PlanRequest):
    import math
    rows = session.exec(select(Exhibit)).all()
    exhibits = [
        {
            "id": ex.id,
            "name": ex.name,
            "type": ex.type,
            "pos": (ex.pos_x, ex.pos_y),
            "score": ex.score,
            "view_min": ex.avg_view_time_min if ex.avg_view_time_min is not None else 5.0,
        }
        for ex in rows
    ]
    speed = 60.0 # meters per minute
    def walk_min(a,b): return math.hypot(a[0]-b[0],a[1]-b[1]) / speed
    pos = req.entrance
    total_walk = 0.0
    total_view = 0.0
    total_time = 0.0
    unvisited = exhibits.copy()
    prefs = req.prefs or {}
    path = []
    remaining = float(req.time_budget_min)

    while True:
        best = None
        best_ratio = -1.0
        best_cost = None
        best_walk = None

        for e in unvisited:
            wmin = walk_min(pos, e["pos"])
            vmin = e["view_min"]
            cost = wmin + vmin
            if cost > remaining:
                continue  # cannot fit

            weight = float(prefs.get(e["type"], 1.0))
            ratio = (e["score"] * weight) / max(cost, 1e-9)
            if ratio > best_ratio:
                best_ratio = ratio
                best = e
                best_cost = cost
                best_walk = wmin

        if best is None:
            break  # nothing else fits

        # take it
        step = {
            "id": best["id"],
            "name": best["name"],
            "type": best["type"],
            "pos": best["pos"],
            "walk_min": round(best_walk, 2),
            "view_min": round(best["view_min"], 2),
            "score": best["score"],
            "score_weighted": round(best["score"] * float(prefs.get(best["type"], 1.0)), 3),
            "eta_min": round(total_time + best_cost, 2),
        }
        path.append(step)

        # update totals/position/remaining
        pos = best["pos"]
        remaining -= best_cost
        total_time += best_cost
        total_walk += best_walk
        total_view += best["view_min"]
        unvisited.remove(best)

        if remaining <= 1e-6 or not unvisited:
            break

    return {
        "path": path,
        "total_time_min": round(total_time, 2),
        "total_walk_min": round(total_walk, 2),
        "total_view_min": round(total_view, 2),
        "visited": [p["id"] for p in path],
        "polyline": [req.entrance] + [p["pos"] for p in path],
        "leftover_minutes": round(remaining, 2),
    }