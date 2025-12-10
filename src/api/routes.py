import json
import math
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Literal

from pyparsing import Dict


from .database import get_session
from .models import Exhibit, Feedback, RouteLog
from .services import apply_feedback
from sqlmodel import Session, select

router = APIRouter()

class PlanRequest(BaseModel):
    time_budget_min: float
    entrance: Tuple[float, float] = (0.0, 0.0)
    prefs: dict | None = None

class ExhibitOut(BaseModel):
    id : str
    name : str
    type : Literal["painting", "sculpture"]
    pos : Tuple[float, float]
    score : float
    avg_view_time_min : float | None

@router.get("/exhibits", response_model= List[ExhibitOut])
def list_exhibits(session : Session =Depends(get_session)):
    exhibits = session.exec(select(Exhibit)).all()
    return [
        ExhibitOut(
            id=ex.id,
            name=ex.name,
            type=ex.type,
            pos=(ex.pos_x, ex.pos_y),
            score=ex.score,
            avg_view_time_min=ex.avg_view_time_min
        ) for ex in exhibits
    ]   

class FeedbackIn(BaseModel):
    exhibit_id : str
    rating : int
    view_seconds : int

@router.post("/feedback")
def submit_feedback(feedback: FeedbackIn, session : Session =Depends(get_session)):
    exhibit = session.get(Exhibit, feedback.exhibit_id)
    if not exhibit:
        raise HTTPException(status_code=404, detail="Exhibit not found")
    session.add(Feedback(exhibit_id=feedback.exhibit_id, rating=feedback.rating, view_seconds=feedback.view_seconds))
    apply_feedback(exhibit, new_rating = feedback.rating, new_view_seconds = feedback.view_seconds)
    session.add(exhibit)
    session.commit()
    session.refresh(exhibit)
    return { "ok" : True, "new_score" : exhibit.score, "new_avg_view_time_min" : exhibit.avg_view_time_min  }

class PlanRequest(BaseModel):
    time_budget_min: float = Field(gt=0, description="Total minutes available")
    entrance: Tuple[float, float] = (0.0, 0.0)
    prefs: Optional[dict[str, float]] = None
    speed_m_per_min: float = Field(default=60.0, gt=0)
    return_to_entrance: bool = False
    must_visit: List[str] = []
    avoid: List[str] = []
    max_stops: Optional[int] = Field(default=None, ge=1)

@router.post("/route/plan")
def plan_route(req: PlanRequest, session: Session = Depends(get_session)):
    rows = session.exec(select(Exhibit)).all()

    # Filter avoids; clamp minimal dwell (2.0 min)
    exhibits = [
        {
            "id": ex.id,
            "name": ex.name,
            "type": ex.type.value if hasattr(ex.type, "value") else str(ex.type),
            "pos": (ex.pos_x, ex.pos_y),
            "score": float(ex.score),
            "view_min": max(2.0, float(ex.avg_view_time_min) if ex.avg_view_time_min is not None else 5.0),
        }
        for ex in rows
        if ex.id not in set(req.avoid)
    ]

    # Build pools: must-visit first, then others
    id_to_obj = {e["id"]: e for e in exhibits}
    must_pool = [id_to_obj[i] for i in req.must_visit if i in id_to_obj]
    other_pool = [e for e in exhibits if e["id"] not in set(req.must_visit)]

    def walk_min(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1]) / req.speed_m_per_min

    pos = tuple(req.entrance)
    remaining = float(req.time_budget_min)
    prefs = req.prefs or {}

    def greedy_pick(pool, pos, remaining):
        best = None
        best_ratio = -1.0
        best_cost = None
        best_walk = None
        for e in pool:
            w = walk_min(pos, e["pos"])
            c = w + e["view_min"]
            if c > remaining:
                continue
            weight = float(prefs.get(e["type"], 1.0))
            ratio = (e["score"] * weight) / max(c, 1e-9)
            if ratio > best_ratio:
                best_ratio, best, best_cost, best_walk = ratio, e, c, w
        return best, best_cost, best_walk

    path = []
    total_walk = total_view = total_time = 0.0
    pools = [must_pool, other_pool]

    while True:
        # Switch to next pool when current one is empty
        while pools and not pools[0]:
            pools.pop(0)
        if not pools:
            break

        best, cost, wmin = greedy_pick(pools[0], pos, remaining)
        if best is None:
            # nothing fits from this pool — try next pool
            pools.pop(0)
            if not pools:
                break
            continue

        step = {
            "id": best["id"],
            "name": best["name"],
            "type": best["type"],
            "pos": best["pos"],
            "walk_min": round(wmin, 2),
            "view_min": round(best["view_min"], 2),
            "score": best["score"],
            "score_weighted": round(best["score"] * float(prefs.get(best["type"], 1.0)), 3),
            "eta_min": round(total_time + cost, 2),
        }
        path.append(step)

        pos = best["pos"]
        remaining -= cost
        total_time += cost
        total_walk += wmin
        total_view += best["view_min"]

        # remove chosen from both pools
        for pool in pools:
            pool[:] = [e for e in pool if e["id"] != best["id"]]

        if req.max_stops and len(path) >= req.max_stops:
            break

    # Optional return to entrance
    back_walk_min = 0.0
    if req.return_to_entrance and path:
        back_walk_min = walk_min(pos, tuple(req.entrance))
        if back_walk_min <= remaining:
            total_time += back_walk_min
            total_walk += back_walk_min
            remaining -= back_walk_min
        # If it doesn't fit, we simply skip the return (plausible UX)

    result = {
        "path": path,
        "total_time_min": round(total_time, 2),
        "total_walk_min": round(total_walk, 2),
        "total_view_min": round(total_view, 2),
        "visited": [p["id"] for p in path],
        "polyline": [tuple(req.entrance)] + [p["pos"] for p in path],
        "leftover_minutes": round(remaining, 2),
    }
    if req.return_to_entrance:
        result["back_walk_min"] = round(back_walk_min, 2)

    # Log to RouteLog
    session.add(RouteLog(
        time_budget_min=req.time_budget_min,
        speed_m_per_min=req.speed_m_per_min,
        return_to_entrance=req.return_to_entrance,
        max_stops=req.max_stops,
        entrance_x=req.entrance[0],
        entrance_y=req.entrance[1],
        prefs_json=json.dumps(req.prefs or {}),
        must_visit_json=json.dumps(req.must_visit),
        avoid_json=json.dumps(req.avoid),
        visited_json=json.dumps(result["visited"]),
        total_time_min=result["total_time_min"],
        total_walk_min=result["total_walk_min"],
        total_view_min=result["total_view_min"],
    ))
    session.commit()

    return result
                       
