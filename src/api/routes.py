from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlmodel import Session, select

from .database import get_session
from .models import Exhibit, Feedback, RouteLog, Room
from .services import apply_feedback

router = APIRouter()



class ExhibitOut(BaseModel):
    id: str
    name: str
    type: Literal["painting", "sculpture"]
    pos: Tuple[float, float]          # global XY for drawing if needed
    score: float
    avg_view_time_min: Optional[float]


class FeedbackIn(BaseModel):
    exhibit_id: str
    rating: int
    view_seconds: int


class RoomOut(BaseModel):
    id: str
    name: str
    pos: Tuple[float, float, int]     # (x, y, z)
    exhibit_count: int = 0


class PlanRoomsRequest(BaseModel):
    time_budget_min: float = Field(gt=0)
    entrance: Tuple[float, float, int] = (0.0, 0.0, 0)
    speed_m_per_min: float = Field(default=60.0, gt=0)
    floor_change_min: float = Field(default=2.0, ge=0)   # penalty per floor transition
    prefs: Optional[Dict[str, float]] = None             # {"painting":1.1, "Impressionism":1.2}
    return_to_entrance: bool = False
    must_visit: List[str] = []       # room IDs
    avoid: List[str] = []            # room IDs
    max_stops: Optional[int] = Field(default=None, ge=1)


class RoomStepOut(BaseModel):
    id: str
    name: str
    pos: Tuple[float, float, int]
    walk_min: float
    view_min: float
    score: float
    score_weighted: float
    eta_min: float


class PlanRoomsResult(BaseModel):
    path: List[RoomStepOut]
    total_time_min: float
    total_walk_min: float
    total_view_min: float
    visited: List[str]
    polyline3d: List[Tuple[float, float, int]]
    leftover_minutes: float
    back_walk_min: Optional[float] = None


# =========================================================
# Helpers
# =========================================================

def _xy_walk_min(a_xy: Tuple[float, float], b_xy: Tuple[float, float], speed: float) -> float:
    return math.hypot(a_xy[0] - b_xy[0], a_xy[1] - b_xy[1]) / speed


def _room_cost_minutes(
    from_pos: Tuple[float, float, int],
    to_pos: Tuple[float, float, int],
    speed: float,
    floor_change_min: float,
    dwell_min: float,
) -> Tuple[float, float]:
    """Return (walk_min, total_cost_min = walk + dwell + floor_penalty)."""
    walk_xy = _xy_walk_min((from_pos[0], from_pos[1]), (to_pos[0], to_pos[1]), speed)
    floor_pen = abs(from_pos[2] - to_pos[2]) * floor_change_min
    walk_min = walk_xy + floor_pen
    return walk_min, walk_min + dwell_min


def _room_score_and_weight(session: Session, room: Room, prefs: Dict[str, float]) -> Tuple[float, float]:
    """
    Base room score = average of exhibit scores in the room (falls back to 3.0 if empty).
    Weight:
      * if prefs has the room's theme, use it.
      * else average the weights of exhibit types present (painting/sculpture) from prefs.
    """
    exs = session.exec(select(Exhibit).where(Exhibit.room_id == room.id)).all()
    base = 3.0 if not exs else float(sum(float(e.score) for e in exs) / len(exs))

    # theme-based weight
    if room.theme and room.theme in (prefs or {}):
        return base, float(prefs[room.theme])

    # fallback: average weights by exhibit type present
    weight = 1.0
    if exs and prefs:
        vals = []
        for e in exs:
            t = e.type.value if hasattr(e.type, "value") else str(e.type)
            vals.append(float(prefs.get(t, 1.0)))
        if vals:
            weight = sum(vals) / len(vals)

    return base, weight


def _exhibit_global_pos(session: Session, ex: Exhibit) -> Tuple[float, float]:
    """
    Compute the exhibit's global XY position.
    - If model has ex.pos_x/pos_y, use them.
    - Else add local_x/local_y to the parent room centroid.
    """
    if hasattr(ex, "pos_x") and hasattr(ex, "pos_y"):
        return float(getattr(ex, "pos_x")), float(getattr(ex, "pos_y"))

    # local offsets + room centroid
    room = session.get(Room, ex.room_id) if getattr(ex, "room_id", None) else None
    rx = float(room.pos_x) if room else 0.0
    ry = float(room.pos_y) if room else 0.0
    lx = float(getattr(ex, "local_x", 0.0))
    ly = float(getattr(ex, "local_y", 0.0))
    return rx + lx, ry + ly


# =========================================================
# Exhibits
# =========================================================

@router.get("/exhibits", response_model=List[ExhibitOut])
def list_exhibits(session: Session = Depends(get_session)):
    exhibits = session.exec(select(Exhibit)).all()
    out: List[ExhibitOut] = []
    for ex in exhibits:
        t = ex.type.value if hasattr(ex.type, "value") else str(ex.type)
        gx, gy = _exhibit_global_pos(session, ex)
        out.append(
            ExhibitOut(
                id=ex.id,
                name=ex.name,
                type=t,  # type: ignore
                pos=(gx, gy),
                score=float(ex.score),
                avg_view_time_min=float(ex.avg_view_time_min) if ex.avg_view_time_min is not None else None,
            )
        )
    return out


# =========================================================
# Feedback
# =========================================================

@router.post("/feedback")
def submit_feedback(feedback: FeedbackIn, session: Session = Depends(get_session)):
    exhibit = session.get(Exhibit, feedback.exhibit_id)
    if not exhibit:
        raise HTTPException(status_code=404, detail="Exhibit not found")

    # persist raw feedback
    session.add(
        Feedback(
            exhibit_id=feedback.exhibit_id,
            rating=feedback.rating,
            view_seconds=feedback.view_seconds,
        )
    )

    # update exhibit stats (Bayesian + EMA)
    apply_feedback(
        exhibit,
        new_rating=feedback.rating,
        new_view_seconds=feedback.view_seconds,
    )
    session.add(exhibit)
    session.commit()
    session.refresh(exhibit)

    # update the room dwell time as the average of its exhibits' averages
    if getattr(exhibit, "room_id", None):
        room = session.get(Room, exhibit.room_id)
        if room:
            exs = session.exec(select(Exhibit).where(Exhibit.room_id == room.id)).all()
            if exs:
                new_dwell = sum((float(e.avg_view_time_min or 5.0)) for e in exs) / len(exs)
                room.dwell_min = max(3.0, new_dwell)
                session.add(room)
                session.commit()

    return {
        "ok": True,
        "new_score": float(exhibit.score),
        "new_avg_view_time_min": float(exhibit.avg_view_time_min),
    }


# =========================================================
# Rooms
# =========================================================

@router.get("/rooms", response_model=List[RoomOut])
def list_rooms(session: Session = Depends(get_session)):
    rooms = session.exec(select(Room)).all()

    # exhibit counts (simple group by)
    rows = session.exec(
        select(Exhibit.room_id, func.count(Exhibit.id)).group_by(Exhibit.room_id)
    ).all()
    counts: Dict[str, int] = {rid: int(c) for rid, c in rows if rid is not None}

    return [
        RoomOut(
            id=r.id,
            name=r.name,
            pos=(float(r.pos_x), float(r.pos_y), int(getattr(r, "pos_z", 0))),
            exhibit_count=counts.get(r.id, 0),
        )
        for r in rooms
    ]


@router.get("/rooms/{room_id}/exhibits", response_model=List[ExhibitOut])
def list_room_exhibits(room_id: str, session: Session = Depends(get_session)):
    room = session.get(Room, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    exs = session.exec(select(Exhibit).where(Exhibit.room_id == room_id)).all()
    out: List[ExhibitOut] = []
    for ex in exs:
        t = ex.type.value if hasattr(ex.type, "value") else str(ex.type)
        gx, gy = _exhibit_global_pos(session, ex)
        out.append(
            ExhibitOut(
                id=ex.id,
                name=ex.name,
                type=t,  # type: ignore
                pos=(gx, gy),
                score=float(ex.score),
                avg_view_time_min=float(ex.avg_view_time_min) if ex.avg_view_time_min is not None else None,
            )
        )
    return out


# =========================================================
# Room-level planner
# =========================================================

@router.post("/route/plan-rooms", response_model=PlanRoomsResult)
def plan_route_rooms(req: PlanRoomsRequest, session: Session = Depends(get_session)):
    prefs = req.prefs or {}

    # candidate rooms (skip avoids)
    rooms: List[Room] = session.exec(select(Room)).all()
    rooms = [r for r in rooms if r.id not in set(req.avoid)]

    # materialize items
    items = []
    for r in rooms:
        base, weight = _room_score_and_weight(session, r, prefs)
        items.append({
            "id": r.id,
            "name": r.name,
            "pos": (float(r.pos_x), float(r.pos_y), int(getattr(r, "pos_z", 0))),
            "dwell_min": float(r.dwell_min),
            "score": float(base),
            "weight": float(weight),
        })

    # organize pools
    by_id = {i["id"]: i for i in items}
    must_pool = [by_id[i] for i in req.must_visit if i in by_id]
    other_pool = [i for i in items if i["id"] not in set(req.must_visit)]

    pos = tuple(req.entrance)
    remaining = float(req.time_budget_min)

    def greedy_pick(pool, pos, remaining):
        best = None
        best_ratio = -1.0
        best_cost = None
        best_walk = None
        for it in pool:
            walk_min, cost = _room_cost_minutes(
                pos, it["pos"], req.speed_m_per_min, req.floor_change_min, it["dwell_min"]
            )
            if cost > remaining:
                continue
            ratio = (it["score"] * it["weight"]) / max(cost, 1e-9)
            if ratio > best_ratio:
                best_ratio, best, best_cost, best_walk = ratio, it, cost, walk_min
        return best, best_cost, best_walk

    # iterate
    path: List[RoomStepOut] = []
    total_walk = total_view = total_time = 0.0
    pools = [must_pool, other_pool]

    while True:
        while pools and not pools[0]:
            pools.pop(0)
        if not pools:
            break

        best, cost, wmin = greedy_pick(pools[0], pos, remaining)
        if best is None:
            pools.pop(0)
            if not pools:
                break
            continue

        step = RoomStepOut(
            id=best["id"],
            name=best["name"],
            pos=best["pos"],
            walk_min=round(wmin, 2),
            view_min=round(best["dwell_min"], 2),
            score=round(best["score"], 2),
            score_weighted=round(best["score"] * best["weight"], 2),
            eta_min=round(total_time + cost, 2),
        )
        path.append(step)

        pos = best["pos"]
        remaining -= cost
        total_time += cost
        total_walk += wmin
        total_view += best["dwell_min"]

        for pool in pools:
            pool[:] = [i for i in pool if i["id"] != best["id"]]

        if req.max_stops and len(path) >= req.max_stops:
            break

    back_walk_min: Optional[float] = None
    if req.return_to_entrance and path:
        wmin, back_cost = _room_cost_minutes(
            pos, tuple(req.entrance), req.speed_m_per_min, req.floor_change_min, 0.0
        )
        if back_cost <= remaining:
            total_time += back_cost
            total_walk += wmin
            remaining -= back_cost
            back_walk_min = round(wmin, 2)

    result = PlanRoomsResult(
        path=path,
        total_time_min=round(total_time, 2),
        total_walk_min=round(total_walk, 2),
        total_view_min=round(total_view, 2),
        visited=[s.id for s in path],
        polyline3d=[tuple(req.entrance)] + [s.pos for s in path],
        leftover_minutes=round(remaining, 2),
        back_walk_min=back_walk_min,
    )

    # optional analytics log
    session.add(RouteLog(
        time_budget_min=req.time_budget_min,
        speed_m_per_min=req.speed_m_per_min,
        return_to_entrance=req.return_to_entrance,
        max_stops=req.max_stops,
        entrance_x=req.entrance[0],
        entrance_y=req.entrance[1],
        entrance_z=req.entrance[2],
        prefs_json=json.dumps(prefs),
        must_visit_json=json.dumps(req.must_visit),
        avoid_json=json.dumps(req.avoid),
        visited_json=json.dumps(result.visited),
        total_time_min=result.total_time_min,
        total_walk_min=result.total_walk_min,
        total_view_min=result.total_view_min,
    ))
    session.commit()

    return result