from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Literal
from .database import get_session
from .models import Exhibit, Feedback
from .services import apply_feedback
from sqlmodel import Session, select

router = APIRouter()

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
    session.add(Feedback(exhibit_id=feedback.exhibit_id, rating=feedback.rating, view_seconeds=feedback.view_seconds))
    apply_feedback(exhibit, new_rating = feedback.rating, new_view_seconds = feedback.view_seconds)
    session.add(exhibit)
    session.commit()
    session.refresh(exhibit)
    return { "ok" : True, "new_score" : exhibit.score, "new_avg_view_time_min" : exhibit.avg_view_time_min  }
                       
