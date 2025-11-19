from typing import Optional, Literal
from sqlmodel import SQLModel, Field
from datetime import datetime

ExhibitType = Literal["painting", "sculpture"]
class Exhibit(SQLModel, table=True):
    id : str = Field(primary_key=True)
    name : str
    type : ExhibitType
    pos_x : float
    pos_y : float

    prior : float = 3.8
    prior_weight : int = 20
    rating_sum : float = 0.0
    rating_count : int = 0
    score : float = 3.8
    avg_view_time_min : float = 5.0
    view_time_count : int = 0

class Feedback(SQLModel, table=True):
    id : Optional[int] = Field(default=None, primary_key=True)
    exhibit_id : str = Field(foreign_key="exhibit.id")
    rating : int
    view_seconeds : int
    timestamp : datetime = Field(default_factory=datetime.utcnow)