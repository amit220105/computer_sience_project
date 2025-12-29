from typing import Optional, Literal
from sqlmodel import Relationship, SQLModel, Field
from datetime import datetime
from enum import Enum

class ExhibitType(str, Enum):
    painting = "painting"
    sculpture = "sculpture"

class Room(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    theme: Optional[str] = None

    # centroid coordinates (meters) for navigation
    pos_x: float
    pos_y: float
    pos_z: float = 0.0     # 0 for floor 1, 1 for floor 2 

    # derived dwell time for the whole room (minutes)
    dwell_min: float = 5.0

    # relationships
    exhibits: list["Exhibit"] = Relationship(back_populates="room")


class Exhibit(SQLModel, table=True):
    id: str = Field(primary_key=True)
    room_id: str = Field(foreign_key="room.id")
    room: Optional[Room] = Relationship(back_populates="exhibits")

    name: str
    type: ExhibitType

    # local offsets inside the room (meters)
    local_x: float = 0.0
    local_y: float = 0.0

    # Bayesian/EMA stats per exhibit
    prior: float = 3.8
    prior_weight: int = 20
    rating_sum: float = 0.0
    rating_count: int = 0
    score: float = 3.8
    avg_view_time_min: float = 5.0
    view_time_count: int = 0

class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    exhibit_id: str = Field(foreign_key="exhibit.id")
    rating: int
    view_seconds: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RouteLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    time_budget_min: float
    speed_m_per_min: float
    return_to_entrance: bool
    max_stops: Optional[int] = None

    entrance_x: float
    entrance_y: float
    entrance_z: float = 0.0

    prefs_json: str = "{}"       # additional preferences
    must_visit_json: str = "[]"  # room IDs
    avoid_json: str = "[]"       # room IDs
    visited_json: str = "[]"     # room IDs

    total_time_min: float
    total_walk_min: float
    total_view_min: float     


    