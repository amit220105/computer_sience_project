
from __future__ import annotations
import json
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from sqlmodel import Session
from src.api.database import engine, init_db
from src.api.models import Room, Exhibit, ExhibitType
import sys
from collections import defaultdict
from sqlmodel import Session, SQLModel, select, delete

# Adjust import to your project layout
from src.api.database import engine, init_db
from src.api.models import Room, Exhibit

def run(json_path: str) -> None:
    init_db()  # ensure tables exist
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    rooms_in = data.get("rooms", [])
    exhibits_in = data.get("exhibits", [])

    with Session(engine) as s:
        # Wipe in FK-safe order (children -> parents)
        s.exec(delete(Exhibit))
        s.exec(delete(Room))
        s.commit()

        # Insert rooms
        rooms = []
        for r in rooms_in:
            pos = r.get("pos", [0, 0, 0])
            rooms.append(
                Room(
                    id=r["id"],
                    name=r["name"],
                    theme=r.get("theme"),
                    pos_x=float(pos[0]),
                    pos_y=float(pos[1]),
                    pos_z=float(pos[2]) if len(pos) > 2 else 0.0,
                    dwell_min=5.0,
                )
            )
        s.add_all(rooms)
        s.commit()

        # Insert exhibits
        exs = []
        for e in exhibits_in:
            exs.append(
                Exhibit(
                    id=e["id"],
                    room_id=e["room_id"],
                    name=e["name"],
                    type=e["type"],
                    avg_view_time_min=float(e.get("avg_view_time_min", 5.0)),
                    prior=3.8,
                    prior_weight=20,
                    rating_sum=0.0,
                    rating_count=0,
                    score=3.8,
                    view_time_count=0,
                )
            )
        s.add_all(exs)
        s.commit()

        # Derive room dwell time as average of exhibits in that room (min 3.0)
        room_to_times = defaultdict(list)
        for e in exs:
            room_to_times[e.room_id].append(e.avg_view_time_min or 5.0)

        for rid, times in room_to_times.items():
            room = s.get(Room, rid)
            if room:
                avg = sum(times) / max(len(times), 1)
                room.dwell_min = max(3.0, avg)
                s.add(room)
        s.commit()

    print(f"Seeded {len(rooms)} rooms and {len(exs)} exhibits from {p.name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.seed <path-to-json>")
        sys.exit(1)
    run(sys.argv[1])