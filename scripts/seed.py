import json
from pathlib import Path
from sqlmodel import Session
from src.api.database import engine, init_db
from src.api.models import Room, Exhibit, ExhibitType


def run(seed_path: str):
    init_db()
    data = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    with Session(engine) as s:
        # rooms
        for r in data["rooms"]:
            s.add(Room(
                id=r["id"],
                name=r["name"],
                theme=r.get("theme"),
                pos_x=r["pos"][0], pos_y=r["pos"][1], pos_z=r["pos"][2],
                dwell_min=5.0  # default value; will be updated later
            ))
        # exhibits
        for e in data["exhibits"]:
            s.add(Exhibit(
                id=e["id"],
                room_id=e["room_id"],
                name=e["name"],
                type=ExhibitType(e["type"]),
                avg_view_time_min=float(e.get("avg_view_time_min", 5.0))
            ))
        s.commit()

        # derive room dwell from its exhibits (avg view time)
        rooms = s.query(Room).all()
        for r in rooms:
            exs = [ex for ex in r.exhibits]
            if exs:
                r.dwell_min = max(3.0, sum(ex.avg_view_time_min for ex in exs)/len(exs))
        s.commit()

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
