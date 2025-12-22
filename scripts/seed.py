import json
from pathlib import Path
from sqlmodel import Session

from src.api.database import engine, init_db
from src.api.models import Room, Exhibit  # <-- import both

INSTANCE_PATH = Path("instances") / "floor_small.json"


def seed(instance_path: Path = INSTANCE_PATH):
    """
    Create tables (if missing) and insert rooms + exhibits from JSON once.

    Order matters:
      1) rooms first (Exhibit has FK room_id -> Room.id)
      2) exhibits second
    """
    init_db()

    if not instance_path.exists():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")

    data = json.loads(instance_path.read_text(encoding="utf-8"))

    rooms_data = data.get("rooms", [])
    exhibits_data = data.get("exhibits", [])

    rooms = []
    for r in rooms_data:
        rooms.append(
            Room(
                id=r["id"],
                name=r["name"],
                floor=int(r["floor"]),
                x=float(r["x"]),
                y=float(r["y"]),
                width=float(r["width"]),
                height=float(r["height"]),
            )
        )

    exhibits = []
    for e in exhibits_data:
        # Optional sanity check: exhibit.floor should match its room.floor
        # (we enforce it in code below if room exists in DB)
        exhibits.append(
            Exhibit(
                id=e["id"],
                name=e["name"],
                type=e["type"],  # "painting" | "sculpture" (Enum will accept str)
                pos_x=float(e["pos_x"]),
                pos_y=float(e["pos_y"]),
                floor=int(e["floor"]),
                room_id=e["room_id"],

                # initial aggregates (updated by /feedback)
                prior=3.8,
                prior_weight=20,
                rating_sum=0.0,
                rating_count=0,
                score=3.8,
                avg_view_time_min=5.0,
                view_time_count=0,
            )
        )

    inserted_rooms = 0
    inserted_exhibits = 0

    with Session(engine) as s:
        # --- insert rooms first ---
        for room in rooms:
            if not s.get(Room, room.id):
                s.add(room)
                inserted_rooms += 1
        s.commit()

        # --- insert exhibits second ---
        # Validate FK + optional floor consistency
        for ex in exhibits:
            if s.get(Exhibit, ex.id):
                continue

            room = s.get(Room, ex.room_id)
            if room is None:
                raise ValueError(
                    f"Exhibit {ex.id} references missing room_id={ex.room_id}"
                )

            # Optional: enforce exhibit.floor = room.floor (consistency)
            if ex.floor != room.floor:
                # either raise, or silently correct
                # raise ValueError(f"Exhibit {ex.id} floor={ex.floor} != Room {room.id} floor={room.floor}")
                ex.floor = room.floor

            s.add(ex)
            inserted_exhibits += 1

        s.commit()

    print(
        f"Seed complete. Inserted {inserted_rooms} rooms and "
        f"{inserted_exhibits} exhibits. File: {instance_path}"
    )


if __name__ == "__main__":
    seed()
