import json
from pathlib import Path
from sqlmodel import Session
from src.api.database import engine, init_db
from src.api.models import Exhibit

INSTANCE_PATH = Path("instances") / "floor_small.json"

def seed(instance_path: Path = INSTANCE_PATH):
    """Create tables (if missing) and insert exhibits from JSON once."""
    init_db()
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")

    data = json.loads(instance_path.read_text(encoding="utf-8"))
    exhibits = []
    for e in data.get("exhibits", []):
        exhibits.append(
            Exhibit(
                id=e["id"],
                name=e["name"],
                type=e["type"],          # "painting" | "sculpture"
                pos_x=float(e["pos"][0]),
                pos_y=float(e["pos"][1]),
                # initial aggregates (will be updated by /feedback)
                prior=3.8,
                prior_weight=20,
                rating_sum=0.0,
                rating_count=0,
                score=3.8,
                avg_view_time_min=5.0,
                view_time_count=0,
            )
        )

    inserted = 0
    with Session(engine) as s:
        for ex in exhibits:
            if not s.get(Exhibit, ex.id):   # insert only if not exists
                s.add(ex)
                inserted += 1
        s.commit()

    print(f"Seed complete. Inserted {inserted} exhibits. File: {instance_path}")

if __name__ == "__main__":
    seed()