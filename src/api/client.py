import requests
from typing import Dict, List, Tuple, Optional

BASE_URL = "http://127.0.0.1:8000"  
S = requests.Session()
S.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
TIMEOUT = 10  # seconds


def get_rooms() -> List[dict]:
    r = S.get(f"{BASE_URL}/rooms", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_room_exhibits(room_id: str) -> List[dict]:
    r = S.get(f"{BASE_URL}/rooms/{room_id}/exhibits", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def plan_route_rooms(
    time_budget_min: float,
    entrance: Tuple[float, float, int] = (0.0, 0.0, 0),
    prefs: Optional[Dict[str, float]] = None,
    speed_m_per_min: float = 60.0,
    return_to_entrance: bool = False,
    must_visit: Optional[List[str]] = None,
    avoid: Optional[List[str]] = None,
    max_stops: Optional[int] = None,
) -> dict:
    payload = {
        "time_budget_min": time_budget_min,
        "entrance": entrance,
        "prefs": prefs or {},
        "speed_m_per_min": speed_m_per_min,
        "return_to_entrance": return_to_entrance,
        "must_visit": must_visit or [],
        "avoid": avoid or [],
        "max_stops": max_stops,
    }
    r = S.post(f"{BASE_URL}/route/plan-rooms", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def submit_feedback(exhibit_id: str, rating: int, view_seconds: int) -> dict:
    payload = {"exhibit_id": exhibit_id, "rating": rating, "view_seconds": view_seconds}
    r = S.post(f"{BASE_URL}/feedback", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    # Example usage:
    rooms = get_rooms()
    print("Rooms:", rooms)

    if rooms:
        rid = rooms[0]["id"]
        exhibits = get_room_exhibits(rid)
        print(f"Exhibits in {rid}:", exhibits[:2])

    route = plan_route_rooms(
        time_budget_min=25,
        entrance=(0, 0, 0),
        prefs={"painting": 1.1, "sculpture": 0.9},
        return_to_entrance=True,
        must_visit=[rooms[0]["id"]] if rooms else [],
    )
    print("Planned route:", route)

    # Send one feedback example (if we saw an exhibit id)
    if rooms:
        exs = get_room_exhibits(rooms[0]["id"])
        if exs:
            print("Feedback result:", submit_feedback(exs[0]["id"], rating=5, view_seconds=420))