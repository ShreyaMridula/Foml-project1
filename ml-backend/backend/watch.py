from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .database import get_db
from .models import UserVideoHistory
from datetime import datetime

router = APIRouter()

class WatchEvent(BaseModel):
    user_id: int
    video_id: int

@router.post("/watch")
def log_watch_event(event: WatchEvent, db: Session = Depends(get_db)):
    record = UserVideoHistory(
        user_id   = event.user_id,
        video_id  = event.video_id,
        watched_at = datetime.utcnow()
    )
    db.add(record)
    db.commit()
    return {"message": "Watch event logged"}
