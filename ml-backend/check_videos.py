# check_videos.py
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend import models

db: Session = SessionLocal()

videos = db.query(models.Video).all()

if not videos:
    print("❌ No videos found in the database.")
else:
    for video in videos:
        print(f"{video.id}: {video.title}")
