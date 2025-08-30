from fastapi import FastAPI, HTTPException, Form, Depends,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy import func
from . import models
from .database import engine, get_db, Base
from .recommend_engine import recommend_videos
import bcrypt
from urllib.parse import quote
import os
from datetime import datetime
import traceback
import sqlite3
import joblib
import numpy as np
import tensorflow as tf
import pickle
from backend import watch
from sqlalchemy import or_
import shutil
import uuid





models.Base.metadata.create_all(bind=engine)


app = FastAPI()
app.include_router(watch.router)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "users.db")

app.mount("/videos", StaticFiles(directory="videos"), name="videos")
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/signup")
def signup(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user = models.User(email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User created successfully", "user_id": user.id}

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.hashed_password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"message": "Login successful", "user_id": user.id}

@app.get("/recommend")
def recommend(user_id: int, top_k: int = 5, db: Session = Depends(get_db)):
    try:
        recommendations = recommend_videos(user_id, db=db, top_k=top_k)  # Pass db here
        return {"user_id": user_id, "recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Unexpected error: " + str(e))

@app.get("/search")
def search_videos(query: str, db: Session = Depends(get_db)):
    videos = db.query(models.Video).filter(
        func.lower(models.Video.title).ilike(f"%{query.lower()}%") |
        func.lower(models.Video.description).ilike(f"%{query.lower()}%")
    ).all()
    results = [
        {
            "id": video.id,
            "title": video.title,
            "description": video.description,
            "channel": video.channel,
            "url": f"http://localhost:8000/videos/{quote(video.video_path.split('/')[-1])}",
            "thumbnail": f"http://localhost:8000/thumbnails/{video.thumbnail_path.split('/')[-1]}"
        }
        for video in videos
    ]
    return results

@app.get("/video/{video_id}")
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "id": video.id,
        "title": video.title,
        "description": video.description,
        "channel": video.channel,
        "likes": video.likes,
        "dislikes": video.dislikes,
        "views": video.views,
        "url": f"http://localhost:8000/videos/{quote(video.video_path.split('/')[-1])}",
        "thumbnail": f"http://localhost:8000/thumbnails/{video.thumbnail_path.split('/')[-1]}"
    }

@app.get("/stream/{video_id}")
def stream_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = video.video_path
    full_path = os.path.join(os.getcwd(), video_path)

    if os.path.exists(full_path):
        return FileResponse(full_path, media_type="video/mp4")
    else:
        raise HTTPException(status_code=404, detail="Video file not found")

@app.post("/video/{video_id}/view")
def increment_view(video_id: int, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video.views += 1
    db.commit()
    return {"message": "View count updated", "views": video.views}

@app.post("/video/{video_id}/like")
def like_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video.likes += 1
    db.commit()
    return {"message": "Video liked", "likes": video.likes}

@app.post("/video/{video_id}/dislike")
def dislike_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    video.dislikes += 1
    db.commit()
    return {"message": "Video disliked", "dislikes": video.dislikes}

@app.post("/video/{video_id}/comment")
def post_comment(video_id: int, user_id: int = Form(...), content: str = Form(...), db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    comment = models.Comment(video_id=video_id, user_id=user_id, content=content, created_at=datetime.now())
    db.add(comment)
    db.commit()
    return {"message": "Comment added"}

@app.get("/video/{video_id}/comments")
def get_comments(video_id: int, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    comments = db.query(models.Comment).filter(models.Comment.video_id == video_id).order_by(models.Comment.created_at.desc()).all()
    return [
        {
            "user_id": c.user_id,
            "content": c.content,
            "created_at": c.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for c in comments
    ]

@app.get("/trending")
def get_trending_videos(db: Session = Depends(get_db)):
    videos = db.query(models.Video).order_by(models.Video.views.desc()).limit(10).all()

    trending = []

    for video in videos:
        # Skip if path is missing
        if not video.video_path or not video.thumbnail_path:
            continue

        try:
            video_url = f"http://localhost:8000/videos/{quote(os.path.basename(video.video_path.strip()))}"
            thumbnail_url = f"http://localhost:8000/thumbnails/{quote(os.path.basename(video.thumbnail_path.strip()))}"

            trending.append({
                "id": video.id,
                "title": video.title,
                "description": video.description,
                "channel": video.channel,
                "url": video_url,
                "thumbnail": thumbnail_url,
                "views": video.views
            })
        except Exception as e:
            # Optional: log or skip bad video records
            print(f"Error processing video {video.id}: {e}")
            continue

    return trending

@app.get("/")
def read_root(db: Session = Depends(get_db)):
    videos = db.query(models.Video).order_by(models.Video.views.desc()).limit(10).all()
    return [
        {
            "id": video.id,
            "title": video.title,
            "description": video.description,
            "channel": video.channel,
            "url": f"http://localhost:8000/videos/{quote(video.video_path.split('/')[-1])}",
            "thumbnail": f"http://localhost:8000/thumbnails/{quote(os.path.basename(video.thumbnail_path.strip()))}",
            "views": video.views
        }
        for video in videos
    ]

@app.get("/api/videos/user/{user_id}")
def get_personalized_videos(user_id: int, db: Session = Depends(get_db)):
    if user_id == 2:
        # User 2: yoga/java/python videos
        keywords = ['%yoga%', '%java%', '%python%']
    elif user_id == 3:
        # User 3: tech/science/coding videos
        keywords = ['%machine learning%', '%science%', '%music%']
    else:
        # Other users: travel/food/funny videos
        keywords = ['%travel%', '%pasta%', '%funny%']

    videos = db.query(models.Video).filter(
        or_(
            models.Video.title.ilike(keywords[0]),
            models.Video.description.ilike(keywords[0]),
            models.Video.title.ilike(keywords[1]),
            models.Video.description.ilike(keywords[1]),
            models.Video.title.ilike(keywords[2]),
            models.Video.description.ilike(keywords[2])
        )
    ).all()

    return [
        {
            "id": v.id,
            "title": v.title,
            "description": v.description,
            "channel": v.channel,
            "url": f"http://localhost:8000/videos/{quote(v.video_path.split('/')[-1])}",
            "thumbnail_path": f"http://localhost:8000/thumbnails/{quote(v.thumbnail_path.split('/')[-1])}"
        }
        for v in videos
    ]
@app.get("/api/videos/library")
def get_all_videos(db: Session = Depends(get_db)):
    videos = db.query(models.Video).all()
    return [
        {
            "id": video.id,
            "title": video.title,
            "description": video.description,
            "channel": video.channel,
            "thumbnail": video.thumbnail_path,
            "url": video.video_path,
        }
        for video in videos
    ]
