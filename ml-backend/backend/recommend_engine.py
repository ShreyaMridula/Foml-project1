import os
import pickle
import numpy as np
import tensorflow as tf
from typing import List
from sqlalchemy.orm import Session
from .database import SessionLocal
from . import models

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "recommendation_model.keras")
USER_ENCODER_PATH = os.path.join(BASE_DIR, "models", "user_encoder.pkl")
VIDEO_ENCODER_PATH = os.path.join(BASE_DIR, "models", "video_encoder.pkl")

# Load model and encoders
print("🚀 Loading recommendation model and encoders...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(USER_ENCODER_PATH, "rb") as f:
    user_encoder = pickle.load(f)

with open(VIDEO_ENCODER_PATH, "rb") as f:
    video_encoder = pickle.load(f)

print("✅ Model and encoders loaded.")

def get_video_metadata(video_ids: List[int], db: Session) -> List[dict]:
    results = []
    for video_id in video_ids:
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if video:
            results.append({
                "video_id": video.id,
                "title": video.title,
                "thumbnail_url": f"http://localhost:8000/{video.thumbnail_path}",
                "channel": video.channel
            })
    return results

def recommend_videos(user_id: int, db: Session, top_k: int = 5):
    # Fallback to a default known user if user_id is not in encoder
    try:
        user_internal_id = user_encoder.transform([user_id])[0]
    except ValueError:
        print(f"⚠️ User ID {user_id} not seen during training. Using fallback user.")
        fallback_user_id = user_encoder.classes_[0]  # Use first known user
        user_internal_id = user_encoder.transform([fallback_user_id])[0]

    # Query all videos from the database
    videos = db.query(models.Video).all()
    if not videos:
        raise ValueError("No videos found in the database.")

    # Encode video IDs
    video_ids = [video.id for video in videos]
    video_internal_ids = video_encoder.transform(video_ids)

    # Predict for all videos
    preds = model.predict([np.full_like(video_internal_ids, user_internal_id), video_internal_ids], verbose=0).flatten()

    # Top K predictions
    top_indices = preds.argsort()[::-1][:top_k]
    recommended_video_ids = video_encoder.inverse_transform(top_indices)

    # Get video metadata from the database
    recommendations = []
    for video_id in recommended_video_ids:
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if video:
            recommendations.append({
                "video_id": video.id,
                "title": video.title,
                "description": video.description,
                "thumbnail_url": f"http://localhost:8000/thumbnails/{video.thumbnail_path.split('/')[-1]}",
                "video_url": f"http://localhost:8000/videos/{video.video_path.split('/')[-1]}"
            })

    print("📊 Recommended videos for user_id", user_id, ":", recommendations)
    print("User encoder classes:", user_encoder.classes_)

    return recommendations
