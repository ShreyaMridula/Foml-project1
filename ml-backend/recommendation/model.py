import numpy as np
import tensorflow as tf
import pickle
from sqlalchemy.orm import Session
from .. import models  # Adjust relative import to match your folder structure

print("🚀 Loading model and encoders for API...")
model = tf.keras.models.load_model("models/recommendation_model.keras")

with open("models/user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("models/video_encoder.pkl", "rb") as f:
    video_encoder = pickle.load(f)

def recommend_videos(user_id: int, db: Session, top_k: int = 5):
    if user_id not in user_encoder.classes_:
        print(f"⚠️ User {user_id} not in encoder. Using fallback.")
        user_internal_id = np.median(user_encoder.transform(user_encoder.classes_))
    else:
        user_internal_id = user_encoder.transform([user_id])[0]

    videos = db.query(models.Video).all()
    if not videos:
        return []

    video_ids = [video.id for video in videos]
    known_ids = [vid for vid in video_ids if vid in video_encoder.classes_]

    if not known_ids:
        return []

    video_internal_ids = video_encoder.transform(known_ids)

    preds = model.predict([
        np.full_like(video_internal_ids, user_internal_id),
        video_internal_ids
    ], verbose=0).flatten()

    top_indices = preds.argsort()[::-1][:top_k]
    recommended_video_ids = video_encoder.inverse_transform(top_indices)

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

    return recommendations
