# backend/recommend_engine.py
import pickle
import numpy as np
import pandas as pd
import os
import tensorflow as tf

# Setup paths
base_dir = os.path.dirname(__file__)
models_dir = os.path.join(base_dir, "models")
data_dir = os.path.join(base_dir, "data")

# Load the trained model
model = tf.keras.models.load_model(os.path.join(models_dir, "recommendation_model.h5"))

# Load encoders
with open(os.path.join(models_dir, "user_encoder.pkl"), "rb") as f:
    user_encoder = pickle.load(f)

with open(os.path.join(models_dir, "video_encoder.pkl"), "rb") as f:
    video_encoder = pickle.load(f)

# Load video metadata
video_metadata_path = os.path.join(data_dir, "videos.csv")

if not os.path.exists(video_metadata_path):
    raise FileNotFoundError(f"Could not find videos.csv at {video_metadata_path}")

video_metadata = pd.read_csv(video_metadata_path)

# Validate columns
required_cols = {"video_id", "title", "thumbnail_url", "description"}
if not required_cols.issubset(video_metadata.columns):
    raise ValueError(f"Missing required columns in videos.csv. Found columns: {video_metadata.columns.tolist()}")

def recommend_videos(user_id: str, top_k: int = 5):
    """
    Recommend top_k videos for a given user_id based on predicted watch probability.
    """

    if user_id not in user_encoder.classes_:
        raise ValueError(f"User ID '{user_id}' not found in the user encoder.")

    # Encode user_id
    user_id_encoded = user_encoder.transform([user_id])[0]

    # Prepare all possible video inputs
    all_video_ids = video_metadata["video_id"].tolist()
    video_ids_encoded = video_encoder.transform(all_video_ids)

    # Prepare model inputs
    user_inputs = np.full(shape=len(video_ids_encoded), fill_value=user_id_encoded)
    video_inputs = np.array(video_ids_encoded)

    # Predict watch probabilities
    predictions = model.predict([user_inputs, video_inputs], batch_size=256, verbose=0).flatten()

    # Get top_k video indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_video_ids = np.array(all_video_ids)[top_indices]

    # Build the recommendations list
    recommendations = []
    for vid in top_video_ids:
        video_row = video_metadata[video_metadata["video_id"] == vid]
        if not video_row.empty:
            video_info = video_row.iloc[0]
            recommendations.append({
                "video_id": int(video_info["video_id"]),
                "title": video_info["title"],
                "thumbnail_url": video_info["thumbnail_url"],
                "description": video_info["description"]
            })
        else:
            recommendations.append({
                "video_id": int(vid),
                "title": "Unknown Title",
                "thumbnail_url": "",
                "description": ""
            })

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }
