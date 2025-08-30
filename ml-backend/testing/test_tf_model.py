import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# Load model and encoders
model = tf.keras.models.load_model("models/recommendation_model.h5")
with open("models/user_encoder.pkl", "rb") as f:
    user_enc = pickle.load(f)

with open("models/video_encoder.pkl", "rb") as f:
    video_enc = pickle.load(f)

# Load video metadata
videos_df = pd.read_csv("data/videos.csv")  # Make sure 'video_id' and 'title' columns exist

# Pick a sample user ID (as string because encoders are usually fitted on strings)
sample_user_id = "1"  # Change this to any valid user_id from users.csv
user_encoded = user_enc.transform([sample_user_id])[0]  # Encode user

# Prepare inputs
video_ids = video_enc.classes_  # All video IDs known to encoder
video_encoded = video_enc.transform(video_ids)  # Encode video IDs

# Predict watch probabilities
user_input = np.full(len(video_encoded), user_encoded)
predictions = model.predict([user_input, video_encoded], verbose=0).flatten()

# Get top 5 recommended video indices
top_indices = predictions.argsort()[-5:][::-1]
recommended_video_ids = video_ids[top_indices]

# Match with video titles
recommended_videos = videos_df[videos_df['video_id'].astype(str).isin(recommended_video_ids)]

# Print top recommendations
print(f"\n🔝 Top 5 recommendations for User ID {sample_user_id}:")
for vid in recommended_video_ids:
    title_row = recommended_videos[recommended_videos['video_id'].astype(str) == vid]
    if not title_row.empty:
        title = title_row['title'].values[0]
        print(f"▶️ {title} (ID: {vid})")
    else:
        print(f"▶️ (Unknown title) (ID: {vid})")
