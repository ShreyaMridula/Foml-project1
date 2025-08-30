from recommendation.model import VideoRecommender
import pandas as pd
import numpy as np

# Load CSV data
print("📂 Loading data...")
users_df = pd.read_csv("data/users.csv")
videos_df = pd.read_csv("data/videos.csv")
logs_df = pd.read_csv("data/watch_logs.csv")

print(f"👤 Users: {len(users_df)}")
print(f"🎬 Videos: {len(videos_df)}")
print(f"📊 Watch logs: {len(logs_df)}")

# Step 1: Create the recommender object
recommender = VideoRecommender(users_df, videos_df, logs_df)

# Step 2: Prepare data
print("🔧 Preparing data...")
X, y = recommender.prepare_data()
print(f"✅ Data shape: X={X.shape}, y={len(y)}")

# Step 3: Build the model
recommender.build_model()

# Step 4: Train the model
print("🔁 Training model...")
y_encoded = np.zeros((len(y), recommender.num_videos))
for i, video_id in enumerate(X[:, 1]):
    y_encoded[i, video_id] = 1

recommender.train(X[:, 0], y_encoded, epochs=5)

# Step 5: Test recommendation for a sample user_id
sample_user_id = users_df.iloc[0]["user_id"]
print(f"\n🎯 Recommendations for user {sample_user_id}:\n")
recommendations = recommender.recommend(sample_user_id)
for video in recommendations:
    print(f"- {video['title']} ({video['category']})")
