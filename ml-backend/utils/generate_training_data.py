import pandas as pd
import random

# Load the watch logs
logs = pd.read_csv('data/watch_logs.csv')

# Create set of (user_id, video_id) that were watched
watched_set = set(zip(logs['user_id'], logs['video_id']))

# Get unique users and videos
all_users = logs['user_id'].unique()
all_videos = logs['video_id'].unique()

data = []

# Positive samples (watched = 1)
for user_id, video_id in watched_set:
    data.append([user_id, video_id, 1])

# Negative samples (not watched = 0)
# For each positive sample, generate one random negative sample
for user_id, video_id in watched_set:
    while True:
        neg_video = random.choice(all_videos)
        if (user_id, neg_video) not in watched_set:
            data.append([user_id, neg_video, 0])
            break

# Create DataFrame and save
df = pd.DataFrame(data, columns=['user_id', 'video_id', 'watched'])
df.to_csv('data/training_data.csv', index=False)
print("✅ training_data.csv generated with labels!")
