import pandas as pd
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# 1. Generate Dummy Users
users = []
for user_id in range(1, 51):  # 50 users
    users.append({
        "user_id": user_id,
        "name": fake.name(),
        "age": random.randint(18, 50)
    })

# Save users to CSV
pd.DataFrame(users).to_csv("data/users.csv", index=False)

# 2. Generate Dummy Videos
categories = ["Music", "Gaming", "Education", "News", "Sports", "Comedy"]
videos = []
for video_id in range(1, 101):  # 100 videos
    videos.append({
        "video_id": video_id,
        "title": fake.sentence(nb_words=4),
        "category": random.choice(categories),
        "tags": ",".join(fake.words(nb=3))
    })

# Save videos to CSV
pd.DataFrame(videos).to_csv("data/videos.csv", index=False)

# 3. Generate Watch Logs
watch_logs = []
for _ in range(1000):  # 1000 logs
    watch_logs.append({
        "user_id": random.randint(1, 50),
        "video_id": random.randint(1, 100),
        "watch_time": fake.date_time_this_year().isoformat()
    })

# Save watch logs to CSV
pd.DataFrame(watch_logs).to_csv("data/watch_logs.csv", index=False)

print("✅ Dummy data successfully generated!")
try:
    pd.DataFrame(users).to_csv("data/users.csv", index=False)
    print("✅ users.csv written")
except Exception as e:
    print("❌ Failed to write users.csv:", e)

try:
    pd.DataFrame(videos).to_csv("data/videos.csv", index=False)
    print("✅ videos.csv written")
except Exception as e:
    print("❌ Failed to write videos.csv:", e)

try:
    pd.DataFrame(watch_logs).to_csv("data/watch_logs.csv", index=False)
    print("✅ watch_logs.csv written")
except Exception as e:
    print("❌ Failed to write watch_logs.csv:", e)
