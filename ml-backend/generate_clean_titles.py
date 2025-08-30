import pandas as pd
import random

# Number of videos (adjust to match your original dataset)
num_videos = 50

# Generate some fun sample titles
sample_titles = [
    "Top 10 Travel Destinations", "Learn Python in 10 Minutes", "Cute Cats Compilation",
    "Space Discoveries You Didn’t Know", "Beginner Yoga Routine", "History of Ancient Rome",
    "Motivational Morning Routine", "Bizarre Food Reviews", "Guitar Lessons for Beginners",
    "How to Make Sushi at Home", "Intro to Machine Learning", "Secrets of the Universe",
    "Horror Short Film", "Life in Tokyo Vlog", "Productivity Tips for Students",
    "Minecraft Survival Guide", "How to Budget Your Money", "10 Minute Workouts", 
    "Exploring Haunted Places", "Nature Walk with Me"
]

# Pad out the list if needed
while len(sample_titles) < num_videos:
    sample_titles += [f"Random Video {i}" for i in range(len(sample_titles), num_videos)]

# Shuffle and slice
random.shuffle(sample_titles)
titles = sample_titles[:num_videos]

# Create DataFrame
df = pd.DataFrame({
    "video_id": list(range(num_videos)),
    "title": titles
})

# Save it to the data folder
df.to_csv("data/videos.csv", index=False)

print("✅ videos.csv updated with clean titles!")
