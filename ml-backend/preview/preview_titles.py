import pandas as pd

# Load your labeled training data
training_data_df = pd.read_csv('data/training_data.csv')

# Load videos.csv which contains video titles
videos_df = pd.read_csv('data/videos.csv')

# Merge on video_id to bring in the title column
merged = training_data_df.merge(videos_df[['video_id', 'title']], on='video_id', how='left')

# View the merged DataFrame
print(merged.head(10))  # Show the first 10 rows with titles
