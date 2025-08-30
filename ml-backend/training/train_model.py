import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.orm import Session
from . import models

def load_data(db: Session):
    interactions = db.query(models.UserVideoHistory).all()
    user_ids = [i.user_id for i in interactions]
    video_ids = [i.video_id for i in interactions]
    watched = np.ones(len(user_ids))  # Implicit feedback: all watched = 1
    return np.array(user_ids), np.array(video_ids), watched

def train_model(db: Session, epochs=5):
    user_ids, video_ids, watched = load_data(db)

    user_encoder = LabelEncoder()
    video_encoder = LabelEncoder()

    user_ids_encoded = user_encoder.fit_transform(user_ids)
    video_ids_encoded = video_encoder.fit_transform(video_ids)

    num_users = len(user_encoder.classes_)
    num_videos = len(video_encoder.classes_)

    # Functional model
    user_input = tf.keras.Input(shape=(1,))
    video_input = tf.keras.Input(shape=(1,))

    user_emb = tf.keras.layers.Embedding(num_users, 16)(user_input)
    video_emb = tf.keras.layers.Embedding(num_videos, 16)(video_input)

    user_vec = tf.keras.layers.Flatten()(user_emb)
    video_vec = tf.keras.layers.Flatten()(video_emb)

    x = tf.keras.layers.Concatenate()([user_vec, video_vec])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[user_input, video_input], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit([user_ids_encoded, video_ids_encoded], watched, epochs=epochs, batch_size=32, verbose=1)

    os.makedirs("models", exist_ok=True)
    model.save("models/recommendation_model.keras")

    with open("models/user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)
    with open("models/video_encoder.pkl", "wb") as f:
        pickle.dump(video_encoder, f)

    print("✅ Model and encoders saved!")
