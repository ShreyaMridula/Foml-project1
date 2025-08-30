import React, { useEffect, useState } from "react";
import axios from "axios";

const LibraryPage = () => {
  const [videos, setVideos] = useState([]);

  useEffect(() => {
    axios
      .get("http://localhost:8000/api/videos/library")
      .then((res) => {
        setVideos(res.data);
      })
      .catch((err) => {
        console.error("Error fetching videos:", err);
      });
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h2>Video Library</h2>
      {videos.length === 0 ? (
        <p>No videos available.</p>
      ) : (
        <div className="video-list" style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
          {videos.map((video) => (
            <div
              key={video.id}
              style={{
                width: "300px",
                border: "1px solid #ccc",
                borderRadius: "8px",
                padding: "10px",
              }}
            >
              <img
                src={`http://localhost:8000/${video.thumbnail}`}
                alt={video.title}
                width="100%"
                style={{ borderRadius: "6px" }}
              />
              <h3>{video.title}</h3>
              <p>{video.channel}</p>
              <p style={{ fontSize: "0.9em", color: "#555" }}>{video.description}</p>
              <a
                href={`http://localhost:8000/${video.url}`}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: "#1a73e8", textDecoration: "none" }}
              >
                ▶ Watch Video
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default LibraryPage;
