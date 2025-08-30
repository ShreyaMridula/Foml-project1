import React, { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import "./TrendingPage.css";

function TrendingPage() {
  const [videos, setVideos] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    axios
      .get("http://localhost:8000/trending")
      .then(res => setVideos(res.data))
      .catch(err => console.error("Trending fetch error:", err));
  }, []);

  const handleVideoClick = (id) => {
    navigate(`/watch/${id}`);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <Navbar />
      <div style={{ display: "flex", flex: 1 }}>
        <Sidebar />
        <div style={{ flex: 1, padding: "1rem" }}>
          <h2 style={{ color: "#fff", marginBottom: "1rem" }}>Trending Videos</h2>

          <div className="video-grid">
            {videos.map((video, idx) => (
              <div
                key={idx}
                className="video-card"
                onClick={() => handleVideoClick(video.id)}
                style={{ cursor: "pointer" }}
              >
                <img
                  src={video.thumbnail}
                  alt={video.title}
                  className="thumbnail"
                />
                <div className="video-info">
                  <h4 style={{ color: "#fff" }}>{video.title}</h4>
                  <p style={{ color: "#ccc" }}>{video.description}</p>
                  <p style={{ color: "#aaa" }}>{video.views} views</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default TrendingPage;
