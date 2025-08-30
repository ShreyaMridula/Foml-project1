import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import Navbar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import "./SearchResultPage.css";

function SearchResultPage() {
  const [videos, setVideos] = useState([]);
  const [title, setTitle] = useState("");
  const location = useLocation();
  const navigate = useNavigate();

  const queryParams = new URLSearchParams(location.search);
  const query = queryParams.get("query");
  const userId = queryParams.get("user_id");

  useEffect(() => {
    if (query) {
      setTitle(`Search Results for "${query}"`);
      axios
        .get("http://127.0.0.1:8000/search", {
          params: { query },
        })
        .then((res) => setVideos(res.data))
        .catch((err) => console.error("Search error:", err));
    } else if (userId) {
      setTitle(`Recommended Videos for User ID "${userId}"`);
      axios
        .get("http://127.0.0.1:8000/recommend", {
          params: { user_id: userId, top_k: 5 },
        })
        .then((res) => setVideos(res.data.recommended_videos))
        .catch((err) => console.error("Recommendation error:", err));
    }
  }, [query, userId]);

  const handleVideoClick = (videoId) => {
    navigate(`/watch/${videoId}`);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <Navbar />
      <div style={{ display: "flex", flex: 1 }}>
        <Sidebar />
        <div style={{ flex: 1, padding: "1rem" }}>
          <h2 style={{ color: "#fff", marginBottom: "1rem" }}>{title}</h2>

          <div className="video-grid">
            {videos.length > 0 ? (
              videos.map((video, index) => (
                <div
                  key={index}
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
                  </div>
                </div>
              ))
            ) : (
              <p style={{ color: "#ccc" }}>No results found.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default SearchResultPage;
