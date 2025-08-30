import React, { useState, useEffect } from "react";
import Navbar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import VideoCard from "../components/VideoCard";
import axios from "axios";
import "./HomePage.css";

export default function HomePage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [videos, setVideos] = useState([]);

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const userId = localStorage.getItem("user_id");
        const endpoint = `http://localhost:8000/api/videos/user/${userId}`;
        const response = await axios.get(endpoint);
        const data = response.data;

        if (Array.isArray(data)) {
          setVideos(data);
        } else {
          console.warn("Invalid response format:", data);
        }
      } catch (error) {
        console.error("Failed to fetch videos:", error);
      }
    };

    fetchVideos();
  }, []);

  return (
    <div className={`homepage ${isSidebarOpen ? "sidebar-open" : ""}`}>
      <Navbar toggleSidebar={toggleSidebar} />
      <div className="content-container">
        <Sidebar isOpen={isSidebarOpen} />
        <main className="video-grid">
          {videos.map((video) => (
            <VideoCard
              key={video.id}
              video={{
                id: video.id,
                title: video.title,
                thumbnail: video.thumbnail_path,
                channel: video.channel || "Channel",
                ...(video.views && { views: video.views }),
                ...(video.time && { time: video.time }),
              }}
            />
          ))}
        </main>
      </div>
    </div>
  );
}
