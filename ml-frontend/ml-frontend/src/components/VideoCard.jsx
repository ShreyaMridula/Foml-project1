import React from "react";
import { Link } from "react-router-dom";
import "./VideoCard.css";


export default function VideoCard({ video }) {
  return (
    <Link to={`/watch/${video.id}`} className="video-card">
      <img src={video.thumbnail} alt={video.title} className="thumbnail" />
      <div className="video-info">
        <h3 className="title">{video.title}</h3>
        <p className="channel">{video.channel}</p>
        {(video.views || video.time) && (
  <p className="meta">
    {video.views ? video.views : ""}
    {video.views && video.time ? " • " : ""}
    {video.time ? video.time : ""}
  </p>
)}

        {/* You can add views/time here if needed */}
      </div>
    </Link>
  );
}