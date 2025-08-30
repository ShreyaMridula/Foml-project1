// src/pages/SubscriptionPage.jsx
import React, { useEffect, useState } from "react";
import axios from "axios";

const SubscriptionPage = () => {
  const [videos, setVideos] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:8000/subscriptions")
      .then((res) => setVideos(res.data))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="p-4 text-white">
      <h2 className="text-2xl font-bold mb-4">Subscribed Channels</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
        {videos.map(video => (
          <div key={video.id} className="bg-gray-900 p-4 rounded-lg shadow-lg">
            <img src={video.thumbnail} alt={video.title} className="rounded mb-2" />
            <h3 className="font-semibold">{video.title}</h3>
            <p className="text-sm text-gray-400">{video.channel}</p>
            <p className="text-sm text-gray-500">{video.views} views</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SubscriptionPage;
