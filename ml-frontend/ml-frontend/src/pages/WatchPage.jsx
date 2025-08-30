import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import "./WatchPage.css";

function WatchPage() {
  const { id } = useParams();
  const [video, setVideo] = useState(null);
  const [comments, setComments] = useState([]);
  const [commentText, setCommentText] = useState("");
  const [error, setError] = useState(null);

  useEffect(() => {
    const currentUserId = 1; // TODO: Replace with actual logged-in user ID

    fetch(`http://localhost:8000/video/${id}`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch video metadata");
        return res.json();
      })
      .then(data => {
        // Update views
        fetch(`http://localhost:8000/video/${id}/view`, { method: "POST" })
          .then(res => res.json())
          .then(viewData => {
            setVideo({ ...data, views: viewData.views });

            // Log interaction for recommendation training
            fetch("http://localhost:8000/log_interaction", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                user_id: currentUserId,
                video_id: parseInt(id),
                watched: 1
              })
            });
          })
          .catch(() => {
            setVideo(data);
          });
      })
      .catch(() => setError("Video not found or failed to load."));

    // Fetch comments
    fetch(`http://localhost:8000/video/${id}/comments`)
      .then(res => res.json())
      .then(setComments)
      .catch(console.error);
  }, [id]);

  const handleLike = () => {
    fetch(`http://localhost:8000/video/${id}/like`, { method: "POST" })
      .then(res => res.json())
      .then(data => setVideo(v => ({ ...v, likes: data.likes })));
  };

  const handleDislike = () => {
    fetch(`http://localhost:8000/video/${id}/dislike`, { method: "POST" })
      .then(res => res.json())
      .then(data => setVideo(v => ({ ...v, dislikes: data.dislikes })));
  };

  const handleCommentSubmit = () => {
    if (!commentText.trim()) return;

    fetch(`http://localhost:8000/video/${id}/comment`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        user_id: 1, // TODO: Replace with real user ID from auth
        content: commentText,
      }),
    }).then(() => {
      setCommentText("");
      fetch(`http://localhost:8000/video/${id}/comments`)
        .then(res => res.json())
        .then(setComments);
    });
  };

  if (error) return <div className="watch-page-error">{error}</div>;
  if (!video) return <div className="watch-page-loading">Loading...</div>;

  return (
    <div className="watch-page">
      <div className="content">
        <div className="video-section">
          <video controls poster={video.thumbnail}>
            <source src={video.url} type="video/mp4" />
            Your browser does not support the video tag.
          </video>

          <h2 className="video-title">{video.title}</h2>

          <div className="meta-bar">
            <span>{video.views} views</span>
            <div className="actions">
              <button onClick={handleLike}>Like ({video.likes})</button>
              <button onClick={handleDislike}>Dislike ({video.dislikes})</button>
            </div>
          </div>

          <p className="description">{video.description}</p>

          <div className="comments">
            <h3>Comments</h3>
            <textarea
              placeholder="Add a comment..."
              value={commentText}
              onChange={e => setCommentText(e.target.value)}
            />
            <button
              onClick={handleCommentSubmit}
              disabled={!commentText.trim()}
            >
              Submit
            </button>

            {comments.map((c, idx) => (
              <div key={idx} className="comment-item">
                <p><strong>User {c.user_id}:</strong> {c.content}</p>
                <p className="comment-time">{c.created_at}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="related-videos">
          <h4>Related Videos</h4>
          <p>Coming soon...</p>
        </div>
      </div>
    </div>
  );
}

export default WatchPage;
