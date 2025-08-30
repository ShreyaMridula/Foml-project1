import React, { useState } from 'react';
import './UploadPage.css';

function UploadPage() {
  const [file, setFile] = useState(null);
  const [thumbnail, setThumbnail] = useState(null);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [channel, setChannel] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleThumbnailChange = (e) => {
    setThumbnail(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const userId = localStorage.getItem("user_id"); // 👈 Get the logged-in user ID
    if (!userId) {
      alert("You must be logged in to upload a video.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', title);
    formData.append('description', description);
    formData.append('channel', channel);
    formData.append('thumbnail', thumbnail);
    formData.append('user_id', userId); // 👈 Include user ID in the form data

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        alert('Video uploaded successfully!');
      } else {
        alert('Error uploading video: ' + result.message);
      }
    } catch (error) {
      alert('Error uploading video: ' + error.message);
    }
  };

  return (
    <div className="upload-page">
      <h1 className="upload-heading">Upload Your Video</h1>
      <form className="upload-form" onSubmit={handleSubmit}>
        <input
          type="file"
          className="file-input"
          onChange={handleFileChange}
          required
        />
        <input
          type="text"
          placeholder="Title"
          className="input-field"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          required
        />
        <textarea
          placeholder="Description"
          className="textarea"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          required
        />
        <input
          type="text"
          placeholder="Channel"
          className="input-field"
          value={channel}
          onChange={(e) => setChannel(e.target.value)}
          required
        />
        <input
          type="file"
          className="file-input"
          onChange={handleThumbnailChange}
          required
        />
        <button type="submit" className="upload-btn">Upload</button>
      </form>
    </div>
  );
}

export default UploadPage;
