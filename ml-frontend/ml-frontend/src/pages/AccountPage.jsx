import React, { useState } from 'react';
import './AccountPage.css';

const AccountPage = () => {
  const [username, setUsername] = useState('User123');
  const [email, setEmail] = useState('user@example.com');
  const [profilePic, setProfilePic] = useState(null);

  const handleProfilePicChange = (e) => {
    setProfilePic(URL.createObjectURL(e.target.files[0]));
  };

  const handleSaveChanges = () => {
    alert('Profile updated successfully!');
  };

  return (
    <div className="account-page">
      <div className="account-container">
        <h2>Your Account</h2>
        <div className="profile-info">
          <div className="profile-pic">
            <img
              src={profilePic || 'https://via.placeholder.com/100'}
              alt="Profile"
            />
            <input type="file" onChange={handleProfilePicChange} />
          </div>
          <div className="user-info">
            <div className="info-item">
              <label>Username:</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div className="info-item">
              <label>Email:</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <button className="save-btn" onClick={handleSaveChanges}>
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AccountPage;
