import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import './LoginPage.css'; // Reusing the same CSS

function SignupPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSignup = async (e) => {
    e.preventDefault();
    try {
      await axios.post(
        'http://localhost:8000/signup',
        new URLSearchParams({
          email: email,         // ✅ must match FastAPI param name
          password: password    // ✅ must match FastAPI param name
        }),
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
        }
      );
      alert("Signup successful!");
      navigate("/login");
    } catch (error) {
      alert(error.response?.data?.detail || "Signup failed");
    }
  };
  

  return (
    <div className="login-wrapper"> {/* Same class as LoginPage */}
      <form className="login-form" onSubmit={handleSignup}>
        <h2 className="title">Sign Up</h2>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
        />
        <button type="submit">Sign Up</button>
        <p className="footer-text">
          Already have an account? <Link to="/">Login</Link>
        </p>
      </form>
    </div>
  );
}

export default SignupPage;
