import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import "./Navbar.css";

const Navbar = ({ toggleSidebar }) => {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  const handleSearch = () => {
    if (query.trim()) {
      navigate(`/search?query=${encodeURIComponent(query)}`);
    }
  };

  return (
    <div className="navbar">
      <div className="navbar-left">
        <button className="hamburger" onClick={toggleSidebar}>
          ☰
        </button>
        <h2 className="app-name">Lyra</h2>
      </div>

      <div className="search-container">
        <input
          type="text"
          placeholder="Search..."
          className="search-bar"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="search-btn" onClick={handleSearch}>
          <Search size={18} />
        </button>
      </div>

      <div className="navbar-right">
        <button className="nav-button" onClick={() => navigate("/upload")}>Create</button>
        <button className="nav-button" onClick={() => navigate("/account")}>Account</button>
      </div>
    </div>
  );
};

export default Navbar;
