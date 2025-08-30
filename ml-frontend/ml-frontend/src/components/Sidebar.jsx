import React from "react";
import { Home, Flame, Youtube, BookOpen, History } from "lucide-react";
import { Link } from "react-router-dom";
import "./Sidebar.css";

const Sidebar = ({ isOpen }) => {
  return (
    <div className={`sidebar ${isOpen ? "open" : ""}`}>
      <Link to="/home" className="sidebar-link">
        <Home size={20} />
        <span>Home</span>
      </Link>
      <Link to="/trending" className="sidebar-link">
        <Flame size={20} />
        <span>Trending</span>
      </Link>
      <Link to="/library" className="sidebar-link">
        <BookOpen size={20} />
        <span>Library</span>
      </Link>
      <Link to="/history" className="sidebar-link">
        <History size={20} />
        <span>History</span>
      </Link>
    </div>
  );
};

export default Sidebar;
