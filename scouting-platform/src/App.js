import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import { useState } from "react";
import LandingPage from "./LandingPage";
import ScoutReportPage from "./ScoutReportPage";
import ShadowSquadsPage from "./ShadowSquadsPage";
import RecruitmentDashboardPage from "./RecruitmentDashboardPage";

// ------------------- HEADER COMPONENT -------------------
function Header() {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);

  const links = [
    { label: "Home", path: "/" },
    { label: "Scout Report", path: "/scout-report" },
    { label: "Shadow Squads", path: "/shadow-squads" },
  ];

  return (
    <header
      style={{
        width: "100%",
        height: 70,
        background: "#1a4d8f",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 30px",
        boxSizing: "border-box",
        position: "fixed",
        top: 0,
        left: 0,
        zIndex: 3,
        boxShadow: "0 2px 8px rgba(0,0,0,0.25)",
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        color: "#fff",
      }}
    >
      <div
        style={{ display: "flex", alignItems: "center", gap: 15, cursor: "pointer" }}
        onClick={() => navigate("/")}
      >
        <span style={{ fontSize: 24, fontWeight: 700 }}>⚽ ScoutPro</span>
      </div>

      <nav className="desktop-nav">
        {links.map((link) => (
          <span
            key={link.label}
            onClick={() => navigate(link.path)}
            className="nav-link"
          >
            {link.label}
          </span>
        ))}
        <a
          href="https://recruitmentdashboard-ggsphhjonkwlx7mqpefpaq.streamlit.app/"
          target="_blank"
          rel="noopener noreferrer"
          className="nav-link"
        >
          Recruitment Dashboard
        </a>
      </nav>

      <div className={`hamburger ${menuOpen ? "open" : ""}`} onClick={() => setMenuOpen(!menuOpen)}>
        <div />
        <div />
        <div />
      </div>

      <div className={`mobile-menu ${menuOpen ? "open" : ""}`}>
        {links.map((link, i) => (
          <span
            key={link.label}
            onClick={() => {
              navigate(link.path);
              setMenuOpen(false);
            }}
            className="nav-link"
            style={{ animationDelay: `${i * 0.1}s` }}
          >
            {link.label}
          </span>
        ))}
        <a
          href="https://recruitmentdashboard-ggsphhjonkwlx7mqpefpaq.streamlit.app/"
          target="_blank"
          rel="noopener noreferrer"
          className="nav-link"
          style={{ animationDelay: `${links.length * 0.1}s` }}
        >
          Recruitment Dashboard
        </a>
      </div>

      <style>{`
        .nav-link { color: #fff; cursor: pointer; margin-left: 25px; font-weight: 600; position: relative; }
        .nav-link::after { content: ""; position: absolute; left: 0; bottom: -3px; width: 0%; height: 2px; background-color: #ffb74d; transition: width 0.3s ease; }
        .nav-link:hover::after { width: 100%; }
        .nav-link:hover { color: #ffb74d; }

        .hamburger { display: none; flex-direction: column; justify-content: space-between; width: 25px; height: 20px; cursor: pointer; z-index: 4; }
        .hamburger div { height: 3px; background: #fff; border-radius: 2px; transition: all 0.3s ease; }
        .hamburger.open div:nth-child(1) { transform: rotate(45deg) translate(5px, 5px); }
        .hamburger.open div:nth-child(2) { opacity: 0; }
        .hamburger.open div:nth-child(3) { transform: rotate(-45deg) translate(5px, -5px); }

        .mobile-menu { position: fixed; top: 70px; left: 0; width: 100%; background: #1a4d8f; display: flex; flex-direction: column; align-items: center; gap: 15px; padding: 0; max-height: 0; overflow: hidden; transition: max-height 0.4s ease; z-index: 2; }
        .mobile-menu.open { max-height: 300px; padding: 15px 0; }
        .mobile-menu .nav-link { opacity: 0; animation: fadeIn 0.4s forwards; }
        @keyframes fadeIn { to { opacity: 1; } }

        @media (max-width: 768px) { .desktop-nav { display: none; } .hamburger { display: flex; } }
      `}</style>
    </header>
  );
}

// ------------------- MAIN APP -------------------
function App() {
  const [shadowSquad, setShadowSquad] = useState([]);

  return (
    <Router>
      <Header />
      <div style={{ paddingTop: 80 }}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route
            path="/scout-report"
            element={
              <ScoutReportPage
                shadowSquad={shadowSquad}
                setShadowSquad={setShadowSquad}
              />
            }
          />
          <Route
            path="/shadow-squads"
            element={
              <ShadowSquadsPage
                shadowSquad={shadowSquad}
                setShadowSquad={setShadowSquad} // ✅ pass setter here
              />
            }
          />
          <Route
            path="/recruitment-dashboard"
            element={<RecruitmentDashboardPage />}
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;


