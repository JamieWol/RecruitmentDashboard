import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { AuthProvider, useAuth } from "./AuthContext";
import { supabase } from "./supabaseClient";
import LandingPage from "./LandingPage";
import ScoutReportPage from "./ScoutReportPage";
import RecruitmentDashboardPage from "./RecruitmentDashboardPage";
import ScoutingReportsPage from "./ScoutingReportsPageFinal";
import ShortlistsPage from "./ShortlistsPage";
import SquadPlanPage from "./SquadPlanPage";
import SquadPlansPage from "./SquadPlansPage";
import CreateAssignmentPage from "./CreateAssignmentPage";
import TeamAnalysisPage from "./TeamAnalysisPage";
import ApprovalRequestsPage from "./ApprovalRequestsPage";
import { exportMyReports } from "./reportExport";
import { ClubBadge } from "./ClubBadge";
import "./App.css";
import "./mobile.css";

function LoginGate() {
  const { user, profile, profileError, loading, refreshProfile } = useAuth();
  const [signup, setSignup] = useState(false), [email, setEmail] = useState(""), [password, setPassword] = useState(""), [name, setName] = useState(""), [club, setClub] = useState(""), [clubs, setClubs] = useState([]), [clubsLoading, setClubsLoading] = useState(false), [clubMenuOpen, setClubMenuOpen] = useState(false), [clubActiveIndex, setClubActiveIndex] = useState(-1), [error, setError] = useState("");
  const matchingClubs = useMemo(() => {
    const query = club.trim().toLocaleLowerCase();
    if (!query) return [];
    return clubs
      .filter((name) => name.toLocaleLowerCase().includes(query))
      .sort((a, b) => {
        const aStarts = a.toLocaleLowerCase().startsWith(query);
        const bStarts = b.toLocaleLowerCase().startsWith(query);
        return Number(bStarts) - Number(aStarts) || a.localeCompare(b);
      })
      .slice(0, 8);
  }, [club, clubs]);
  useEffect(() => {
    if (!signup) return;
    let active = true;
    const loadClubs = async () => {
      setClubsLoading(true);
      try {
        // Imported player datasets use different names for the team column.
        // Probe one candidate at a time so one missing column cannot break the
        // whole signup club list, and only download the club field.
        const clubFields = ["Team", "team", "club", "Club", "team_name", "club_name"];
        let names = [];
        let lastError = null;
        for (const field of clubFields) {
          const values = [];
          let offset = 0;
          let fieldFailed = false;
          while (active) {
            const { data, error: loadError } = await supabase.from("players").select(field).range(offset, offset + 999);
            if (loadError) {
              lastError = loadError;
              fieldFailed = true;
              break;
            }
            values.push(...(data || []).map((item) => item[field]));
            if (!data || data.length < 1000) break;
            offset += 1000;
          }
          if (!active) return;
          if (!fieldFailed) {
            names = [...new Set(values.map((value) => String(value || "").trim()).filter(Boolean))];
            if (names.length) break;
          }
        }
        if (!active) return;
        names.sort((a, b) => a.localeCompare(b));
        setClubs(names);
        if (!names.length && lastError) console.error("Could not load club options", lastError);
      } catch (loadError) {
        if (!active) return;
        console.error("Could not load club options", loadError);
        setClubs([]);
      } finally {
        if (active) setClubsLoading(false);
      }
    };
    loadClubs();
    return () => { active = false; };
  }, [signup]);
  if (loading) return <div className="sr-auth-screen">Loading…</div>;
  if (user && profile?.approved) return <AppRoutes />;
  if (user) return <div className="sr-auth-screen"><div className="sr-pending-card"><h1>{profile?.approval_status === "rejected" ? "Access request declined" : "Awaiting admin approval"}</h1><p>{profileError || (profile?.approval_status === "rejected" ? "Your club has declined this account request. Contact your club administrator if you think this is a mistake." : "Your account has been created. An administrator must assign your club before you can access the platform.")}</p><button className="sr-cyan" onClick={refreshProfile}>Check approval status</button><button className="sr-outline" onClick={() => supabase.auth.signOut()}>Sign out</button></div></div>;
  const selectClub = (clubName) => { setClub(clubName); setClubMenuOpen(false); setClubActiveIndex(-1); };
  const handleClubKeyDown = (event) => {
    if (event.key === "ArrowDown" && matchingClubs.length) {
      event.preventDefault(); setClubMenuOpen(true); setClubActiveIndex((index) => (index + 1) % matchingClubs.length);
    } else if (event.key === "ArrowUp" && matchingClubs.length) {
      event.preventDefault(); setClubMenuOpen(true); setClubActiveIndex((index) => (index <= 0 ? matchingClubs.length - 1 : index - 1));
    } else if (event.key === "Enter" && clubMenuOpen && matchingClubs.length) {
      event.preventDefault(); selectClub(matchingClubs[Math.max(0, clubActiveIndex)]);
    } else if (event.key === "Escape") setClubMenuOpen(false);
  };
  const submit = async (e) => {
    e.preventDefault(); setError("");
    if (signup && !clubs.some((item) => item.toLocaleLowerCase() === club.trim().toLocaleLowerCase())) {
      setError("Choose your club from the suggestions so its badge and club name match."); setClubMenuOpen(true); return;
    }
    const result = signup ? await supabase.auth.signUp({ email, password, options: { data: { full_name: name, club } } }) : await supabase.auth.signInWithPassword({ email, password });
    if (result.error) setError(result.error.message);
  };
  return <main className="sr-auth-screen"><section className="sr-auth-landing"><div className="sr-auth-copy"><div className="sr-auth-brand">⚽ ScoutPro</div><h1>Football Recruitment<br />Organised Properly.</h1><p>Manage assignments, reports and shortlists securely with your scouting team.</p></div><form className="sr-form sr-auth-form" onSubmit={submit}><h2>{signup ? "Request access" : "Welcome back"}</h2>{signup && <><input placeholder="Full name" value={name} onChange={e => setName(e.target.value)} required /><div className="sr-club-picker"><input role="combobox" aria-autocomplete="list" aria-expanded={clubMenuOpen && Boolean(club.trim())} aria-controls="signup-clubs" aria-activedescendant={clubActiveIndex >= 0 ? `signup-club-${clubActiveIndex}` : undefined} autoComplete="off" placeholder="Start typing your club" value={club} onFocus={() => setClubMenuOpen(true)} onBlur={() => window.setTimeout(() => setClubMenuOpen(false), 180)} onChange={e => { setClub(e.target.value); setClubMenuOpen(true); setClubActiveIndex(-1); }} onKeyDown={handleClubKeyDown} required />{clubMenuOpen && club.trim() && <div className="sr-club-options" id="signup-clubs" role="listbox">{matchingClubs.length ? matchingClubs.map((clubName, index) => <button id={`signup-club-${index}`} className={`sr-club-option${index === clubActiveIndex ? " active" : ""}`} type="button" role="option" aria-selected={clubName === club} key={clubName} onPointerDown={(event) => event.preventDefault()} onMouseDown={(event) => event.preventDefault()} onClick={() => selectClub(clubName)}><ClubBadge club={clubName} size={30} /><span>{clubName}</span></button>) : <div className="sr-club-no-results">{clubsLoading ? "Loading clubs…" : clubs.length ? "No matching clubs found" : "Could not load the club list"}</div>}</div>}</div><small className="sr-auth-note">Your club admin will review your request.</small></>}<input type="email" placeholder="Email address" value={email} onChange={e => setEmail(e.target.value)} required /><input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required minLength={6} />{error && <p className="sr-auth-error">{error}</p>}<button className="sr-cyan">{signup ? "Request account" : "Sign in"}</button><button type="button" className="sr-outline" onClick={() => { setSignup(!signup); setClubMenuOpen(false); setError(""); }}>{signup ? "Already have an account? Sign in" : "Create an account"}</button></form></section></main>;
}

function AppRoutes() {
  const [shadowSquad, setShadowSquad] = useState([]);
  return <><Header /><div className="sr-app-shell" style={{ paddingTop: 80 }}><Routes><Route path="/" element={<LandingPage />} /><Route path="/scout-report" element={<ScoutReportPage shadowSquad={shadowSquad} setShadowSquad={setShadowSquad} />} /><Route path="/scouting-reports" element={<ScoutingReportsPage />} /><Route path="/shortlists" element={<ShortlistsPage />} /><Route path="/squad-plans" element={<SquadPlansPage />} /><Route path="/squad-plan" element={<SquadPlanPage />} /><Route path="/create-assignment" element={<CreateAssignmentPage />} /><Route path="/team-analysis" element={<TeamAnalysisPage />} /><Route path="/approval-requests" element={<ApprovalRequestsPage />} /><Route path="/recruitment-dashboard" element={<RecruitmentDashboardPage />} /></Routes></div></>;
}

// ------------------- HEADER COMPONENT -------------------
function Header() {
  const navigate = useNavigate();
  const { user, profile, appState } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [exporting, setExporting] = useState("");
  const [settingsMessage, setSettingsMessage] = useState("");
  const canReviewAccess = profile?.role === "platform_admin" || profile?.role === "club_admin";

  const exportReports = async (format) => {
    setExporting(format);
    setSettingsMessage("");
    try {
      const count = await exportMyReports({ appState, user, profile, format });
      setSettingsMessage(`${count} report${count === 1 ? "" : "s"} exported.`);
    } catch (error) {
      setSettingsMessage(error?.message || "Could not export your reports.");
    } finally {
      setExporting("");
    }
  };

  const goTo = (path) => {
    if (path === "/scouting-reports") {
      localStorage.removeItem("scoutingProfilePlayer");
      localStorage.removeItem("scoutingProfileOrigin");
    }
    navigate(path);
  };

  const links = [
    { label: "Home", path: "/" },
    { label: "Scout Report", path: "/scout-report" },
    { label: "Reports & Assignments", path: "/scouting-reports" },
    { label: "Shortlists", path: "/shortlists" },
    { label: "Squad Plans", path: "/squad-plans" },
    { label: "Team Analysis", path: "/team-analysis" },
  ];

  return (
    <header
      className="sr-app-header"
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
        <span style={{ fontSize: 24, fontWeight: 700 }}>&#9917; ScoutPro</span>
      </div>

      <nav className="desktop-nav" style={{ display: "flex", flexDirection: "row", flexWrap: "nowrap", alignItems: "center", justifyContent: "flex-end", gap: 18, whiteSpace: "nowrap", minWidth: 0 }}>
        {links.map((link) => (
          <span
            key={link.label}
            onClick={() => goTo(link.path)}
            className="nav-link"
          >
            {link.label}
          </span>
        ))}
        <a
          href="https://recruitmentdashboard-iffxqqyyyuvoe2xamnaldh.streamlit.app/"
          target="_blank"
          rel="noopener noreferrer"
          className="nav-link"
        >
          Recruitment Dashboard
        </a>
        {user && <div className="sr-settings-menu" style={{ marginLeft: 0, flex: "0 0 auto" }}><button type="button" className="sr-settings-trigger" aria-label="Settings" title="Settings" aria-expanded={settingsOpen} onClick={() => { setSettingsOpen((open) => !open); setSettingsMessage(""); }}>⚙</button>{settingsOpen && <div className="sr-settings-dropdown">{canReviewAccess && <button type="button" onClick={() => { goTo("/approval-requests"); setSettingsOpen(false); }}>Access approvals</button>}<button type="button" disabled={Boolean(exporting)} onClick={() => exportReports("xlsx")}>{exporting === "xlsx" ? "Exporting…" : "Export my reports to Excel"}</button><button type="button" disabled={Boolean(exporting)} onClick={() => exportReports("csv")}>{exporting === "csv" ? "Exporting…" : "Export my reports to CSV"}</button>{settingsMessage && <p className="sr-settings-message" role="status">{settingsMessage}</p>}</div>}</div>}
        {user && <button className="header-signout" style={{ marginLeft: 0, flex: "0 0 auto" }} onClick={() => supabase.auth.signOut()}>Sign out</button>}
      </nav>

      <button type="button" className={`hamburger ${menuOpen ? "open" : ""}`} aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"} aria-expanded={menuOpen} aria-controls="mobile-navigation" onClick={() => setMenuOpen(!menuOpen)}>
        <span />
        <span />
        <span />
      </button>

      <nav id="mobile-navigation" className={`mobile-menu ${menuOpen ? "open" : ""}`} aria-label="Main navigation">
        {links.map((link, i) => (
          <span
            key={link.label}
            onClick={() => {
              goTo(link.path);
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
          onClick={() => setMenuOpen(false)}
        >
          Recruitment Dashboard
        </a>
        {user && <div className="sr-mobile-account-actions"><button type="button" className="sr-settings-trigger" onClick={() => { setSettingsOpen((open) => !open); setSettingsMessage(""); }} aria-expanded={settingsOpen}>⚙ Settings</button><button className="header-signout" onClick={() => supabase.auth.signOut()}>Sign out</button>{settingsOpen && <>{canReviewAccess && <button type="button" className="sr-settings-mobile-link" onClick={() => { goTo("/approval-requests"); setSettingsOpen(false); setMenuOpen(false); }}>Access approvals</button>}<button type="button" className="sr-settings-mobile-link" disabled={Boolean(exporting)} onClick={() => exportReports("xlsx")}>{exporting === "xlsx" ? "Exporting…" : "Export my reports to Excel"}</button><button type="button" className="sr-settings-mobile-link" disabled={Boolean(exporting)} onClick={() => exportReports("csv")}>{exporting === "csv" ? "Exporting…" : "Export my reports to CSV"}</button>{settingsMessage && <span className="sr-settings-message" role="status">{settingsMessage}</span>}</>}</div>}
      </nav>

      <style>{`
        .nav-link { color: #fff; cursor: pointer; margin-left: 0; font-weight: 600; position: relative; }
        .nav-link::after { content: ""; position: absolute; left: 0; bottom: -3px; width: 0%; height: 2px; background-color: #ffb74d; transition: width 0.3s ease; }
        .nav-link:hover::after { width: 100%; }
        .nav-link:hover { color: #ffb74d; }

        .desktop-nav { display: flex; align-items: center; gap: 18px; white-space: nowrap; }

        .sr-settings-menu { position: relative; margin-left: 20px; }
        .sr-settings-trigger { min-height: 40px; padding: 7px 11px; border: 1px solid #ffffff66; border-radius: 8px; background: transparent; color: #fff; font-size: 20px; cursor: pointer; }
        .sr-settings-dropdown { position: absolute; top: calc(100% + 8px); right: 0; min-width: 190px; padding: 6px; border: 1px solid #78b4d8; border-radius: 9px; background: #173f70; box-shadow: 0 12px 28px #00152e66; }
        .sr-settings-dropdown button,.sr-settings-mobile-link { width: 100%; padding: 11px 12px; border: 0; border-radius: 6px; background: transparent; color: white; font: inherit; font-weight: 700; text-align: left; cursor: pointer; }
        .sr-settings-dropdown button:hover,.sr-settings-mobile-link:hover { background: #285b8d; }
        .sr-settings-dropdown button:disabled,.sr-settings-mobile-link:disabled { opacity: .6; cursor: wait; }
        .sr-settings-message { display: block; margin: 6px 8px; color: #c9efff; font-size: 12px; line-height: 1.4; white-space: normal; }
        .sr-mobile-account-actions { width: 100%; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding-top: 10px; }
        .sr-mobile-account-actions .sr-settings-trigger { width: 100%; font-size: 15px; }
        .sr-mobile-account-actions .header-signout { margin: 0; }
        .sr-settings-mobile-link { grid-column: 1 / -1; border: 1px solid #ffffff30; }

        .hamburger { display: none; flex-direction: column; justify-content: space-between; width: 25px; height: 20px; cursor: pointer; z-index: 4; }
        .hamburger span { height: 3px; background: #fff; border-radius: 2px; transition: all 0.3s ease; }
        .hamburger.open span:nth-child(1) { transform: rotate(45deg) translate(5px, 5px); }
        .hamburger.open span:nth-child(2) { opacity: 0; }
        .hamburger.open span:nth-child(3) { transform: rotate(-45deg) translate(5px, -5px); }

        .mobile-menu { position: fixed; top: 70px; left: 0; width: 100%; background: #1a4d8f; display: flex; flex-direction: column; align-items: center; gap: 15px; padding: 0; max-height: 0; overflow: hidden; transition: max-height 0.4s ease; z-index: 2; }
        .mobile-menu.open { max-height: 300px; padding: 15px 0; }
        .mobile-menu .nav-link { opacity: 0; animation: fadeIn 0.4s forwards; }
        @keyframes fadeIn { to { opacity: 1; } }

        @media (max-width: 1100px) { .desktop-nav { display: none !important; } .hamburger { display: flex; } }
      `}</style>
    </header>
  );
}

// ------------------- MAIN APP -------------------
function App() {
  return (
    <Router>
      <AuthProvider><LoginGate /></AuthProvider>
      {/* <Header />
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
          <Route path="/scouting-reports" element={<ScoutingReportsPage />} />
          <Route path="/shortlists" element={<ShortlistsPage />} />
          <Route path="/create-assignment" element={<CreateAssignmentPage />} />
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
        </Routes> */}
    </Router>
  );
}

export default App;
