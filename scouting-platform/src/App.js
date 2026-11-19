import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { AuthProvider, useAuth } from "./AuthContext";
import { supabase } from "./supabaseClient";
import LandingPage from "./LandingPage";
import ScoutReportPage from "./ScoutReportPage";
import ScoutingReportsPage from "./ScoutingReportsPageFinal";
import ShortlistsPage from "./ShortlistsPage";
import SquadPlanPage from "./SquadPlanPage";
import SquadPlansPage from "./SquadPlansPage";
import CreateAssignmentPage from "./CreateAssignmentPage";
import TeamAnalysisPage from "./TeamAnalysisPage";
import ApprovalRequestsPage from "./ApprovalRequestsPage";
import ReportManagementPage from "./ReportManagementPage";
import { ClubBadge } from "./ClubBadge";
import "./App.css";
import "./mobile.css";

function LoginGate() {
  const { user, profile, profileError, loading, refreshProfile } = useAuth();
  const [signup, setSignup] = useState(false), [showAuth, setShowAuth] = useState(false), [workspaceType, setWorkspaceType] = useState("club"), [email, setEmail] = useState(""), [password, setPassword] = useState(""), [name, setName] = useState(""), [club, setClub] = useState(""), [clubs, setClubs] = useState([]), [clubsLoading, setClubsLoading] = useState(false), [clubMenuOpen, setClubMenuOpen] = useState(false), [clubActiveIndex, setClubActiveIndex] = useState(-1), [error, setError] = useState("");
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
    if (!signup || workspaceType !== "club") return;
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
  }, [signup, workspaceType]);
  if (loading) return <div className="sr-auth-screen">Loading…</div>;
  if (user && profile?.approved) return <AppRoutes />;
  if (user) {
    const independent = user.user_metadata?.workspace_type === "independent";
    return <div className="sr-auth-screen"><div className="sr-pending-card"><h1>{profile?.approval_status === "rejected" ? "Access Request Declined" : independent ? "Your Platform Request Is Being Reviewed" : "Awaiting Admin Approval"}</h1><p>{profileError || (profile?.approval_status === "rejected" ? "Your access request was declined. Contact your administrator if you think this is a mistake." : independent ? "Your independent workspace request has been received. The ScoutPro team will review the setup and subscription details with you." : "Your account has been created. A club administrator must approve your access before you can use the platform.")}</p><button className="sr-cyan" onClick={refreshProfile}>Check Approval Status</button><button className="sr-outline" onClick={() => supabase.auth.signOut()}>Sign Out</button></div></div>;
  }
  if (!showAuth) return <PublicLanding onSignIn={() => { setSignup(false); setError(""); setShowAuth(true); }} onRequestAccess={() => { setSignup(true); setWorkspaceType("club"); setError(""); setShowAuth(true); }} onOwnPlatform={() => { setSignup(true); setWorkspaceType("independent"); setClub(""); setError(""); setShowAuth(true); }} />;
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
    if (signup && workspaceType === "club" && !clubs.some((item) => item.toLocaleLowerCase() === club.trim().toLocaleLowerCase())) {
      setError("Choose your club from the suggestions so its badge and club name match."); setClubMenuOpen(true); return;
    }
    if (signup && workspaceType === "independent" && !club.trim()) {
      setError("Enter a name for your platform."); return;
    }
    const result = signup ? await supabase.auth.signUp({ email, password, options: { data: { full_name: name, club, workspace_type: workspaceType, requested_role: workspaceType === "independent" ? "club_admin" : "member" } } }) : await supabase.auth.signInWithPassword({ email, password });
    if (result.error) setError(result.error.message);
  };
  return (
    <main className="sr-auth-screen">
      <button type="button" className="sr-auth-back" onClick={() => setShowAuth(false)}>← Back To ScoutPro</button>
      <section className="sr-auth-landing">
        <div className="sr-auth-copy">
          <div className="sr-auth-brand">⚽ ScoutPro</div>
          <h1>Football Scouting,<br />Made Clear.</h1>
          <p>Manage Assignments, Reports And Shortlists For Your Club Or Independent Scouting Platform.</p>
        </div>
        <form className="sr-form sr-auth-form" onSubmit={submit}>
          <h2>{signup ? (workspaceType === "independent" ? "Set Up Your Own Platform" : "Request Club Access") : "Welcome Back"}</h2>
          {signup && <>
            <div className="sr-workspace-choice" role="group" aria-label="Choose your workspace type">
              <button type="button" className={workspaceType === "club" ? "active" : ""} onClick={() => { setWorkspaceType("club"); setClub(""); setError(""); }}>Joining A Club</button>
              <button type="button" className={workspaceType === "independent" ? "active" : ""} onClick={() => { setWorkspaceType("independent"); setClub(""); setClubMenuOpen(false); setError(""); }}>My Own Platform</button>
            </div>
            <input placeholder="Full Name" value={name} onChange={e => setName(e.target.value)} required />
            {workspaceType === "independent" ? <>
              <label className="sr-independent-workspace-label" htmlFor="independent-workspace-name">Platform Or Workspace Name</label>
              <input id="independent-workspace-name" placeholder="For example, Jamie Scouting" value={club} onChange={e => setClub(e.target.value)} required />
              <small className="sr-auth-note">For Individual Scouts Or Organisations Looking To Run Their Own Platform.</small>
            </> : <>
              <div className="sr-club-picker">
                <input role="combobox" aria-autocomplete="list" aria-expanded={clubMenuOpen && Boolean(club.trim())} aria-controls="signup-clubs" aria-activedescendant={clubActiveIndex >= 0 ? `signup-club-${clubActiveIndex}` : undefined} autoComplete="off" placeholder="Start Typing Your Club" value={club} onFocus={() => setClubMenuOpen(true)} onBlur={() => window.setTimeout(() => setClubMenuOpen(false), 180)} onChange={e => { setClub(e.target.value); setClubMenuOpen(true); setClubActiveIndex(-1); }} onKeyDown={handleClubKeyDown} required />
                {clubMenuOpen && club.trim() && <div className="sr-club-options" id="signup-clubs" role="listbox">{matchingClubs.length ? matchingClubs.map((clubName, index) => <button id={`signup-club-${index}`} className={`sr-club-option${index === clubActiveIndex ? " active" : ""}`} type="button" role="option" aria-selected={clubName === club} key={clubName} onPointerDown={event => event.preventDefault()} onMouseDown={event => event.preventDefault()} onClick={() => selectClub(clubName)}><ClubBadge club={clubName} size={30} /><span>{clubName}</span></button>) : <div className="sr-club-no-results">{clubsLoading ? "Loading Clubs…" : clubs.length ? "No Matching Clubs Found" : "Could Not Load The Club List"}</div>}</div>}
              </div>
              <small className="sr-auth-note">Your Club Administrator Will Review Your Request.</small>
            </>}
          </>}
          <input type="email" placeholder="Email Address" value={email} onChange={e => setEmail(e.target.value)} required />
          <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required minLength={6} />
          {error && <p className="sr-auth-error">{error}</p>}
          <button className="sr-cyan">{signup ? (workspaceType === "independent" ? "Request Platform Setup" : "Request Account") : "Sign In"}</button>
          <button type="button" className="sr-outline" onClick={() => { setSignup(!signup); setWorkspaceType("club"); setClubMenuOpen(false); setError(""); }}>{signup ? "Already Have An Account? Sign In" : "Create An Account"}</button>
        </form>
      </section>
    </main>
  );
}

function PublicLanding({ onSignIn, onRequestAccess, onOwnPlatform }) {
  const features = [
    { icon: "◎", title: "Player Reports", text: "Create consistent reports and bring scouting observations together." },
    { icon: "▤", title: "Assignments", text: "Coordinate fixtures, players and work across your scouting team." },
    { icon: "☆", title: "Shortlists & Squad Plans", text: "Keep targets organised and map options across every position." },
    { icon: "▥", title: "Team Analysis", text: "Explore player metrics and compare performance across teams." },
  ];
  return (
    <main className="sr-public-site">
      <header className="sr-public-nav">
        <div className="sr-public-brand"><span aria-hidden="true">⚽</span> ScoutPro</div>
        <div className="sr-public-nav-actions">
          <button type="button" className="sr-public-signin" onClick={onSignIn}>Sign In</button>
          <button type="button" className="sr-public-request" onClick={onRequestAccess}>Request Access</button>
        </div>
      </header>
      <section className="sr-public-hero">
        <div className="sr-public-hero-copy">
          <span className="sr-public-kicker">A Clearer Way To Scout</span>
          <h1>Turn Scouting Into A <em>Team Process.</em></h1>
          <p>ScoutPro Brings Player Reports, Assignments, Shortlists And Squad Planning Together In One Secure Workspace For Clubs, Teams And Independent Scouts.</p>
          <div className="sr-public-hero-actions">
            <button type="button" className="sr-public-request" onClick={onOwnPlatform}>Get Your Own Platform <span aria-hidden="true">→</span></button>
            <button type="button" className="sr-public-signin" onClick={onRequestAccess}>Request Club Access</button>
          </div>
          <small>Choose A Club Workspace Or Request An Independent Setup.</small>
        </div>
        <div className="sr-public-visual" aria-hidden="true">
        <div className="sr-public-visual-top"><span>SCOUTING WORKSPACE</span><span className="sr-public-live-dot" /></div>
          <div className="sr-public-visual-title">Player Overview</div>
          <div className="sr-public-player-card"><span className="sr-public-avatar">JW</span><div><strong>Player Assessment</strong><small>Reports · Performance · Potential</small></div><b>✓</b></div>
          <div className="sr-public-bars"><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
          <div className="sr-public-visual-footer"><span>Reports</span><span>Shortlists</span><span>Squad Plans</span></div>
        </div>
      </section>
      <section className="sr-public-features" aria-labelledby="public-features-title">
        <div className="sr-public-section-heading"><span className="sr-public-kicker">Built For Your Workflow</span><h2 id="public-features-title">Everything Your Scouting Team Needs To Stay Organised.</h2></div>
        <div className="sr-public-feature-grid">{features.map((feature) => <article className="sr-public-feature" key={feature.title}><span className="sr-public-feature-icon" aria-hidden="true">{feature.icon}</span><h3>{feature.title}</h3><p>{feature.text}</p></article>)}</div>
      </section>
      <section className="sr-public-own-platform">
        <div><span className="sr-public-kicker">FOR INDEPENDENT SCOUTS AND ORGANISATIONS</span><h2>Your Own Scouting Platform.</h2><p>Set up a workspace under your own name, manage your scouting team and keep your reports and player targets together.</p></div>
        <button type="button" className="sr-public-request" onClick={onOwnPlatform}>Request Your Own Platform <span aria-hidden="true">→</span></button>
      </section>
      <footer className="sr-public-footer"><span>ScoutPro · Football Scouting Workspace</span><button type="button" onClick={onSignIn}>Sign In</button><button type="button" onClick={onRequestAccess}>Request Club Access</button><button type="button" onClick={onOwnPlatform}>Get Your Own Platform</button></footer>
    </main>
  );
}

function AppRoutes() {
  const [shadowSquad, setShadowSquad] = useState([]);
  return <><Header /><div className="sr-app-shell" style={{ paddingTop: 80 }}><Routes><Route path="/" element={<LandingPage />} /><Route path="/scout-report" element={<ScoutReportPage shadowSquad={shadowSquad} setShadowSquad={setShadowSquad} />} /><Route path="/scouting-reports" element={<ScoutingReportsPage />} /><Route path="/shortlists" element={<ShortlistsPage />} /><Route path="/squad-plans" element={<SquadPlansPage />} /><Route path="/squad-plan" element={<SquadPlanPage />} /><Route path="/create-assignment" element={<CreateAssignmentPage />} /><Route path="/team-analysis" element={<TeamAnalysisPage />} /><Route path="/approval-requests" element={<ApprovalRequestsPage />} /><Route path="/report-management" element={<ReportManagementPage />} /></Routes></div></>;
}

// ------------------- HEADER COMPONENT -------------------
function Header() {
  const navigate = useNavigate();
  const { user, profile } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const canReviewAccess = profile?.role === "platform_admin" || profile?.role === "club_admin";

  const settingsMenuItems = () => <>
    {canReviewAccess && <button type="button" onClick={() => { goTo("/approval-requests"); setSettingsOpen(false); }}>Access approvals</button>}
    <button type="button" onClick={() => { goTo("/report-management"); setSettingsOpen(false); }}>Report Management</button>
  </>;

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
        {user && <div className="sr-settings-menu" style={{ position: "relative", marginLeft: 0, flex: "0 0 auto" }}><button type="button" className="sr-settings-trigger" aria-label="Settings" title="Settings" aria-expanded={settingsOpen} onClick={() => setSettingsOpen((open) => !open)}>⚙</button>{settingsOpen && <div className="sr-settings-dropdown">{settingsMenuItems()}</div>}</div>}
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
        {user && <div className="sr-mobile-account-actions"><div className="sr-mobile-account-buttons"><button type="button" className="sr-settings-trigger" onClick={() => setSettingsOpen((open) => !open)} aria-expanded={settingsOpen}>⚙ Settings</button><button className="header-signout" onClick={() => supabase.auth.signOut()}>Sign out</button></div>{settingsOpen && <div className="sr-settings-dropdown sr-settings-dropdown-mobile">{settingsMenuItems()}</div>}</div>}
      </nav>

      <style>{`
        .nav-link { color: #fff; cursor: pointer; margin-left: 0; font-weight: 600; position: relative; }
        .nav-link::after { content: ""; position: absolute; left: 0; bottom: -3px; width: 0%; height: 2px; background-color: #ffb74d; transition: width 0.3s ease; }
        .nav-link:hover::after { width: 100%; }
        .nav-link:hover { color: #ffb74d; }

        .desktop-nav { display: flex; align-items: center; gap: 18px; white-space: nowrap; }

        .sr-settings-menu { position: relative; margin-left: 0; }
        .sr-settings-trigger { min-height: 40px; padding: 7px 11px; border: 1px solid #ffffff66; border-radius: 8px; background: transparent; color: #fff; font-size: 20px; cursor: pointer; }
        .sr-settings-dropdown { position: absolute; top: calc(100% + 8px); right: 0; z-index: 120; display: flex; flex-direction: column; gap: 3px; width: 280px; max-width: calc(100vw - 24px); min-width: 0; box-sizing: border-box; padding: 8px; border: 1px solid #78b4d8; border-radius: 9px; background: #173f70; box-shadow: 0 12px 28px #00152e66; white-space: normal; }
        .sr-settings-dropdown button { display: block; width: 100%; box-sizing: border-box; padding: 11px 12px; border: 0; border-radius: 6px; background: transparent; color: white; font: inherit; font-weight: 700; line-height: 1.35; text-align: left; white-space: normal; cursor: pointer; }
        .sr-settings-dropdown button:hover { background: #285b8d; }
        .sr-settings-dropdown button:disabled { opacity: .6; cursor: wait; }
        .sr-settings-message { display: block; margin: 6px 8px; color: #c9efff; font-size: 12px; line-height: 1.4; white-space: normal; }
        .sr-mobile-account-actions { width: 100%; display: flex; flex-direction: column; gap: 8px; padding-top: 10px; }
        .sr-mobile-account-buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; width: 100%; }
        .sr-mobile-account-actions .sr-settings-trigger { width: 100%; font-size: 15px; }
        .sr-mobile-account-actions .header-signout { margin: 0; }
        .sr-settings-dropdown-mobile { position: static; width: 100%; max-width: none; box-shadow: none; }

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
        </Routes> */}
    </Router>
  );
}

export default App;
