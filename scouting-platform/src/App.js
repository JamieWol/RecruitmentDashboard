import { BrowserRouter as Router, Routes, Route, useNavigate, useSearchParams } from "react-router-dom";
import { useEffect, useMemo, useRef, useState } from "react";
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
import BillingPage from "./BillingPage";
import { ClubBadge } from "./ClubBadge";
import { hasFullToolAccess } from "./planAccess";
import "./App.css";
import "./mobile.css";

function FullToolsGate({ children }) {
  const navigate = useNavigate();
  const { profile, entitlement, workspaceEntitlement } = useAuth();
  if (hasFullToolAccess(profile, entitlement, workspaceEntitlement)) return children;
  return <main className="sr-page sr-plan-locked"><section className="sr-plan-locked-card">
    <span className="sr-kicker">PREMIUM</span>
    <h1>Unlock Shortlists, Squad Plans And Team Analysis</h1>
    <p>Your current plan includes player reports and assignments. Premium adds shortlists, squad plans and team analysis. Start or manage a subscription to unlock these tools.</p>
    <div className="sr-plan-locked-actions"><button type="button" className="sr-cyan" onClick={() => navigate("/billing")}>View Plans And Billing</button><button type="button" className="sr-outline" onClick={() => navigate("/scouting-reports")}>Back To Reports And Assignments</button></div>
  </section></main>;
}

function LoginGate() {
  const navigate = useNavigate();
  const { user, profile, profileError, loading, refreshProfile } = useAuth();
  const [searchParams] = useSearchParams();
  const [signup, setSignup] = useState(false), [showAuth, setShowAuth] = useState(false), [workspaceType, setWorkspaceType] = useState("club"), [preferredPlan, setPreferredPlan] = useState(""), [email, setEmail] = useState(""), [password, setPassword] = useState(""), [name, setName] = useState(""), [club, setClub] = useState(""), [clubs, setClubs] = useState([]), [clubsLoading, setClubsLoading] = useState(false), [clubMenuOpen, setClubMenuOpen] = useState(false), [clubActiveIndex, setClubActiveIndex] = useState(-1), [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [checkoutBusy, setCheckoutBusy] = useState(false);
  const paidIndependentRequest = profile?.workspace_type === "independent"
    && ["Independent Scout", "Independent Scout + Full Tools Bundle"].includes(profile?.requested_plan)
    && profile?.approval_status !== "rejected";
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
        // A club can be registered by its admin before any player records
        // exist. Include club names already present on profiles so new
        // members can request access to those workspaces straight away.
        let registeredClubs = [];
        const { data: registeredData, error: registeredError } = await supabase.rpc("get_signup_clubs");
        if (!registeredError) {
          registeredClubs = (registeredData || [])
            .map((item) => String(item.club || "").trim())
            .filter(Boolean);
        }

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
        names = [...new Set([...registeredClubs, ...names])];
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
  const startPendingCheckout = async (interval = "monthly") => {
    setCheckoutBusy(true);
    setError("");
    const plan = profile?.requested_plan === "Independent Scout + Full Tools Bundle" ? "independent_bundle" : "independent";
    const { data, error: invokeError } = await supabase.functions.invoke("create-checkout-session", { body: { plan, interval: plan === "independent_bundle" ? "monthly" : interval } });
    setCheckoutBusy(false);
    if (invokeError || !data?.url) {
      setError(data?.error || invokeError?.message || "Could not open secure checkout. Please try again.");
      return;
    }
    window.location.assign(data.url);
  };
  usePendingCheckoutRefresh(searchParams.get("checkout"), refreshProfile, paidIndependentRequest);
  if (loading) return <div className="sr-auth-screen">Loading…</div>;
  if (user && profile?.approved) return <AppRoutes />;
  if (user) {
    const ownerSetup = ["independent", "organization"].includes(user.user_metadata?.workspace_type);
    const trialRequest = user.user_metadata?.requested_plan === "Independent Scout Free Trial";
    if (paidIndependentRequest) return <div className="sr-auth-screen"><div className="sr-pending-card">
      <h1>{searchParams.get("checkout") === "success" ? "Confirming Your Subscription" : "Finish Setting Up Your Independent Account"}</h1>
      <p>{searchParams.get("checkout") === "success"
        ? "Payment completed. Stripe is confirming your subscription and activating your workspace. This usually takes a few seconds."
        : searchParams.get("checkout") === "canceled"
          ? "Checkout was canceled, so no subscription was started. You can try again whenever you’re ready."
          : "Your account is ready. Choose a billing option to continue to secure Stripe Checkout. Your workspace will activate when Stripe confirms the subscription."}</p>
      {error && <p className="sr-auth-error" role="alert">{error}</p>}
      <button className="sr-cyan" type="button" disabled={checkoutBusy} onClick={() => startPendingCheckout("monthly")}>{checkoutBusy ? "Opening Checkout…" : profile.requested_plan === "Independent Scout + Full Tools Bundle" ? "Subscribe To Premium · £17.98 / Month" : "Subscribe Monthly · £9.99"}</button>
      {profile.requested_plan !== "Independent Scout + Full Tools Bundle" && <button className="sr-outline" type="button" disabled={checkoutBusy} onClick={() => startPendingCheckout("yearly")}>Subscribe Yearly · £99</button>}
      <button className="sr-outline" type="button" onClick={refreshProfile}>Check Subscription Status</button>
      <button className="sr-outline" type="button" onClick={() => supabase.auth.signOut()}>Sign Out</button>
    </div></div>;
    const pendingTitle = profile?.approval_status === "revoked" ? "Access Revoked" : profile?.approval_status === "rejected" ? "Access Request Declined" : trialRequest ? "Your Free Trial Request Is Being Reviewed" : ownerSetup ? "Your Platform Request Is Being Reviewed" : "Awaiting Admin Approval";
    const pendingMessage = profileError || (profile?.approval_status === "revoked"
      ? "The main ScoutPro administrator has revoked access to this account. Contact them if you think this is a mistake."
      : profile?.approval_status === "rejected"
      ? "Your access request was declined. Contact your administrator if you think this is a mistake."
      : trialRequest
        ? "Once approved, your 30-day trial will begin. It includes up to 4 published reports and does not require payment details."
        : ownerSetup
          ? "Your workspace and pricing request has been received. The ScoutPro team will review the setup details with you."
          : "Your account has been created. A club administrator must approve your access before you can use the platform.");
    return <div className="sr-auth-screen"><div className="sr-pending-card"><h1>{pendingTitle}</h1><p>{pendingMessage}</p>{notice && <p role="status">{notice}</p>}<button className="sr-cyan" onClick={refreshProfile}>Check Approval Status</button><button className="sr-outline" onClick={() => supabase.auth.signOut()}>Sign Out</button></div></div>;
  }
  if (!showAuth) return <PublicLanding onSignIn={() => { setSignup(false); setError(""); setShowAuth(true); }} onRequestAccess={() => { setSignup(true); setWorkspaceType("club"); setPreferredPlan(""); setError(""); setShowAuth(true); }} onChoosePlan={(plan, type) => { setSignup(true); setPreferredPlan(plan); setWorkspaceType(type); setClub(""); setError(""); setShowAuth(true); }} />;
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
    if (signup && workspaceType !== "club" && !club.trim()) {
      setError("Enter a name for your workspace."); return;
    }
    const result = signup ? await supabase.auth.signUp({ email, password, options: { data: { full_name: name, club, workspace_type: workspaceType, requested_role: workspaceType === "club" ? "member" : "club_admin", requested_plan: preferredPlan } } }) : await supabase.auth.signInWithPassword({ email, password });
    if (result.error) setError(result.error.message);
    else if (!signup && result.data?.session) navigate("/", { replace: true });
    else if (signup && !result.data?.session) setNotice("Account created. Check your email to confirm it, then sign in to complete your subscription.");
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
          <h2>{signup ? (workspaceType === "club" ? "Request Club Access" : workspaceType === "organization" ? "Request Club Workspace Setup" : preferredPlan === "Independent Scout Free Trial" ? "Request Free Trial" : "Create Independent Account") : "Welcome Back"}</h2>
          {signup && <>
            {preferredPlan && <div className="sr-selected-plan"><small>Requested Setup</small><strong>{preferredPlan}</strong></div>}
            <div className="sr-workspace-choice" role="group" aria-label="Choose your workspace type">
              <button type="button" className={workspaceType === "club" ? "active" : ""} onClick={() => { setWorkspaceType("club"); setPreferredPlan(""); setClub(""); setError(""); }}>Joining A Club</button>
              <button type="button" className={workspaceType === "independent" ? "active" : ""} onClick={() => { setWorkspaceType("independent"); setPreferredPlan("Independent Scout"); setClub(""); setClubMenuOpen(false); setError(""); }}>Independent</button>
              <button type="button" className={workspaceType === "organization" ? "active" : ""} onClick={() => { setWorkspaceType("organization"); setPreferredPlan("Club Workspace"); setClub(""); setClubMenuOpen(false); setError(""); }}>Own Club</button>
            </div>
            <input placeholder="Full Name" value={name} onChange={e => setName(e.target.value)} required />
            {workspaceType !== "club" ? <>
              <label className="sr-independent-workspace-label" htmlFor="independent-workspace-name">{workspaceType === "organization" ? "Club Or Organisation Name" : "Platform Or Workspace Name"}</label>
              <input id="independent-workspace-name" placeholder={workspaceType === "organization" ? "Enter Your Club Or Organisation" : "For Example, Jamie Scouting"} value={club} onChange={e => setClub(e.target.value)} required />
              <small className="sr-auth-note">{workspaceType === "independent" && preferredPlan !== "Independent Scout Free Trial" ? "You’ll Choose Your Billing Option And Complete Secure Checkout After Creating Your Account." : "We’ll Review Your Workspace Setup And Requested Plan With You."}</small>
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
          {error && <p className="sr-auth-error" role="alert">{error}</p>}
          {notice && <p className="sr-auth-note" role="status">{notice}</p>}
          <button className="sr-cyan">{signup ? (workspaceType === "club" ? "Request Account" : workspaceType === "independent" && preferredPlan === "Independent Scout Free Trial" ? "Request Free Trial" : workspaceType === "independent" ? "Create Account & Continue" : "Request Setup And Pricing") : "Sign In"}</button>
          <button type="button" className="sr-outline" onClick={() => { setSignup(!signup); setWorkspaceType("club"); setClubMenuOpen(false); setError(""); }}>{signup ? "Already Have An Account? Sign In" : "Create An Account"}</button>
        </form>
      </section>
    </main>
  );
}

function usePendingCheckoutRefresh(checkoutState, refreshProfile, shouldRefresh) {
  useEffect(() => {
    if (checkoutState !== "success" || !shouldRefresh) return undefined;
    let checks = 0;
    const timer = window.setInterval(() => {
      checks += 1;
      refreshProfile().catch(() => {});
      if (checks >= 12) window.clearInterval(timer);
    }, 2500);
    return () => window.clearInterval(timer);
  }, [checkoutState, refreshProfile, shouldRefresh]);
}

function PublicLanding({ onSignIn, onRequestAccess, onChoosePlan }) {
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
            <button type="button" className="sr-public-request" onClick={() => onChoosePlan("Independent Scout", "independent")}>Get Your Own Platform <span aria-hidden="true">→</span></button>
            <button type="button" className="sr-public-signin" onClick={onRequestAccess}>Request Club Access</button>
          </div>
          <small>Choose A Club Workspace Or Request An Independent Setup.</small>
        </div>
        <div className="sr-public-showcase" aria-label="Examples of ScoutPro screens">
          <article className="sr-product-window sr-product-report">
            <div className="sr-product-window-bar"><span className="sr-product-window-dots"><i /><i /><i /></span><span>PLAYER REPORT</span><span className="sr-product-preview-tag">PREVIEW</span></div>
            <div className="sr-product-player"><span className="sr-product-player-avatar">EP</span><div><strong>Example Player</strong><small>Position · Club</small></div><span className="sr-product-status">Reviewed</span></div>
            <div className="sr-product-grades"><div><small>Potential</small><strong>B</strong></div><div><small>Performance</small><strong>4 / 5</strong></div><div><small>Reports</small><strong>3</strong></div></div>
            <div className="sr-product-report-text"><strong>Player Summary</strong><span>Reports, observations and scout assessments in one profile.</span></div>
          </article>
          <div className="sr-product-window-row">
            <article className="sr-product-window sr-product-shortlist">
              <div className="sr-product-window-bar"><span className="sr-product-window-dots"><i /><i /><i /></span><span>SHORTLIST</span></div>
              <strong className="sr-product-window-heading">Summer Targets</strong>
              <div className="sr-product-list-row"><b>LW</b><span>Example Player</span><em>Added</em></div>
              <div className="sr-product-list-row"><b>CM</b><span>Example Player</span><em>Added</em></div>
              <div className="sr-product-list-row"><b>CB</b><span>Example Player</span><em>Added</em></div>
            </article>
            <article className="sr-product-window sr-product-squad">
              <div className="sr-product-window-bar"><span className="sr-product-window-dots"><i /><i /><i /></span><span>SQUAD PLAN</span></div>
              <strong className="sr-product-window-heading">4-3-3 Formation</strong>
              <div className="sr-product-pitch">
                <div><i>ST</i></div><div><i>LW</i><i>CF</i><i>RW</i></div><div><i>CM</i><i>CM</i><i>CM</i></div><div><i>LB</i><i>CB</i><i>CB</i><i>RB</i></div><div><i>GK</i></div>
              </div>
            </article>
          </div>
        </div>
      </section>
      <section className="sr-public-features" aria-labelledby="public-features-title">
        <div className="sr-public-section-heading"><span className="sr-public-kicker">Built For Your Workflow</span><h2 id="public-features-title">Everything Your Scouting Team Needs To Stay Organised.</h2></div>
        <div className="sr-public-feature-grid">{features.map((feature) => <article className="sr-public-feature" key={feature.title}><span className="sr-public-feature-icon" aria-hidden="true">{feature.icon}</span><h3>{feature.title}</h3><p>{feature.text}</p></article>)}</div>
      </section>
      <section className="sr-public-pricing" aria-labelledby="public-pricing-title">
        <div className="sr-public-section-heading"><span className="sr-public-kicker">Plans For Different Scouting Setups</span><h2 id="public-pricing-title">Choose The Workspace That Fits Your Team.</h2><p>Club workspaces include the full ScoutPro toolkit. Independent scouts can subscribe securely online, starting with the core tools and adding more when they need them. Free trial and club access requests are reviewed before activation.</p></div>
        <div className="sr-public-plan-grid">
          <article className="sr-public-plan-card">
            <span className="sr-public-plan-label">TRY SCOUTPRO</span><h3>Free Trial</h3><p>Try the core scouting workflow before choosing a plan.</p><div className="sr-public-plan-price">Free</div><small className="sr-public-plan-term">30 days or 4 completed reports, whichever comes first</small><ul><li>Player reports</li><li>Reports and assignments</li><li>No payment details requested</li></ul><button type="button" className="sr-public-outline-button" onClick={() => onChoosePlan("Independent Scout Free Trial", "independent")}>Request Free Trial</button>
          </article>
          <article className="sr-public-plan-card">
            <span className="sr-public-plan-label">FOR INDIVIDUAL SCOUTS</span><h3>Independent Scout</h3><p>Your own workspace with the essential tools to scout and track players.</p><div className="sr-public-plan-price">£9.99 <small>/ month</small></div><small className="sr-public-plan-term">Or £99 per year</small><ul><li>Player reports and summaries</li><li>Reports and assignments</li><li>One scout workspace</li></ul><button type="button" className="sr-public-request" onClick={() => onChoosePlan("Independent Scout", "independent")}>Choose Independent Scout</button>
          </article>
          <article className="sr-public-plan-card">
            <span className="sr-public-plan-label">OPTIONAL ADD-ON</span><h3>Premium</h3><p>Expand your Independent Scout workspace with more ways to organise and analyse targets.</p><div className="sr-public-plan-price">+£7.99 <small>/ month</small></div><small className="sr-public-plan-term">Add to the Independent Scout plan</small><ul><li>Shortlists</li><li>Squad plans</li><li>Team analysis</li></ul><button type="button" className="sr-public-outline-button" onClick={() => onChoosePlan("Independent Scout + Full Tools Bundle", "independent")}>Choose Premium</button>
          </article>
          <article className="sr-public-plan-card sr-public-plan-card-featured">
            <span className="sr-public-plan-label">FOR CLUBS · ALL CORE FEATURES</span><h3>Club Workspace</h3><p>A shared platform for scouts and recruitment staff at one club, with the full ScoutPro toolkit.</p><div className="sr-public-plan-price">From £49 <small>/ month</small></div><small className="sr-public-plan-term">Includes up to 7 users</small><ul><li>Reports, summaries and assignments</li><li>Shared shortlists and squad plans</li><li>Team analysis tools</li></ul><button type="button" className="sr-public-request" onClick={() => onChoosePlan("Club Workspace", "organization")}>Request This Plan</button>
          </article>
          <article className="sr-public-plan-card">
            <span className="sr-public-plan-label">FOR MULTI-TEAM SETUPS</span><h3>Larger Organisation</h3><p>For larger scouting departments, multiple teams or a tailored setup.</p><div className="sr-public-plan-price">Custom</div><small className="sr-public-plan-term">Pricing based on your requirements</small><ul><li>Workspace requirements review</li><li>Team access and setup discussion</li><li>Tailored pricing</li></ul><button type="button" className="sr-public-outline-button" onClick={() => onChoosePlan("Larger Organisation", "organization")}>Discuss Your Setup</button>
          </article>
        </div>
      </section>
      <footer className="sr-public-footer"><span>ScoutPro · Football Scouting Workspace</span><button type="button" onClick={onSignIn}>Sign In</button><button type="button" onClick={onRequestAccess}>Request Club Access</button><button type="button" onClick={() => onChoosePlan("Independent Scout", "independent")}>Get Your Own Platform</button></footer>
    </main>
  );
}

function AppRoutes() {
  const [shadowSquad, setShadowSquad] = useState([]);
  return <><Header /><div className="sr-app-shell" style={{ paddingTop: 80 }}><Routes><Route path="/" element={<LandingPage />} /><Route path="/scout-report" element={<ScoutReportPage shadowSquad={shadowSquad} setShadowSquad={setShadowSquad} />} /><Route path="/scouting-reports" element={<ScoutingReportsPage />} /><Route path="/shortlists" element={<FullToolsGate><ShortlistsPage /></FullToolsGate>} /><Route path="/squad-plans" element={<FullToolsGate><SquadPlansPage /></FullToolsGate>} /><Route path="/squad-plan" element={<FullToolsGate><SquadPlanPage /></FullToolsGate>} /><Route path="/create-assignment" element={<CreateAssignmentPage />} /><Route path="/team-analysis" element={<FullToolsGate><TeamAnalysisPage /></FullToolsGate>} /><Route path="/billing" element={<BillingPage />} /><Route path="/approval-requests" element={<ApprovalRequestsPage />} /><Route path="/report-management" element={<ReportManagementPage />} /></Routes></div></>;
}

// ------------------- HEADER COMPONENT -------------------
function Header() {
  const navigate = useNavigate();
  const { user, profile } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const profileMenuRef = useRef(null);
  const canReviewAccess = profile?.role === "platform_admin" || profile?.role === "club_admin";
  const profileName = profile?.full_name || user?.user_metadata?.full_name || user?.email?.split("@")[0] || "ScoutPro account";
  const profileEmail = user?.email || "";
  const profileInitials = profileName.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "SP";

  useEffect(() => {
    if (!settingsOpen) return undefined;
    const closeOnOutsideClick = (event) => {
      if (!profileMenuRef.current?.contains(event.target)) setSettingsOpen(false);
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setSettingsOpen(false);
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("touchstart", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("touchstart", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [settingsOpen]);

  const settingsMenuItems = () => <>
    {canReviewAccess && <button type="button" role="menuitem" onClick={() => { goTo("/approval-requests"); setSettingsOpen(false); setMenuOpen(false); }}>Access approvals</button>}
    <button type="button" role="menuitem" onClick={() => { goTo("/billing"); setSettingsOpen(false); setMenuOpen(false); }}>Plans &amp; Billing</button>
    <button type="button" role="menuitem" onClick={() => { goTo("/report-management"); setSettingsOpen(false); setMenuOpen(false); }}>Report Management</button>
    <div className="sr-profile-dropdown-divider" />
    <button type="button" role="menuitem" className="sr-profile-signout" onClick={() => { setSettingsOpen(false); setMenuOpen(false); supabase.auth.signOut(); }}>Sign out</button>
  </>;

  const profileMenu = (mobile = false) => <div className={`sr-profile-menu${mobile ? " sr-profile-menu-mobile" : ""}`}>
    <button type="button" className="sr-profile-trigger" aria-label={`Profile menu for ${profileName}`} aria-haspopup="menu" aria-expanded={settingsOpen} onClick={() => setSettingsOpen((open) => !open)}>
      <span className="sr-profile-avatar" aria-hidden="true">{profileInitials}</span>
      <span className="sr-profile-trigger-name">{profileName}</span>
      <svg className="sr-profile-trigger-chevron" viewBox="0 0 20 20" aria-hidden="true"><path d="m5 7.5 5 5 5-5" /></svg>
    </button>
    {settingsOpen && <div className={`sr-settings-dropdown sr-profile-dropdown${mobile ? " sr-settings-dropdown-mobile" : ""}`} role="menu">
      <div className="sr-profile-dropdown-heading"><span className="sr-profile-avatar" aria-hidden="true">{profileInitials}</span><span className="sr-profile-summary"><strong>{profileName}</strong><small>{profileEmail}</small></span></div>
      <div className="sr-profile-dropdown-divider" />
      {settingsMenuItems()}
    </div>}
  </div>;

  const goTo = (path) => {
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
      ref={profileMenuRef}
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
        {user && profileMenu()}
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
        {user && profileMenu(true)}
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
        .sr-profile-menu { position: relative; flex: 0 1 auto; width: auto; max-width: 230px; }
        .sr-profile-trigger { display: flex; align-items: center; gap: 9px; width: auto; max-width: 100%; min-width: 48px; min-height: 48px; padding: 4px 10px 4px 4px; border: 1px solid #ffffff45; border-radius: 28px; background: #ffffff0b; color: white; text-align: left; cursor: pointer; }
        .sr-profile-trigger:hover, .sr-profile-trigger[aria-expanded="true"] { background: #ffffff18; border-color: #ffffff80; }
        .sr-profile-trigger-name { overflow: hidden; color: white; font-size: 13px; font-weight: 700; text-overflow: ellipsis; white-space: nowrap; }
        .sr-profile-trigger-chevron { width: 13px; height: 13px; flex: 0 0 13px; fill: none; stroke: currentColor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 2.2; }
        .sr-profile-avatar { display: grid; place-items: center; width: 38px; height: 38px; flex: 0 0 38px; border: 1.5px solid #d5f2ff; border-radius: 50%; color: #d5f2ff; font-size: 13px; font-weight: 800; }
        .sr-profile-summary { display: flex; flex: 1 1 auto; flex-direction: column; min-width: 0; gap: 2px; }
        .sr-profile-summary strong, .sr-profile-summary small { overflow: hidden; color: white; text-overflow: ellipsis; white-space: nowrap; }
        .sr-profile-summary strong { font-size: 12px; }
        .sr-profile-summary small { color: #d1e4f7; font-size: 10px; }
        .sr-profile-dropdown { width: 300px; padding: 10px; }
        .sr-profile-dropdown-heading { display: flex; align-items: center; gap: 11px; min-width: 0; padding: 7px 7px 12px; }
        .sr-profile-dropdown-heading .sr-profile-avatar { width: 42px; height: 42px; flex-basis: 42px; }
        .sr-profile-dropdown-heading .sr-profile-summary strong { font-size: 14px; }
        .sr-profile-dropdown-heading .sr-profile-summary small { font-size: 12px; }
        .sr-profile-dropdown-divider { height: 1px; margin: 3px 4px 6px; background: #ffffff24; }
        .sr-settings-dropdown .sr-profile-signout { color: #ffd2cc; }
        .sr-settings-dropdown .sr-profile-signout:hover { background: #713c48; }
        .sr-profile-menu-mobile { width: 100%; max-width: none; display: flex; justify-content: flex-end; padding-top: 10px; }
        .sr-profile-menu-mobile .sr-profile-trigger { min-height: 48px; }
        .sr-profile-menu-mobile .sr-settings-dropdown-mobile { position: static; width: 100%; max-width: none; margin-top: 8px; box-shadow: none; }

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
