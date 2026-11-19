import React from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "./AuthContext";
import { ClubBadge } from "./ClubBadge";

const workspaceLinks = [
  {
    title: "Scout a Player",
    description: "Create a player report and record your assessment.",
    icon: "search",
    path: "/scout-report",
    action: "Start a report",
    featured: true,
  },
  {
    title: "Reports & Assignments",
    description: "Review scouting reports and manage your team's assignments.",
    icon: "▤",
    path: "/scouting-reports",
    action: "Open reports",
  },
  {
    title: "Shortlists",
    description: "Organise target players into lists and folders.",
    icon: "☆",
    path: "/shortlists",
    action: "View shortlists",
  },
  {
    title: "Squad Plans",
    description: "Build and review position-by-position squad plans.",
    icon: "jersey",
    path: "/squad-plans",
    action: "Open squad plans",
  },
  {
    title: "Team Analysis",
    description: "Explore player metrics and compare performances.",
    icon: "chart",
    path: "/team-analysis",
    action: "Analyse a team",
  },
];

function ToolIcon({ type }) {
  if (type === "search") {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8" /><path d="m16 16 5 5" /></svg>;
  }
  if (type === "chart") {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20V11h4v9M10 20V4h4v16M16 20v-7h4v7M2 20h20" /></svg>;
  }
  if (type === "jersey") {
    return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 3-5 3 2 5 3-1v11h8V10l3 1 2-5-5-3c-.7 1.5-2 2.2-4 2.2S8.7 4.5 8 3Z" /><path d="M9 3.5c.5 1.2 1.5 1.8 3 1.8s2.5-.6 3-1.8" /></svg>;
  }
  return <span aria-hidden="true">{type}</span>;
}

function LandingPage() {
  const navigate = useNavigate();
  const { profile } = useAuth();
  const clubName = profile?.club?.trim();
  const userName = profile?.full_name?.trim();

  return (
    <main className="sr-home">
      <section className="sr-home-hero" aria-labelledby="home-title">
        <span className="sr-home-eyebrow">SCOUTPRO WORKSPACE</span>
        <h1 id="home-title" className="sr-home-club-title">
          {clubName && <ClubBadge club={clubName} size={68} />}
          <span>{clubName || "Your scouting workspace"}</span>
        </h1>
        <p>{userName ? `Welcome back, ${userName}` : "Welcome back"}</p>
      </section>

      <section className="sr-home-tools" aria-labelledby="home-tools-title">
        <h2 id="home-tools-title" className="sr-home-visually-hidden">Scouting tools</h2>
        <div className="sr-home-tools-grid">
          {workspaceLinks.map((item) => (
            <button
              className={`sr-home-tool${item.featured ? " featured" : ""}`}
              key={item.path}
              onClick={() => navigate(item.path)}
              type="button"
            >
              <span className={`sr-home-tool-icon${["search", "chart", "jersey"].includes(item.icon) ? " graphic" : ""}`}><ToolIcon type={item.icon} /></span>
              <span className="sr-home-tool-title">{item.title}</span>
              <span className="sr-home-tool-description">{item.description}</span>
              <span className="sr-home-tool-action">{item.action}<span aria-hidden="true"> →</span></span>
            </button>
          ))}
        </div>
      </section>
    </main>
  );
}

export default LandingPage;
