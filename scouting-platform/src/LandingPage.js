import React from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "./AuthContext";
import { ClubBadge } from "./ClubBadge";

const workspaceLinks = [
  {
    title: "Scout a player",
    description: "Create a player report and record your assessment.",
    icon: "◎",
    path: "/scout-report",
    action: "Start a report",
    featured: true,
  },
  {
    title: "Reports & assignments",
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
    title: "Squad plans",
    description: "Build and review position-by-position squad plans.",
    icon: "♧",
    path: "/squad-plans",
    action: "Open squad plans",
  },
  {
    title: "Team analysis",
    description: "Explore player metrics and compare performances.",
    icon: "▥",
    path: "/team-analysis",
    action: "Analyse a team",
  },
];

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
              <span className="sr-home-tool-icon" aria-hidden="true">{item.icon}</span>
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
