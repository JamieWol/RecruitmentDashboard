import React from "react";
import { useNavigate } from "react-router-dom";

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

  return (
    <main className="sr-home">
      <section className="sr-home-hero" aria-labelledby="home-title">
        <span className="sr-home-eyebrow">SCOUTPRO WORKSPACE</span>
        <h1 id="home-title">Your scouting work, all in one place.</h1>
        <p>
          Find and assess players, share reports, organise targets and plan your squad.
        </p>
      </section>

      <section className="sr-home-tools" aria-labelledby="home-tools-title">
        <div className="sr-home-section-heading">
          <div>
            <h2 id="home-tools-title">What would you like to do?</h2>
            <p>Pick up where your scouting work needs you.</p>
          </div>
        </div>
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
