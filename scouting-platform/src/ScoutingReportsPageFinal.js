import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "./supabaseClient";
const empty = {
  type: "Long Report",
  foot: "",
  performance: "",
  potential: "",
  conclusion: "",
  reasons: "",
  inPossession: "",
  outPossession: "",
  physical: "",
  behaviour: "",
  strengths: "",
  weaknesses: "",
};
const playerPhoto = (name) =>
  `https://syjsmvvsvvprxibqoizw.supabase.co/storage/v1/object/public/player-photos/player-photos/${String(
    name || "",
  )
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((x) =>
      x
        .normalize("NFD")
        .replace(/[̀-ͯ]/g, "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_"),
    )
    .join("_")}.png`;
export default function ScoutingReportsPageFinal() {
  const nav = useNavigate();
  const [items, setItems] = useState(() =>
    JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"),
  );
  const [tab, setTab] = useState("My Assignments");
  const [query, setQuery] = useState("");
  const [profile, setProfile] = useState(() => {
    const n = JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"),
      name = localStorage.getItem("scoutingProfilePlayer");
    return name ? n.find((x) => x.player === name) || { player: name } : null;
  });
  const [playerData, setPlayerData] = useState(null);
  useEffect(() => {
    if (!profile?.player) {
      setPlayerData(null);
      return;
    }
    supabase
      .from("players")
      .select("*")
      .ilike("Name", profile.player)
      .limit(5)
      .then(({ data }) => {
        const exact = (data || []).find(
          (row) => String(row.Name || row.name || "").toLowerCase() === profile.player.toLowerCase(),
        );
        setPlayerData(exact || data?.[0] || null);
      })
      .catch(() => setPlayerData(null));
  }, [profile]);
  const [active, setActive] = useState(null);
  const [report, setReport] = useState(empty);
  const saveItems = (n) => {
    setItems(n);
    localStorage.setItem("scoutingAssignments", JSON.stringify(n));
  };
  const shown = useMemo(
    () =>
      items
        .filter((x) =>
          `${x.player} ${x.club} ${x.scout}`
            .toLowerCase()
            .includes(query.toLowerCase()),
        )
        .filter((x) =>
          tab === "Published"
            ? x.status === "Published"
            : tab === "My Assignments"
              ? x.status !== "Published"
              : true,
        ),
    [items, query, tab],
  );
  const openReport = (x) => {
    setActive(x);
    setReport(x.report || empty);
  };
  const saveReport = () => {
    const n = items.map((x) =>
      x.id === active.id ? { ...x, report, status: "Published" } : x,
    );
    saveItems(n);
    setActive(null);
    setProfile(n.find((x) => x.id === active.id) || profile);
  };
  const field = (l, k, r = 5) => (
    <label className="sr-field">
      <span>{l}</span>
      <textarea
        rows={r}
        value={report[k]}
        onChange={(e) => setReport({ ...report, [k]: e.target.value })}
      />
    </label>
  );
  const dataValue = (...keys) => {
    const source = playerData || profile || {};
    const key = keys.find((k) => source[k] !== undefined && source[k] !== null && source[k] !== "");
    return key ? source[key] : "—";
  };
  const reportPage = active && (
    <div className="sr-modal">
      <section className="sr-form sr-report">
        <div className="sr-form-head">
          <div>
            <div className="sr-kicker">PLAYER REPORT</div>
            <h2>{active.player}</h2>
            <p>
              {active.game} · {active.viewing} · {active.date}
            </p>
          </div>
          <button className="sr-close" onClick={() => setActive(null)}>
            ×
          </button>
        </div>
        <div className="sr-form-grid">
          <label className="sr-field">
            <span>Report type</span>
            <select
              value={report.type}
              onChange={(e) => setReport({ ...report, type: e.target.value })}
            >
              <option>Long Report</option>
              <option>Short Report</option>
            </select>
          </label>
          <label className="sr-field">
            <span>Preferred foot</span>
            <select
              value={report.foot}
              onChange={(e) => setReport({ ...report, foot: e.target.value })}
            >
              <option value="">Select</option>
              <option>Right</option>
              <option>Left</option>
              <option>Both</option>
            </select>
          </label>
        </div>
        {report.type === "Long Report" && (
          <div className="sr-report-columns">
            <div>
              {field("In Possession", "inPossession")}
              {field("Out of Possession", "outPossession")}
              {field("Physical", "physical")}
              {field("On-Pitch Behaviour", "behaviour")}
            </div>
            <div>
              {field("Strengths", "strengths")}
              {field("Weaknesses", "weaknesses")}
            </div>
          </div>
        )}
        {field("Conclusion", "conclusion", 4)}
        <div className="sr-grade-row">
          <label className="sr-field">
            <span>Performance Grade (1–5)</span>
            <select
              value={report.performance}
              onChange={(e) =>
                setReport({ ...report, performance: e.target.value })
              }
            >
              <option value="">Select</option>
              {[5, 4, 3, 2, 1].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
          <label className="sr-field">
            <span>Potential Grade (A–F)</span>
            <select
              value={report.potential}
              onChange={(e) =>
                setReport({ ...report, potential: e.target.value })
              }
            >
              <option value="">Select</option>
              {["A", "B", "C", "D", "E", "F"].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
        </div>
        {field("Reasons Why", "reasons", 6)}
        <button className="sr-cyan sr-save" onClick={saveReport}>
          Publish Report
        </button>
      </section>
    </div>
  );
  if (profile)
    return (
      <main className="sr-page sr-profile-page">
        <button
          className="sr-back"
          onClick={() => {
            if (
              localStorage.getItem("scoutingProfileOrigin") === "shortlists"
            ) {
              localStorage.removeItem("scoutingProfilePlayer");
              localStorage.removeItem("scoutingProfileOrigin");
              nav("/shortlists");
            } else {
              localStorage.removeItem("scoutingProfilePlayer");
              localStorage.removeItem("scoutingProfileOrigin");
              setProfile(null);
            }
          }}
        >
          ‹{" "}
          {localStorage.getItem("scoutingProfileOrigin") === "shortlists"
            ? "Back to previous page"
            : "Back to assignments"}
        </button>
        <section className="sr-profile-head">
          <div>
            <div className="sr-kicker">PLAYER PROFILE</div>
            <h1>{profile.player}</h1>
            <p>
              {profile.club || "Club not added"} ·{" "}
              {profile.position || "Position not added"}
            </p>
          </div>
          <button className="sr-cyan" onClick={() => openReport(profile)}>
            Open Report
          </button>
        </section>
        <section className="sr-profile-summary">
          <div className="sr-avatar">
            <img
              src={playerPhoto(profile.player)}
              alt=""
              onLoad={(e) => {
                e.currentTarget.nextElementSibling.style.display = "none";
              }}
              onError={(e) => {
                e.currentTarget.style.display = "none";
              }}
            />
            <span>{profile.player.slice(0, 2).toUpperCase()}</span>
          </div>
          <div>
            <h2>{profile.player}</h2>
            <p>{profile.club || "Club not added"}</p>
            <span>
              {
                items.filter(
                  (x) =>
                    x.player === profile.player && x.status === "Published",
                ).length
              }{" "}
              published reports
            </span>
          </div>
          <button className="sr-outline">Add to Shortlist</button>
        </section>
        <section className="sr-profile-layout">
          <section className="sr-player-details">
            <h2>Player Information</h2>
            <div className="sr-detail-list">
              {[
                ["Name", ["Name", "name"]],
                ["DOB", ["DOB", "Date of Birth", "date_of_birth"]],
                ["Age", ["Age", "age"]],
                ["Place of birth", ["Place of Birth", "place_of_birth"]],
                ["Nationality", ["Nationality", "nationality"]],
                ["Dominant Foot", ["Dominant Foot", "Preferred Foot", "preferred_foot"]],
                ["Contract Expiry", ["Contract Expiry", "contract_expiry"]],
                ["Height", ["Height", "height"]],
                ["Position", ["Position", "position"]],
                ["Agent", ["Agent", "agent"]],
              ].map(([label, keys]) => (
                <div className="sr-detail-row" key={label}>
                  <strong>{label}:</strong>
                  <span>{dataValue(...keys)}</span>
                </div>
              ))}
            </div>
          </section>
          <section className="sr-profile-reports">
            <h2>Reports</h2>
            <div className="sr-report-table">
              <div className="sr-table-head">
                <span>Date</span>
                <span>Fixture</span>
                <span>Scout</span>
                <span>Potential</span>
                <span>Performance</span>
                <span>Status</span>
              </div>
              {items
                .filter((x) => x.player === profile.player)
                .map((x) => (
                  <button
                    className="sr-table-row"
                    key={x.id}
                    onClick={() => openReport(x)}
                  >
                    <span>{x.date || "—"}</span>
                    <span>{x.game || "—"}</span>
                    <span>{x.scout || "—"}</span>
                    <span>{x.report?.potential || "—"}</span>
                    <span>{x.report?.performance || "—"}</span>
                    <span>{x.status}</span>
                  </button>
                ))}
            </div>
          </section>
        </section>
        {reportPage}
      </main>
    );
  return (
    <main className="sr-page">
      <section className="sr-dashboard-head">
        <div>
          <div className="sr-kicker">SCOUTING PLATFORM</div>
          <h1>Reports and Assignments</h1>
          <p>Manage player observations and complete your scouting reports.</p>
        </div>
        <div className="sr-head-actions">
          <button className="sr-outline" onClick={() => nav("/shortlists")}>
            View Shortlists
          </button>
          <button className="sr-cyan" onClick={() => nav("/create-assignment")}>
            Create New Assignment
          </button>
        </div>
      </section>
      <input
        className="sr-search"
        placeholder="Search scout or player..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <section className="sr-tabs">
        {["My Assignments", "All Assigned", "Published"].map((x) => (
          <button
            className={tab === x ? "selected" : ""}
            onClick={() => setTab(x)}
            key={x}
          >
            {x}
          </button>
        ))}
      </section>
      <div className="sr-section-line">
        <h2>{tab}</h2>
        <span>{shown.length} assignments</span>
      </div>
      <section className="sr-grid">
        {shown.map((x) => (
          <article
            className="sr-card"
            key={x.id}
            onClick={() => {
              localStorage.setItem("scoutingProfileOrigin", "assignments");
              setProfile(x);
            }}
          >
            <div className="sr-card-top">
              <span className="sr-card-status">{x.status}</span>
              <button
                className="sr-trash"
                onClick={(e) => {
                  e.stopPropagation();
                  saveItems(items.filter((y) => y.id !== x.id));
                }}
              >
                Delete
              </button>
            </div>
            <h3>{x.player}</h3>
            <p>
              {x.club || "Club not added"} ·{" "}
              {x.position || "Position not added"}
            </p>
            <div className="sr-fixture">{x.game || "Game not added"}</div>
            <div className="sr-card-meta">
              <span>{x.date || "Date not added"}</span>
              <span>{x.viewing}</span>
              <span>Scout: {x.scout || "Unassigned"}</span>
            </div>
          </article>
        ))}
      </section>
      {!shown.length && <div className="sr-empty">No assignments found.</div>}
    </main>
  );
}
