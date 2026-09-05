import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "./supabaseClient";
const empty = {
  type: "Long Report",
  foot: "",
  playedPosition: "",
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
const photoBase =
  "https://syjsmvvsvvprxibqoizw.supabase.co/storage/v1/object/public/player-photos/player-photos/";
const photoSlug = (name, lower = false) =>
  String(name || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((x) =>
      x
        .normalize("NFD")
        .replace(/[̀-ͯ]/g, "")
        .replace(/[^a-zA-Z0-9]+/g, "_")
        [lower ? "toLowerCase" : "toString"](),
    )
    .join("_");
const playerPhoto = (name, lower = true) =>
  `${photoBase}${photoSlug(name, lower)}.png`;
const retryPhoto = (e, name) => {
  if (e.currentTarget.dataset.fallback !== "1") {
    e.currentTarget.dataset.fallback = "1";
    e.currentTarget.src = playerPhoto(name, false);
  } else e.currentTarget.style.display = "none";
};
export default function ScoutingReportsPageFinal() {
  const nav = useNavigate();
  const [items, setItems] = useState(() =>
    JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"),
  );
  const [tab, setTab] = useState("My Assignments");
  const [query, setQuery] = useState("");
  const [playerSearch, setPlayerSearch] = useState("");
  const [playerMatches, setPlayerMatches] = useState([]);
  const [profile, setProfile] = useState(() => {
    const n = JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"),
      name = localStorage.getItem("scoutingProfilePlayer");
    return name ? n.find((x) => x.player === name) || { player: name } : null;
  });
  const [active, setActive] = useState(null);
  const [playerData, setPlayerData] = useState(null);
  useEffect(() => {
    const selectedPlayer = profile || active;
    if (!selectedPlayer?.player) {
      setPlayerData(null);
      return;
    }
    supabase
      .from("players")
      .select("*")
      .ilike("Name", selectedPlayer.player)
      .limit(5)
      .then(({ data }) => {
        const exact = (data || []).find(
          (row) =>
            String(row.Name || row.name || "").toLowerCase() ===
            selectedPlayer.player.toLowerCase(),
        );
        setPlayerData(exact || data?.[0] || null);
      })
      .catch(() => setPlayerData(null));
  }, [profile, active]);
  const [report, setReport] = useState(empty);
  const [editing, setEditing] = useState(false);
  const [shortlistPicker, setShortlistPicker] = useState(false);
  const [selectedShortlist, setSelectedShortlist] = useState("");
  const [selectedPosition, setSelectedPosition] = useState("CF-0");
  useEffect(() => {
    if (playerSearch.trim().length < 2) {
      setPlayerMatches([]);
      return;
    }
    supabase
      .from("players")
      .select("*")
      .ilike("Name", playerSearch.trim())
      .limit(8)
      .then(({ data }) => setPlayerMatches(data || []))
      .catch(() => setPlayerMatches([]));
  }, [playerSearch]);
  useEffect(() => {
    if (active) setReport(active.report || { ...empty });
  }, [active]);
  const saveItems = (n) => {
    setItems(n);
    localStorage.setItem("scoutingAssignments", JSON.stringify(n));
  };
  const shortlistPositions = [
    "GK",
    "LB",
    "LCB",
    "CB",
    "RCB",
    "RB",
    "DM",
    "LM",
    "LCM",
    "CM",
    "RCM",
    "RM",
    "LW",
    "AM",
    "RW",
    "CF",
    "ST",
  ];
  const addProfileToShortlist = () => {
    const lists = JSON.parse(
      localStorage.getItem("scoutingShortlists") || "[]",
    );
    const list = lists.find((x) => x.id === selectedShortlist);
    if (!list) return;
    const id = profile.id || profile.playerId || profile.player;
    const player = {
      ...profile,
      id,
      player: profile.player,
      slot: selectedPosition,
    };
    const next = {
      ...list,
      players: [
        ...(list.players || []).filter(
          (x) => !(x.id === id && x.slot === selectedPosition),
        ),
        player,
      ],
    };
    localStorage.setItem(
      "scoutingShortlists",
      JSON.stringify(lists.map((x) => (x.id === list.id ? next : x))),
    );
    setShortlistPicker(false);
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
    setEditing(x.status !== "Published");
  };
  const searchPlayer = () => {
    const match =
      items.find(
        (x) => x.player?.toLowerCase() === query.trim().toLowerCase(),
      ) ||
      items.find((x) =>
        x.player?.toLowerCase().includes(query.trim().toLowerCase()),
      );
    if (match) {
      localStorage.setItem("scoutingProfileOrigin", "assignments");
      setActive(null);
      setProfile(match);
      setQuery("");
    }
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
    const source = playerData || profile || active || {};
    const key = keys.find(
      (k) => source[k] !== undefined && source[k] !== null && source[k] !== "",
    );
    return key ? source[key] : "—";
  };
  const reportFoot = [...items]
    .reverse()
    .find((x) => x.player === profile?.player && x.report?.foot)?.report?.foot;
  const grades = (
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
          onChange={(e) => setReport({ ...report, potential: e.target.value })}
        >
          <option value="">Select</option>
          {["A", "B", "C", "D", "E", "F"].map((x) => (
            <option key={x}>{x}</option>
          ))}
        </select>
      </label>
    </div>
  );
  const reportPage = active && (
    <div className="sr-modal">
      <section
        className={`sr-form sr-report ${active?.status === "Published" && !editing ? "readonly" : ""}`}
      >
        <button
          className="sr-report-back"
          onClick={() => {
            setActive(null);
            setProfile(null);
          }}
        >
          ‹ Back to Assignments
        </button>
        <div className="sr-form-head sr-report-banner">
          <div className="sr-report-banner-main">
            <img
              className="sr-report-banner-photo"
              src={playerPhoto(active.player)}
              alt=""
              onError={(e) => retryPhoto(e, active.player)}
            />
            <div>
              <div className="sr-kicker">PLAYER REPORT</div>
              <button
                className="sr-report-player-link"
                onClick={() => {
                  setActive(null);
                  setProfile(active);
                }}
              >
                <span>{active.player}</span>
              </button>
              <p>{dataValue("club", "Club", "team", "Team")}</p>
              <p className="sr-report-player-meta">
                Primary:{" "}
                {dataValue(
                  "Playing Position",
                  "Primary Position",
                  "Position",
                  "playing_position",
                  "primary_position",
                  "position",
                )}{" "}
                · Secondary:{" "}
                {dataValue(
                  "Secondary Position",
                  "secondary_position",
                  "Secondary position",
                )}{" "}
                · DOB: {dataValue("DOB", "Date of Birth", "date_of_birth")}
              </p>
            </div>
          </div>
          {(!active || active.status !== "Published" || editing) && (
            <button className="sr-cyan sr-banner-publish" onClick={saveReport}>
              Publish Report
            </button>
          )}
        </div>
        <div className="sr-fixture-box">
          <strong>Assigned Fixture</strong>
          <span>{active.game || "Fixture not added"}</span>
          <small>
            {active.fixtureDates?.[0] || active.date || "Date not added"} ·{" "}
            {active.viewing || "Viewing not added"}
          </small>
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
            <span>Footage</span>
            <select
              value={report.footage || "Full game"}
              onChange={(e) =>
                setReport({ ...report, footage: e.target.value })
              }
            >
              <option>Full game</option>
              <option>Edited footage</option>
            </select>
          </label>
          <label className="sr-field">
            <span>Viewing</span>
            <select
              value={report.viewing || active.viewing || "Video"}
              onChange={(e) =>
                setReport({ ...report, viewing: e.target.value })
              }
            >
              <option>Live</option>
              <option>Video</option>
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
          <label className="sr-field">
            <span>Position Played</span>
            <select
              value={report.playedPosition || active.position || ""}
              onChange={(e) =>
                setReport({ ...report, playedPosition: e.target.value })
              }
            >
              <option value="">Select position</option>
              {[
                "GK",
                "LB",
                "LCB",
                "CB",
                "RCB",
                "RB",
                "DM",
                "LM",
                "LCM",
                "CM",
                "RCM",
                "RM",
                "LW",
                "AM",
                "RW",
                "CF",
                "ST",
              ].map((x) => (
                <option key={x}>{x}</option>
              ))}
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
              {field("Conclusion", "conclusion", 4)}
            </div>
          </div>
        )}
        {report.type === "Short Report" && field("Conclusion", "conclusion", 4)}
        {grades}
        {field("Reasons Why", "reasons", 6)}
        {active?.status === "Published" && !editing && (
          <div className="sr-report-bottom-actions">
            <button className="sr-outline" onClick={() => setEditing(true)}>
              Edit Report
            </button>
          </div>
        )}
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
          <button className="sr-cyan" onClick={() => nav("/create-assignment")}>
            Create Assignment
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
              onError={(e) => retryPhoto(e, profile.player)}
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
          <button
            className="sr-outline"
            onClick={() => {
              const lists = JSON.parse(
                localStorage.getItem("scoutingShortlists") || "[]",
              );
              setSelectedShortlist(lists[0]?.id || "");
              setShortlistPicker(true);
            }}
          >
            Add to Shortlist
          </button>
        </section>
        <section className="sr-profile-layout">
          <section className="sr-player-details">
            <h2>Player Information</h2>
            <div className="sr-detail-list">
              {[
                ["Name", ["Name", "name"]],
                ["DOB", ["DOB", "Date of Birth", "date_of_birth"]],
                ["Age", ["Age", "age"]],
                ["Nationality", ["Nationality", "nationality"]],
                [
                  "Dominant Foot",
                  ["Dominant Foot", "Preferred Foot", "preferred_foot"],
                ],
                ["Contract Expiry", ["Contract Expiry", "contract_expiry"]],
                ["Height", ["Height", "height"]],
                [
                  "Position",
                  [
                    "Playing Position",
                    "Primary Position",
                    "Position",
                    "playing_position",
                    "primary_position",
                    "position",
                  ],
                ],
                ["Agent", ["Agent", "agent"]],
              ].map(([label, keys]) => (
                <div className="sr-detail-row" key={label}>
                  <strong>{label}:</strong>
                  <span>
                    {label === "Dominant Foot"
                      ? reportFoot || dataValue(...keys)
                      : dataValue(...keys)}
                  </span>
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
        {shortlistPicker && (
          <div className="sr-modal" onClick={() => setShortlistPicker(false)}>
            <section
              className="sr-form sr-shortlist-picker"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="sr-form-head">
                <h2>Add to Shortlist</h2>
                <button
                  className="sr-close"
                  onClick={() => setShortlistPicker(false)}
                >
                  ×
                </button>
              </div>
              <label className="sr-field">
                <span>Shortlist</span>
                <select
                  value={selectedShortlist}
                  onChange={(e) => setSelectedShortlist(e.target.value)}
                >
                  <option value="">Select shortlist</option>
                  {JSON.parse(
                    localStorage.getItem("scoutingShortlists") || "[]",
                  ).map((x) => (
                    <option key={x.id} value={x.id}>
                      {x.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="sr-field">
                <span>Position</span>
                <select
                  value={selectedPosition}
                  onChange={(e) => setSelectedPosition(e.target.value)}
                >
                  {shortlistPositions.map((x, i) => (
                    <option key={`${x}-${i}`} value={`${x}-${i}`}>
                      {x}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="sr-cyan"
                disabled={!selectedShortlist}
                onClick={addProfileToShortlist}
              >
                Add Player
              </button>
            </section>
          </div>
        )}
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
          <div className="sr-player-search-wrap">
            <input
              className="sr-player-search"
              placeholder="Search player..."
              value={playerSearch}
              onChange={(e) => setPlayerSearch(e.target.value)}
            />
            {!!playerMatches.length && (
              <div className="sr-player-search-results">
                {playerMatches.map((p) => {
                  const name = p.Name || p.name;
                  return (
                    <button
                      key={p.id || name}
                      onClick={() => {
                        setProfile({
                          ...p,
                          player: name,
                          club: p.club || p.Club || p.team || p.Team,
                        });
                        setPlayerSearch("");
                        setPlayerMatches([]);
                        localStorage.setItem(
                          "scoutingProfileOrigin",
                          "assignments",
                        );
                      }}
                    >
                      {name}
                      <small>
                        {p.club ||
                          p.Club ||
                          p.team ||
                          p.Team ||
                          "Club not added"}
                      </small>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
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
        onKeyDown={(e) => {
          if (e.key === "Enter") searchPlayer();
        }}
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
              setProfile(null);
              setActive(x);
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
            <button
              type="button"
              className="sr-assignment-player-link"
              onClick={(e) => {
                e.stopPropagation();
                setProfile(null);
                setActive(x);
              }}
            >
              {x.player}
            </button>
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
      {reportPage}
    </main>
  );
}
