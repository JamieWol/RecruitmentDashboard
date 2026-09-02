import React, { useMemo, useState } from "react";

const emptyReport = { type: "Long Report", foot: "", match: "", date: "", viewing: "", inPossession: "", outPossession: "", physical: "", behaviour: "", strengths: "", weaknesses: "", conclusion: "", performance: "", potential: "", reasons: "" };

export default function ScoutingReportsPage() {
  const [assignments, setAssignments] = useState(() => JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"));
  const [active, setActive] = useState(null);
  const [showCreate, setShowCreate] = useState(false);
  const [draft, setDraft] = useState({ player: "", club: "", position: "", scout: "", game: "", date: "", viewing: "Live" });
  const [report, setReport] = useState(emptyReport);
  const [query, setQuery] = useState("");

  const visible = useMemo(() => assignments.filter(a => `${a.player} ${a.club}`.toLowerCase().includes(query.toLowerCase())), [assignments, query]);
  const update = (key, value) => setReport(r => ({ ...r, [key]: value }));
  const saveAssignments = next => { setAssignments(next); localStorage.setItem("scoutingAssignments", JSON.stringify(next)); };
  const createAssignment = e => { e.preventDefault(); if (!draft.player.trim()) return; const item = { ...draft, id: Date.now(), status: "Not Started", report: null }; saveAssignments([...assignments, item]); setDraft({ player: "", club: "", position: "", scout: "", game: "", date: "", viewing: "Live" }); setShowCreate(false); };
  const open = item => { setActive(item); setReport(item.report || emptyReport); };
  const saveReport = () => { saveAssignments(assignments.map(a => a.id === active.id ? { ...a, report, status: "Complete" } : a)); setActive(null); };
  const deleteAssignment = id => { if (!window.confirm("Delete this assignment? This cannot be undone.")) return; saveAssignments(assignments.filter(a => a.id !== id)); };
  const field = (label, key, rows = 4) => <label className="sr-field"><span>{label}</span><textarea rows={rows} value={report[key]} onChange={e => update(key, e.target.value)} /></label>;

  return <div className="sr-page">
    <div className="sr-heading"><div><p className="eyebrow">SCOUTING REPORTS</p><h1>My Assignments</h1><p>Live and video observations stored in one place.</p></div><button className="sr-primary" onClick={() => setShowCreate(true)}>+ Create Assignment</button></div>
    <input className="sr-search" placeholder="Search assigned players..." value={query} onChange={e => setQuery(e.target.value)} />
    <div className="sr-grid">{visible.map(a => <div className="sr-card" key={a.id} onClick={() => open(a)} role="button" tabIndex={0} onKeyDown={e => e.key === "Enter" && open(a)}><div className="sr-card-top"><strong>{a.player}</strong><span className={`sr-status ${a.status === "Complete" ? "done" : ""}`}>{a.status}</span></div><p>{a.club || "Club not added"} · {a.position || "Position not added"}</p><small>{a.game || "Game not added"} {a.date && `· ${a.date}`} · {a.viewing}</small><small>Assigned to: {a.scout || "Unassigned"}</small><button className="sr-delete" onClick={e => { e.stopPropagation(); deleteAssignment(a.id); }}>Delete Assignment</button></div>)}</div>
    {!visible.length && <div className="sr-empty">No assignments yet. Create an assignment to start a player report.</div>}
    {showCreate && <div className="sr-modal"><form className="sr-form" onSubmit={createAssignment}><h2>Create Assignment</h2><p>Add the game a scout needs to watch.</p>{[ ["Player", "player"], ["Club", "club"], ["Position", "position"], ["Scout", "scout"], ["Game", "game"], ["Date", "date"] ].map(([label,key]) => <label className="sr-field" key={key}><span>{label}</span><input value={draft[key]} onChange={e => setDraft({ ...draft, [key]: e.target.value })} /></label>)}<label className="sr-field"><span>Viewing type</span><select value={draft.viewing} onChange={e => setDraft({ ...draft, viewing: e.target.value })}><option>Live</option><option>Video</option></select></label><div className="sr-actions"><button type="button" onClick={() => setShowCreate(false)}>Cancel</button><button className="sr-primary">Create Assignment</button></div></form></div>}
    {active && <div className="sr-modal"><div className="sr-form sr-report"><div className="sr-heading"><div><p className="eyebrow">{active.game || "SCOUTING REPORT"}</p><h2>{active.player}</h2><p>{active.club} · {active.viewing}</p></div><button type="button" onClick={() => setActive(null)}>Close</button></div><label className="sr-field"><span>Report type</span><select value={report.type} onChange={e => update("type", e.target.value)}><option>Long Report</option><option>Short Report</option></select></label><div className="sr-two"><label className="sr-field"><span>Preferred foot</span><select value={report.foot} onChange={e => update("foot", e.target.value)}><option value="">Select</option><option>Right</option><option>Left</option><option>Both</option></select></label><label className="sr-field"><span>Performance grade</span><select value={report.performance} onChange={e => update("performance", e.target.value)}><option value="">Select 1–5</option>{[5,4,3,2,1].map(x => <option key={x}>{x}</option>)}</select></label></div>{report.type === "Long Report" && <>{field("In Possession", "inPossession")} {field("Out of Possession", "outPossession")} {field("Physical", "physical")} {field("On-Pitch Behaviour", "behaviour")} {field("Strengths", "strengths")} {field("Weaknesses", "weaknesses")}</>} {field("Conclusion", "conclusion")}<div className="sr-two"><label className="sr-field"><span>Potential grade</span><select value={report.potential} onChange={e => update("potential", e.target.value)}><option value="">Select A–F</option>{["A","B","C","D","E","F"].map(x => <option key={x}>{x}</option>)}</select></label></div>{field("Reasons Why", "reasons", 6)}<button className="sr-primary" onClick={saveReport}>Save Report</button></div></div>}
  </div>;
}
