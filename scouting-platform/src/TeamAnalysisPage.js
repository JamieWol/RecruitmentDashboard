import React, { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { Bar, BarChart, CartesianGrid, PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer, XAxis, YAxis } from "recharts";

const normalise = (value) => String(value ?? "").trim();
const number = (value) => {
  const parsed = Number(String(value ?? "").replace(/%/g, "").replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
};
const scoreColour = (score) => score >= 75 ? "#15c77a" : score >= 50 ? "#ff9f1a" : score >= 25 ? "#ffd21c" : "#ff4740";
const metricGroup = (metric) => {
  const label = metric.toLowerCase();
  if (/set.?piece|corner|free.?kick|dead.?ball/.test(label)) return "Set-Pieces";
  if (/tackle|intercept|clearance|pressure|regain|defensive|duel|block|conceded|aerial|defend|offside/.test(label)) return "Out Of Possession";
  return "In Possession";
};
const styleGroup = (metric) => {
  const label = String(metric || "").toLowerCase();
  if (/set.?piece|corner|free.?kick|dead.?ball/.test(label)) return "Set-Pieces";
  if (/tackle|intercept|clearance|pressure|regain|defensive|duel|block|conceded|aerial|defend|offside/.test(label)) return "Defensive Work";
  if (/goal|finish|conversion|shot/.test(label)) return "Finishing";
  if (/assist|chance|key pass|scoring|touch|creation/.test(label)) return "Chance Creation";
  if (/xg|expected|carry|dribble|possession|pass/.test(label)) return "Build-up & Possession";
  return "Attacking Output";
};
class ChartBoundary extends React.Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    return this.state.failed ? <div style={{ padding: 24, color: "#667" }}>Chart unavailable for this upload.</div> : this.props.children;
  }
}

export default function TeamAnalysisPage() {
  const [rows, setRows] = useState([]);
  const [selectedTeam, setSelectedTeam] = useState("");
  const [error, setError] = useState("");
  const dashboardRef = useRef(null);

  const teamKey = useMemo(() => {
    const keys = Object.keys(rows[0] || {});
    return keys.find((key) => /^(team|club|squad|side|team name|club name)$/i.test(key)) || keys[0] || "Team";
  }, [rows]);

  const teams = useMemo(() => [...new Set(rows.map((row) => normalise(row[teamKey])).filter(Boolean))], [rows, teamKey]);
  const metrics = useMemo(() => {
    const keys = Object.keys(rows[0] || {});
    return keys.filter((key) => key !== teamKey && !/^(name|player|league|competition|position|country|season|id|rank|games?|games played|matches?|appearances?)$/i.test(key) && rows.filter((row) => number(row[key]) !== null).length >= Math.max(3, Math.floor(rows.length * 0.6)));
  }, [rows, teamKey]);

  const selected = rows.find((row) => normalise(row[teamKey]) === selectedTeam) || rows[0] || null;
  const gamesKey = Object.keys(selected || {}).find((key) => /^(games?|games played|matches?|appearances?)$/i.test(key));
  const chartData = metrics.map((metric) => {
    const values = rows.map((row) => number(row[metric])).filter((value) => value !== null);
    const value = number(selected?.[metric]);
    const average = values.length ? values.reduce((sum, item) => sum + item, 0) / values.length : 0;
    const maximum = values.length ? Math.max(...values) : 0;
    const percentile = value === null || !values.length ? 0 : (values.filter((item) => item <= value).length / values.length) * 100;
    return { metric, percentile: Math.round(percentile), teamValue: value ?? 0, leagueAverage: Number(average.toFixed(2)), leagueAveragePct: maximum ? Math.round((average / maximum) * 100) : 0 };
  });
  const styleData = ["Build-up & Possession", "Chance Creation", "Finishing", "Defensive Work", "Pressing & Regains", "Set-Pieces"].map((style) => {
    const matching = chartData.filter((item) => styleGroup(item.metric) === style || (style === "Pressing & Regains" && /pressure|regain/i.test(item.metric)));
    return { style, score: matching.length ? Math.round(matching.reduce((sum, item) => sum + item.percentile, 0) / matching.length) : 0 };
  });

  const upload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError("");
    const done = (data) => {
      const clean = (data || []).filter((row) => row && typeof row === "object" && Object.values(row).some((value) => normalise(value)));
      if (!clean.length) {
        setRows([]);
        setError("No readable team data was found in that file.");
        return;
      }
      setRows(clean);
      const keys = Object.keys(clean[0] || {});
      const detectedTeamKey = keys.find((key) => /^(team|club|squad|side|team name|club name)$/i.test(key)) || keys[0] || "Team";
      setSelectedTeam(normalise(clean[0]?.[detectedTeamKey]) || "");
    };
    if (/\.xlsx?$/.test(file.name.toLowerCase())) {
      file.arrayBuffer().then((buffer) => {
        const workbook = XLSX.read(buffer, { type: "array" });
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        done(XLSX.utils.sheet_to_json(sheet, { defval: "" }));
      }).catch(() => setError("Could not read that spreadsheet."));
    } else {
      Papa.parse(file, { header: true, skipEmptyLines: true, complete: (result) => done(result.data), error: () => setError("Could not read that CSV file.") });
    }
  };

  const exportPDF = async () => {
    if (!dashboardRef.current || !rows.length) return;
    const canvas = await html2canvas(dashboardRef.current, {
      scale: 2,
      backgroundColor: "#062c63",
      useCORS: true,
      logging: false,
    });
    const pdf = new jsPDF("landscape", "mm", "a4");
    const margin = 8;
    const pageWidth = pdf.internal.pageSize.getWidth() - margin * 2;
    const pageHeight = pdf.internal.pageSize.getHeight() - margin * 2;
    pdf.setFillColor(255, 255, 255);
    pdf.rect(0, 0, pdf.internal.pageSize.getWidth(), pdf.internal.pageSize.getHeight(), "F");
    const ratio = Math.min(pageWidth / canvas.width, pageHeight / canvas.height);
    const width = canvas.width * ratio;
    const height = canvas.height * ratio;
    pdf.addImage(canvas.toDataURL("image/png"), "PNG", margin + (pageWidth - width) / 2, margin + (pageHeight - height) / 2, width, height);
    pdf.save(`${normalise(selected?.[teamKey]) || "team"}-analysis-report.pdf`);
  };

  return (
    <main style={{ minHeight: "calc(100vh - 80px)", background: "linear-gradient(135deg,#062c63 0%,#063d74 100%)", color: "#fff", padding: "42px clamp(22px,5vw,72px) 70px", boxSizing: "border-box" }}>
      <div style={{ maxWidth: 1400, margin: "0 auto" }}>
        <div style={{ color: "#6bd7fa", fontSize: 14, letterSpacing: 1 }}>SCOUTPRO PLATFORM</div>
        <h1 style={{ margin: "10px 0", fontSize: 48, fontWeight: 800, color: "#62dcff", textTransform: "uppercase", fontFamily: 'Impact,"Arial Narrow",sans-serif' }}>Team Analysis</h1>
        <p style={{ color: "#d7e8f8", fontSize: 18 }}>Upload league team data to compare performance across every available playing metric.</p>
        <section style={{ marginTop: 28, padding: 24, borderRadius: 14, background: "rgba(255,255,255,.12)", border: "1px solid rgba(255,255,255,.35)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 18, flexWrap: "wrap" }}><label style={{ fontWeight: 700 }}>Upload team data</label><button type="button" onClick={exportPDF} disabled={!rows.length} style={{ border: "1px solid #b9eaff", borderRadius: 7, padding: "11px 18px", fontWeight: 700, cursor: rows.length ? "pointer" : "not-allowed", background: rows.length ? "#62dcff" : "#47718e", color: "#063d63" }}>Export one-page PDF</button></div>
          <input type="file" accept=".csv,.xlsx,.xls" onChange={upload} style={{ color: "#fff" }} />
          {error && <p style={{ color: "#ffd0d0" }}>{error}</p>}
        </section>
        {!rows.length ? <div style={{ padding: "90px 20px", textAlign: "center", color: "#d7e8f8", fontSize: 20 }}>Upload a league team spreadsheet to build the dashboard.</div> : (
          <>
            <section style={{ marginTop: 28, display: "flex", gap: 16, alignItems: "end", flexWrap: "wrap" }}>
              <label style={{ display: "grid", gap: 8, fontWeight: 700 }}>Team
                <select value={selectedTeam} onChange={(event) => setSelectedTeam(event.target.value)} style={{ minWidth: 280, padding: 12, borderRadius: 8, fontSize: 16 }}>
                  {teams.map((team) => <option key={team}>{team}</option>)}
                </select>
              </label>
              <div style={{ color: "#d7e8f8", paddingBottom: 12 }}>{teams.length} teams · {metrics.length} detected metrics</div>
            </section>
            <div ref={dashboardRef} style={{ background: "#062c63", padding: 18, borderRadius: 16, width: "100%", boxSizing: "border-box", display: "flex", flexDirection: "column" }}>
            <section style={{ marginTop: 28, padding: "22px 28px", borderRadius: 14, background: "linear-gradient(110deg,#0b3c73,#155b91)", border: "2px solid #78b4d8", textAlign: "center" }}><h2 style={{ margin: "0 0 7px", color: "#fff", fontSize: 32 }}><span>{normalise(selected?.[teamKey])}</span><span style={{ marginLeft: "0.3em" }}>Team Analysis</span></h2><div style={{ color: "#d2e5fa", fontSize: 14 }}>{gamesKey ? `Games Played: ${normalise(selected?.[gamesKey]) || "-"}` : ""} · {metrics.length} metrics available</div></section>
            <section style={{ order: 2, marginTop: 24, background: "#fff", color: "#123", borderRadius: 14, padding: "24px 28px", border: "2px solid #2080bd" }}><div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap", marginBottom: 22 }}><h2 style={{ color: "#1680bd", margin: 0 }}><span>Metric</span><span style={{ marginLeft: "0.3em" }}>Percentiles</span></h2><div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: 12, fontWeight: 700 }}><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#15c77a", marginRight: 6, verticalAlign: "-2px" }} />Top 25%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ff9f1a", marginRight: 6, verticalAlign: "-2px" }} />50%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ff4740", marginRight: 6, verticalAlign: "-2px" }} />Below 25%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ffd21c", marginRight: 6, verticalAlign: "-2px" }} />League Average</span></div></div>{chartData.map((item) => <div key={item.metric} style={{ display: "grid", gridTemplateColumns: "220px minmax(0,1fr)", alignItems: "center", gap: 18, marginBottom: 16 }}><div><strong style={{ display: "block", fontSize: 15, color: "#111" }}>{item.metric}</strong><small style={{ color: "#667" }}>{item.teamValue}</small></div><div><div style={{ display: "flex", justifyContent: "flex-end", fontSize: 13, fontWeight: 800, color: "#111", marginBottom: 4 }}><span>{item.percentile}%</span></div><div style={{ height: 13, borderRadius: 8, background: "#e2e4e7", overflow: "hidden" }}><div style={{ width: `${item.percentile}%`, height: "100%", borderRadius: 8, background: scoreColour(item.percentile) }} /></div><div style={{ height: 13, marginTop: 5, borderRadius: 8, background: "#f0f1f2", overflow: "hidden" }}><div style={{ width: `${item.leagueAveragePct}%`, height: "100%", borderRadius: 8, background: "#ffd21c" }} /><span style={{ position: "relative", display: "block", marginTop: -13, textAlign: "center", fontSize: 10, fontWeight: 800, color: "#111" }}>{item.leagueAveragePct}</span></div></div></div>)}</section>
            <section style={{ order: 1, marginTop: 28, background: "#fff", color: "#123", border: "2px solid #2080bd", borderRadius: 14, padding: "24px 28px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(180px,1fr))", gap: 30, maxWidth: 760, margin: "0 auto" }}>
                {["In Possession", "Out Of Possession", "Set-Pieces"].map((group) => {
                  const groupMetrics = chartData.filter((item) => metricGroup(item.metric) === group);
                  const score = groupMetrics.length ? Math.round(groupMetrics.reduce((sum, item) => sum + item.percentile, 0) / groupMetrics.length) : 0;
                  return <div key={group} style={{ textAlign: "center" }}><svg viewBox="0 0 140 140" width="150" height="150" role="img" aria-label={`${group}: ${score}%`}><circle cx="70" cy="70" r="56" fill="none" stroke="#e5e9ef" strokeWidth="12" /><circle cx="70" cy="70" r="56" fill="none" stroke={scoreColour(score)} strokeWidth="12" strokeLinecap="round" strokeDasharray={`${score * 3.518} 351.8`} transform="rotate(-90 70 70)" /><text x="70" y="78" textAnchor="middle" fontSize="27" fontWeight="800" fill="#123">{score}%</text></svg><div style={{ fontWeight: 800, fontSize: 17 }}>{group}</div><div style={{ color: "#667", fontSize: 12, marginTop: 5 }}>{groupMetrics.length} metrics combined</div></div>;
                })}
              </div>
            </section>
            <ChartBoundary><section style={{ order: 3, marginTop: 24, background: "#fff", color: "#123", border: "2px solid #2080bd", borderRadius: 14, padding: "20px 28px" }}><div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: 28 }}><div><h2 style={{ color: "#1680bd", margin: "0 0 4px" }}><span>Team</span><span style={{ marginLeft: "0.3em" }}>Style</span></h2><p style={{ color: "#667", marginTop: 0 }}>Profile based on the uploaded metric percentiles.</p><ResponsiveContainer width="100%" height={300}><RadarChart data={styleData} cx="50%" cy="50%" outerRadius="68%"><PolarGrid /><PolarAngleAxis dataKey="style" tick={{ fontSize: 12, fill: "#123" }} /><Radar name="Team style" dataKey="score" stroke="#1680bd" fill="#1680bd" fillOpacity={0.45} /></RadarChart></ResponsiveContainer></div><div><h2 style={{ color: "#1680bd", margin: "0 0 4px" }}><span>Style</span><span style={{ marginLeft: "0.3em" }}>Areas</span></h2><p style={{ color: "#667", marginTop: 0 }}>Overall scores by playing area.</p><ResponsiveContainer width="100%" height={300}><BarChart data={styleData} layout="vertical" margin={{ left: 20, right: 20 }}><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" domain={[0, 100]} hide /><YAxis type="category" dataKey="style" width={145} tick={{ fontSize: 11, fill: "#123" }} /><Bar dataKey="score" fill="#1680bd" radius={[0, 5, 5, 0]} /></BarChart></ResponsiveContainer></div></div></section></ChartBoundary>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
