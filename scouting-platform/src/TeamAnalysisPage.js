import React, { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer } from "recharts";

const normalise = (value) => String(value ?? "").trim();
const displayName = (value) => normalise(value).replace(/([a-z])([A-Z])/g, "$1 $2");
const number = (value) => {
  const parsed = Number(String(value ?? "").replace(/%/g, "").replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
};
const lowerIsBetter = (metric) => /conceded|faced|against|turnovers lost|errors?/.test(String(metric || "").toLowerCase());
const scoreColour = (score) => score >= 75 ? "#15c77a" : score >= 50 ? "#ff9f1a" : score >= 25 ? "#ffd21c" : "#ff4740";
const metricGroup = (metric) => {
  const label = metric.toLowerCase();
  if (/set.?piece|corner|free.?kick|dead.?ball|throw.?in|dfk/.test(label)) return "Set-Pieces";
  if (/tackle|intercept|clearance|pressure|regain|defensive|duel|block|conceded|faced|against|aerial|defend|offside/.test(label)) return "Out Of Possession";
  return "In Possession";
};
const styleGroup = (metric) => {
  const label = String(metric || "").toLowerCase();
  if (/set.?piece|corner|free.?kick|dead.?ball|throw.?in|dfk/.test(label)) return "Set-Pieces";
  if (/tackle|intercept|clearance|pressure|regain|defensive|duel|block|conceded|faced|against|aerial|defend|offside/.test(label)) return "Defensive Work";
  if (/xg|expected goals|assist|chance|key pass|scoring contribution|touch(es)? in box|creation/.test(label)) return "Chance Creation";
  if (/goal|finish|conversion|shot/.test(label)) return "Finishing";
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
  const [selectedMetrics, setSelectedMetrics] = useState([]);
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

  const activeMetrics = selectedMetrics.length ? metrics.filter((metric) => selectedMetrics.includes(metric)) : metrics;

  const selected = rows.find((row) => normalise(row[teamKey]) === selectedTeam) || rows[0] || null;
  const selectedTeamName = displayName(selected?.[teamKey]);
  const gamesKey = Object.keys(selected || {}).find((key) => /^(games?|games played|matches?|appearances?)$/i.test(key));
  const chartData = activeMetrics.map((metric) => {
    const values = rows.map((row) => number(row[metric])).filter((value) => value !== null);
    const value = number(selected?.[metric]);
    const average = values.length ? values.reduce((sum, item) => sum + item, 0) / values.length : 0;
    const maximum = values.length ? Math.max(...values) : 0;
    const percentile = value === null || !values.length ? 0 : (lowerIsBetter(metric) ? values.filter((item) => item >= value).length : values.filter((item) => item <= value).length) / values.length * 100;
    return { metric, percentile: Math.round(percentile), teamValue: value ?? 0, leagueAverage: Number(average.toFixed(2)), leagueAveragePct: maximum ? Math.round((average / maximum) * 100) : 0 };
  });
  const styleData = ["Build-up & Possession", "Chance Creation", "Finishing", "Defensive Work", "Pressing & Regains", "Set-Pieces"].map((style) => {
    const matching = chartData.filter((item) => styleGroup(item.metric) === style || (style === "Pressing & Regains" && /pressure|regain/i.test(item.metric)));
    return { style, score: matching.length ? Math.round(matching.reduce((sum, item) => sum + item.percentile, 0) / matching.length) : 0, leagueAverage: matching.length ? Math.round(matching.reduce((sum, item) => sum + item.leagueAveragePct, 0) / matching.length) : 0 };
  });
  const metricColumns = ["In Possession", "Out Of Possession", "Set-Pieces"];
  const renderMetric = (item) => <div key={item.metric} style={{ marginBottom: 14, breakInside: "avoid" }}><div style={{ display: "flex", justifyContent: "space-between", alignItems: "end", gap: 10 }}><div><strong style={{ display: "block", fontSize: 14, color: "#fff" }}>{item.metric}</strong><small style={{ color: "#d2e5fa" }}>{item.teamValue}</small></div><strong style={{ color: "#fff", fontSize: 13 }}>{item.percentile}%</strong></div><div style={{ height: 11, marginTop: 4, borderRadius: 8, background: "#365b80", overflow: "hidden" }}><div style={{ width: `${item.percentile}%`, height: "100%", background: scoreColour(item.percentile), borderRadius: 8 }} /></div><div style={{ height: 11, marginTop: 4, borderRadius: 8, background: "#284d73", overflow: "hidden" }}><div style={{ width: `${item.leagueAveragePct}%`, height: "100%", background: "#ffd21c", borderRadius: 8 }} /><span style={{ position: "relative", display: "block", marginTop: -11, textAlign: "center", fontSize: 9, fontWeight: 800, color: "#111" }}>{item.leagueAveragePct}%</span></div></div>;

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
      setSelectedMetrics([]);
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
    const pdf = new jsPDF("portrait", "mm", "a4");
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
      <style>{`[data-old-metric-section]{display:none!important}[data-grouped-metrics]{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;order:3!important}[data-grouped-metrics] span{color:#fff!important}section[style*="order: 3"]{order:2!important}@media(max-width:800px){[data-grouped-metrics]{grid-template-columns:1fr}}`}</style>
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
            <section style={{ marginTop: 28, display: "grid", gap: 16 }}>
              <div style={{ display: "flex", gap: 16, alignItems: "end", flexWrap: "wrap" }}>
              <label style={{ display: "grid", gap: 8, fontWeight: 700 }}>Team
                <select value={selectedTeam} onChange={(event) => setSelectedTeam(event.target.value)} style={{ minWidth: 280, padding: 12, borderRadius: 8, fontSize: 16 }}>
                  {teams.map((team) => <option key={team}>{team}</option>)}
                </select>
              </label>
              <div style={{ color: "#d7e8f8", paddingBottom: 12 }}>{teams.length} teams · {activeMetrics.length} of {metrics.length} metrics selected</div>
              </div>
              <div style={{ padding: 18, borderRadius: 12, background: "rgba(255,255,255,.12)", border: "1px solid rgba(255,255,255,.28)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
                  <strong>Select metrics to include</strong>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button type="button" onClick={() => setSelectedMetrics([])} style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #b9eaff", background: "#62dcff", color: "#063d63", fontWeight: 700 }}>Use all</button>
                    <button type="button" onClick={() => setSelectedMetrics(metrics.filter((metric) => metricGroup(metric) === "In Possession"))} style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid #b9eaff", background: "transparent", color: "#fff", fontWeight: 700 }}>In Possession only</button>
                  </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))", gap: 16 }}>
                  {["In Possession", "Out Of Possession", "Set-Pieces"].map((group) => <div key={group}>
                    <div style={{ color: "#62dcff", fontWeight: 800, marginBottom: 8 }}>{group}</div>
                    <div style={{ display: "grid", gap: 6 }}>{metrics.filter((metric) => metricGroup(metric) === group).map((metric) => <label key={metric} style={{ display: "flex", gap: 7, alignItems: "center", color: "#fff", fontSize: 13 }}><input type="checkbox" checked={!selectedMetrics.length || selectedMetrics.includes(metric)} onChange={() => setSelectedMetrics((current) => { const base = current.length ? current : [...metrics]; return base.includes(metric) ? base.filter((item) => item !== metric) : [...base, metric]; })} />{metric}</label>)}</div>
                  </div>)}
                </div>
              </div>
            </section>
            <div ref={dashboardRef} style={{ background: "#062c63", padding: 18, borderRadius: 16, width: "100%", boxSizing: "border-box", display: "flex", flexDirection: "column" }}>
            <section style={{ marginTop: 28, padding: "22px 28px", borderRadius: 14, background: "linear-gradient(110deg,#0b3c73,#155b91)", border: "2px solid #78b4d8", textAlign: "center" }}><h2 style={{ margin: "0 0 7px", color: "#fff", fontSize: 32 }}><span>{selectedTeamName}</span><span style={{ marginLeft: "0.35em" }}>Team&nbsp;Analysis</span></h2><div style={{ color: "#d2e5fa", fontSize: 14 }}>{gamesKey ? `Games Played: ${normalise(selected?.[gamesKey]) || "-"}` : ""} · {metrics.length} metrics available</div></section>
            <section data-old-metric-section style={{ order: 2, marginTop: 24, background: "#173f70", color: "#fff", borderRadius: 14, padding: "24px 28px", border: "2px solid #6a96bd" }}><div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap", marginBottom: 22 }}><h2 style={{ color: "#62dcff", margin: 0 }}><span>Metric</span><span style={{ marginLeft: "0.3em" }}>Percentiles</span></h2><div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: 12, fontWeight: 700 }}><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#15c77a", marginRight: 6, verticalAlign: "-2px" }} />Top 25%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ff9f1a", marginRight: 6, verticalAlign: "-2px" }} />50%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ff4740", marginRight: 6, verticalAlign: "-2px" }} />Below 25%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ffd21c", marginRight: 6, verticalAlign: "-2px" }} />League Average</span></div></div><div style={{ columnCount: 2, columnGap: 30 }}>{chartData.map((item) => <div key={item.metric} style={{ display: "grid", gridTemplateColumns: "150px minmax(0,1fr)", alignItems: "center", gap: 12, marginBottom: 14, breakInside: "avoid" }}><div><strong style={{ display: "block", fontSize: 14, color: "#fff" }}>{item.metric}</strong><small style={{ color: "#d2e5fa" }}>{item.teamValue}</small></div><div><div style={{ display: "flex", justifyContent: "flex-end", fontSize: 12, fontWeight: 800, color: "#fff", marginBottom: 4 }}><span>{item.percentile}%</span></div><div style={{ height: 11, borderRadius: 8, background: "#365b80", overflow: "hidden" }}><div style={{ width: `${item.percentile}%`, height: "100%", borderRadius: 8, background: scoreColour(item.percentile) }} /></div><div style={{ height: 11, marginTop: 4, borderRadius: 8, background: "#284d73", overflow: "hidden" }}><div style={{ width: `${item.leagueAveragePct}%`, height: "100%", borderRadius: 8, background: "#ffd21c" }} /><span style={{ position: "relative", display: "block", marginTop: -11, textAlign: "center", fontSize: 9, fontWeight: 800, color: "#111" }}>{item.leagueAveragePct}%</span></div></div></div>)}</div></section>
            <section style={{ order: 1, marginTop: 28, background: "#173f70", color: "#fff", border: "2px solid #6a96bd", borderRadius: 14, padding: "24px 28px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(180px,1fr))", gap: 30, maxWidth: 760, margin: "0 auto" }}>
                {["In Possession", "Out Of Possession", "Set-Pieces"].map((group) => {
                  const groupMetrics = chartData.filter((item) => metricGroup(item.metric) === group);
                  const score = groupMetrics.length ? Math.round(groupMetrics.reduce((sum, item) => sum + item.percentile, 0) / groupMetrics.length) : 0;
                  return <div key={group} style={{ textAlign: "center" }}><svg viewBox="0 0 140 140" width="150" height="150" role="img" aria-label={`${group}: ${score}%`}><circle cx="70" cy="70" r="56" fill="none" stroke="#365b80" strokeWidth="12" /><circle cx="70" cy="70" r="56" fill="none" stroke={scoreColour(score)} strokeWidth="12" strokeLinecap="round" strokeDasharray={`${score * 3.518} 351.8`} transform="rotate(-90 70 70)" /><text x="70" y="78" textAnchor="middle" fontSize="27" fontWeight="800" fill="#fff">{score}%</text></svg><div style={{ color: "#fff", fontWeight: 800, fontSize: 17 }}>{group}</div><div style={{ color: "#d2e5fa", fontSize: 12, marginTop: 5 }}>{groupMetrics.length} metrics combined</div></div>;
                })}
              </div>
            </section>
            <section data-grouped-metrics style={{ order: 2.5, marginTop: 24, background: "#173f70", color: "#fff", border: "2px solid #6a96bd", borderRadius: 14, padding: "24px 28px" }}><div style={{ gridColumn: "1 / -1", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap", marginBottom: 8 }}><h2 style={{ color: "#62dcff", margin: 0 }}><span>Metric</span><span style={{ marginLeft: "0.3em" }}>Percentiles</span></h2><div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: 12, fontWeight: 700 }}><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#15c77a", marginRight: 6, verticalAlign: "-2px" }} />Top 25%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ff9f1a", marginRight: 6, verticalAlign: "-2px" }} />50%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ff4740", marginRight: 6, verticalAlign: "-2px" }} />Below 25%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ffd21c", marginRight: 6, verticalAlign: "-2px" }} />League Average</span></div></div>{metricColumns.map((group) => <div key={group}><h3 style={{ color: "#fff", fontSize: 18, margin: "0 0 16px", borderBottom: "1px solid #6a96bd", paddingBottom: 8 }}>{group}</h3>{chartData.filter((item) => metricGroup(item.metric) === group).map(renderMetric)}</div>)}</section>
            <ChartBoundary><section style={{ order: 3, marginTop: 24, background: "#173f70", color: "#fff", border: "2px solid #6a96bd", borderRadius: 14, padding: "20px 28px" }}><h2 style={{ color: "#62dcff", margin: "0 0 4px", textAlign: "center" }}><span>Team</span><span style={{ marginLeft: "0.3em" }}>Style</span></h2><p style={{ color: "#d2e5fa", marginTop: 0, textAlign: "center" }}>Team profile compared with the league average.</p><ResponsiveContainer width="100%" height={300}><RadarChart data={styleData} cx="50%" cy="50%" outerRadius="68%"><PolarGrid stroke="#6a96bd" /><PolarAngleAxis dataKey="style" tick={{ fontSize: 12, fill: "#d2e5fa" }} /><Radar name="League Average" dataKey="leagueAverage" stroke="#ffd21c" fill="#ffd21c" fillOpacity={0.28} /><Radar name={normalise(selected?.[teamKey]) || "Team"} dataKey="score" stroke="#62dcff" fill="#62dcff" fillOpacity={0.5} /></RadarChart></ResponsiveContainer><div style={{ display: "flex", justifyContent: "center", gap: 24, fontSize: 13, fontWeight: 700 }}><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#62dcff", marginRight: 6, verticalAlign: "-2px" }} />{normalise(selected?.[teamKey]) || "Team"}</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ffd21c", marginRight: 6, verticalAlign: "-2px" }} />League Average</span></div></section></ChartBoundary>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
