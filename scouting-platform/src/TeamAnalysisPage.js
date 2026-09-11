import React, { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";

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

  const upload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError("");
    const done = (data) => {
      const clean = (data || []).filter((row) => Object.values(row || {}).some((value) => normalise(value)));
      setRows(clean);
      setSelectedTeam(normalise(clean[0]?.[Object.keys(clean[0] || {})[0]]) || "");
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
    pdf.addImage(canvas.toDataURL("image/png"), "PNG", margin, margin, pageWidth, pageHeight);
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
            <div ref={dashboardRef} style={{ background: "#062c63", padding: 18, borderRadius: 16, width: "100%", boxSizing: "border-box" }}>
            <section style={{ marginTop: 28, display: "grid", gridTemplateColumns: "minmax(260px, .8fr) minmax(420px, 1.8fr)", gap: 24 }}>
              <div style={{ background: "#fff", color: "#123", borderRadius: 14, padding: 24, border: "2px solid #2080bd" }}><h2 style={{ marginTop: 0, color: "#000" }}>{normalise(selected?.[teamKey])}</h2><p style={{ color: "#1f77b4", fontWeight: 700 }}>Team information</p><p><strong>Games Played:</strong> {gamesKey ? normalise(selected?.[gamesKey]) || "-" : "-"}</p><p>{metrics.length} metrics available for comparison.</p><p>{rows.length} league records uploaded.</p></div>
              <div style={{ background: "#fff", color: "#123", borderRadius: 14, padding: "24px 28px", border: "2px solid #2080bd" }}><div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 16, flexWrap: "wrap", marginBottom: 22 }}><h2 style={{ color: "#1680bd", margin: 0 }}>Metric Percentiles</h2><div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: 12, fontWeight: 700 }}><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#15c77a", marginRight: 6, verticalAlign: "-2px" }} />Top 25%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ff9f1a", marginRight: 6, verticalAlign: "-2px" }} />50%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ff4740", marginRight: 6, verticalAlign: "-2px" }} />Below 25%</span><span><i style={{ display: "inline-block", width: 13, height: 13, borderRadius: 4, background: "#ffd21c", marginRight: 6, verticalAlign: "-2px" }} />League Average</span></div></div>{chartData.map((item) => <div key={item.metric} style={{ display: "grid", gridTemplateColumns: "220px minmax(0,1fr)", alignItems: "center", gap: 18, marginBottom: 16 }}><div><strong style={{ display: "block", fontSize: 15, color: "#111" }}>{item.metric}</strong><small style={{ color: "#667" }}>{item.teamValue}</small></div><div><div style={{ display: "flex", justifyContent: "flex-end", fontSize: 13, fontWeight: 800, color: "#111", marginBottom: 4 }}><span>{item.percentile}%</span></div><div style={{ height: 13, borderRadius: 8, background: "#e2e4e7", overflow: "hidden" }}><div style={{ width: `${item.percentile}%`, height: "100%", borderRadius: 8, background: scoreColour(item.percentile) }} /></div><div style={{ height: 13, marginTop: 5, borderRadius: 8, background: "#f0f1f2", overflow: "hidden" }}><div style={{ width: `${item.leagueAveragePct}%`, height: "100%", borderRadius: 8, background: "#ffd21c" }} /><span style={{ position: "relative", display: "block", marginTop: -13, textAlign: "center", fontSize: 10, fontWeight: 800, color: "#111" }}>{item.leagueAveragePct}</span></div></div></div>)}</div>
            </section>
            <section style={{ marginTop: 28, background: "#fff", color: "#123", border: "2px solid #2080bd", borderRadius: 14, padding: "24px 28px" }}>
              <h2 style={{ color: "#000", margin: "0 0 22px" }}>Team Scorecard</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(180px,1fr))", gap: 30, maxWidth: 760, margin: "0 auto" }}>
                {["In Possession", "Out Of Possession", "Set-Pieces"].map((group) => {
                  const groupMetrics = chartData.filter((item) => metricGroup(item.metric) === group);
                  const score = groupMetrics.length ? Math.round(groupMetrics.reduce((sum, item) => sum + item.percentile, 0) / groupMetrics.length) : 0;
                  return <div key={group} style={{ textAlign: "center" }}><svg viewBox="0 0 140 140" width="150" height="150" role="img" aria-label={`${group}: ${score}%`}><circle cx="70" cy="70" r="56" fill="none" stroke="#e5e9ef" strokeWidth="12" /><circle cx="70" cy="70" r="56" fill="none" stroke={scoreColour(score)} strokeWidth="12" strokeLinecap="round" strokeDasharray={`${score * 3.518} 351.8`} transform="rotate(-90 70 70)" /><text x="70" y="78" textAnchor="middle" fontSize="27" fontWeight="800" fill="#123">{score}%</text></svg><div style={{ fontWeight: 800, fontSize: 17 }}>{group}</div><div style={{ color: "#667", fontSize: 12, marginTop: 5 }}>{groupMetrics.length} metrics combined</div></div>;
                })}
              </div>
              <div style={{ display: "flex", justifyContent: "center", gap: 22, flexWrap: "wrap", marginTop: 22, fontSize: 13, fontWeight: 700 }}>
                {[['#15c77a','Top 25%'],['#ff9f1a','50%+'],['#ff4740','Below 25%'],['#ffd21c','League Average']].map(([colour, label]) => <span key={label} style={{ display: "flex", alignItems: "center", gap: 7 }}><i style={{ width: 14, height: 14, borderRadius: 4, background: colour, display: "inline-block" }} />{label}</span>)}
              </div>
            </section>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
