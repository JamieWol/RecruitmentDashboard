import React, { useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const normalise = (value) => String(value ?? "").trim();
const number = (value) => {
  const parsed = Number(String(value ?? "").replace(/%/g, "").replace(/,/g, ""));
  return Number.isFinite(parsed) ? parsed : null;
};
const scoreColour = (score) => score >= 75 ? "#15c77a" : score >= 50 ? "#ff9f1a" : score >= 25 ? "#ffd21c" : "#ff4740";

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
    const percentile = value === null || !values.length ? 0 : (values.filter((item) => item <= value).length / values.length) * 100;
    return { metric, percentile: Math.round(percentile), teamValue: value ?? 0, leagueAverage: Number(average.toFixed(2)) };
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
    const ratio = Math.min(pageWidth / canvas.width, pageHeight / canvas.height);
    const width = canvas.width * ratio;
    const height = canvas.height * ratio;
    pdf.addImage(canvas.toDataURL("image/png"), "PNG", margin + (pageWidth - width) / 2, margin + (pageHeight - height) / 2, width, height);
    pdf.save(`${normalise(selected?.[teamKey]) || "team"}-analysis-report.pdf`);
  };

  return (
    <main style={{ minHeight: "calc(100vh - 80px)", background: "linear-gradient(135deg,#062c63 0%,#063d74 100%)", color: "#fff", padding: "42px clamp(22px,5vw,72px) 70px", boxSizing: "border-box" }}>
      <div ref={dashboardRef} style={{ maxWidth: 1400, margin: "0 auto" }}>
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
            <section style={{ marginTop: 28, display: "grid", gridTemplateColumns: "minmax(260px, .8fr) minmax(420px, 1.8fr)", gap: 24 }}>
              <div style={{ background: "#fff", color: "#123", borderRadius: 14, padding: 24, border: "2px solid #2080bd" }}><h2 style={{ marginTop: 0, color: "#000" }}>{normalise(selected?.[teamKey])}</h2><p style={{ color: "#1f77b4", fontWeight: 700 }}>Team information</p><p><strong>Games Played:</strong> {gamesKey ? normalise(selected?.[gamesKey]) || "-" : "-"}</p><p>{metrics.length} metrics available for comparison.</p><p>{rows.length} league records uploaded.</p></div>
              <div style={{ background: "#fff", color: "#123", borderRadius: 14, padding: 18, border: "2px solid #2080bd" }}><h2 style={{ color: "#000", margin: "4px 8px 12px" }}>Team Percentiles</h2><ResponsiveContainer width="100%" height={420}><BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 20 }}><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} /><YAxis type="category" dataKey="metric" width={150} tick={{ fontSize: 11 }} /><Tooltip formatter={(value) => `${value}%`} /><Bar dataKey="percentile" fill="#1f77b4" name="Team percentile" /></BarChart></ResponsiveContainer></div>
            </section>
            <section style={{ marginTop: 28, background: "#fff", color: "#123", border: "2px solid #2080bd", borderRadius: 14, padding: "24px 28px" }}>
              <h2 style={{ color: "#000", margin: "0 0 22px" }}>Team Scorecard</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))", gap: 22 }}>
                {chartData.slice().sort((a, b) => b.percentile - a.percentile).slice(0, 8).map((item) => (
                  <div key={item.metric} style={{ textAlign: "center" }}>
                    <svg viewBox="0 0 120 120" width="112" height="112" role="img" aria-label={`${item.metric}: ${item.percentile}%`}>
                      <circle cx="60" cy="60" r="48" fill="none" stroke="#e5e9ef" strokeWidth="10" />
                      <circle cx="60" cy="60" r="48" fill="none" stroke={scoreColour(item.percentile)} strokeWidth="10" strokeLinecap="round" strokeDasharray={`${item.percentile * 3.016} 301.6`} transform="rotate(-90 60 60)" />
                      <text x="60" y="66" textAnchor="middle" fontSize="24" fontWeight="800" fill="#123">{item.percentile}%</text>
                    </svg>
                    <div style={{ fontWeight: 700, fontSize: 13, lineHeight: 1.2 }}>{item.metric}</div>
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", justifyContent: "center", gap: 22, flexWrap: "wrap", marginTop: 22, fontSize: 13, fontWeight: 700 }}>
                {[['#15c77a','Top 25%'],['#ff9f1a','50%+'],['#ff4740','Below 25%'],['#ffd21c','League Average']].map(([colour, label]) => <span key={label} style={{ display: "flex", alignItems: "center", gap: 7 }}><i style={{ width: 14, height: 14, borderRadius: 4, background: colour, display: "inline-block" }} />{label}</span>)}
              </div>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
