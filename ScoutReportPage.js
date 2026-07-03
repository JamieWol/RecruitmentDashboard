import React, { useState, useRef, useEffect } from "react";
import Papa from "papaparse";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import { POSITION_MAP } from "./POSITION_MAP";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  Legend
} from "recharts";

function ScoutReportPage({ shadowSquad, setShadowSquad }) {
  const [players, setPlayers] = useState([]);
  const [isExporting, setIsExporting] = useState(false);
  const exportRef = useRef(null);
  const [clipsLink, setClipsLink] = useState("");
  const [filteredPlayers, setFilteredPlayers] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [scatterMetrics, setScatterMetrics] = useState({ x: "", y: "" });
  const [competitions, setCompetitions] = useState([]);
  const [filters, setFilters] = useState({
    minMinutes: 0,
    maxMinutes: 99999,
    competition: "All",
    positions: []
  });

    const exportPDF = async () => {
      if (!exportRef.current || !selectedPlayer) return;

      const canvas = await html2canvas(exportRef.current, { scale: 2 });
      const imgData = canvas.toDataURL("image/png");

      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, "PNG", 0, 10, pdfWidth, pdfHeight);
      pdf.save(`${selectedPlayer["Player Name"]}_Report.pdf`);
    };

    const getPlayerPhoto = (player) => {
      // Use CSV photo column if available
      if (player?._photoUrl) return player._photoUrl;

      if (!player || !player["Player Name"]) {
        return "/placeholder.png";
      }

      const safeName = player["Player Name"]
        .toLowerCase()
        .trim()
        .replace(/\s+/g, "_")
        .replace(/[^a-z0-9_]/g, "");

      return `/player-photos/${safeName}.png`;
    };






  const excludedColumns = [
    "Account","Default Radar Template","Name","Team","Competition",
    "Competition Type","Competition Rank","Season","Nationality",
    "Country Code","Date of Birth","Age","Woman Player?",
    "Team Color 1","Team Color 2","Player Id","Player Name",
    "First Name","Last Name","Nickname","Weight","Height",
    "Birth Date","Country Id","Country","Team Id",
    "Team Color 1st","Team Color 2nd","Competition Id",
    "Competition Name","Season Id","Seasons",
    "Primary Position","Secondary Position","Most Recent Match",
    "Player SBData Id",
    "90s Played","Appearances","Minutes Played","Starting Appearances"
  ];

  const lowerIsBetterMetrics = ["Turnovers", "Fouls", "Positioning Error"];

  const detectMetrics = (data) =>
    Object.keys(data[0] || {}).filter(
      k =>
        !excludedColumns.includes(k) &&
        data.every(p => !isNaN(Number(p[k])))
    );

  const handleUpload = (e) => {
    Papa.parse(e.target.files[0], {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
          const parsed = res.data.map(p => {
            const obj = {};
            Object.keys(p).forEach(k => {
              obj[k] = isNaN(p[k]) ? p[k] : Number(p[k]);
            });

            obj._photoUrl = p["Photo"]; // ⬅️ EXACT column name from CSV

            return obj;
          });


        const detected = detectMetrics(parsed);

        setPlayers(parsed);
        setFilteredPlayers(parsed);
        setMetrics(detected);
        setScatterMetrics({ x: detected[0] || "", y: detected[1] || "" });

        const comps = Array.from(new Set(parsed.map(p => p["Competition Name"]))).filter(Boolean);
        setCompetitions(["All", ...comps]);
      }
    });
  };

  const applyFilters = () => {
    const f = players.filter(p =>
      p["Minutes Played"] >= filters.minMinutes &&
      p["Minutes Played"] <= filters.maxMinutes &&
      (filters.competition === "All" ? true : p["Competition Name"] === filters.competition) &&
      (filters.positions.length === 0 ? true : filters.positions.includes(p["Primary Position"]))
    );

    setFilteredPlayers(f);
    if (selectedPlayer && !f.includes(selectedPlayer)) {
      setSelectedPlayer(null);
    }
  };
    

    
  const percentilesForPlayer = (player) =>
    metrics.map(m => {
      const values = filteredPlayers.map(p => p[m]);
      let pct = (values.filter(v => v <= player[m]).length / values.length) * 100;
      if (lowerIsBetterMetrics.includes(m)) pct = 100 - pct;

      const avgValue = values.reduce((a,b)=>a+b,0)/values.length;
      let avgPct = (values.filter(v => v <= avgValue).length / values.length) * 100;
      if (lowerIsBetterMetrics.includes(m)) avgPct = 100 - avgPct;
        


      return {
        metric: m,
        value: Math.round(pct),
        raw: Number(player[m]).toFixed(2),
        leagueAvg: Math.round(avgPct)
      };
    });

  const averageRating = (player) => {
    const pcts = percentilesForPlayer(player).map(p => p.value);
    return (pcts.reduce((a, b) => a + b, 0) / pcts.length).toFixed(1);
  };

  const overallRank = (player) => {
    const ranked = [...filteredPlayers]
      .map(p => ({ player: p, rating: Number(averageRating(p)) }))
      .sort((a, b) => b.rating - a.rating);

    return ranked.findIndex(r => r.player === player) + 1;
  };

  const CustomScatterTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const player = payload[0].payload;
      return (
        <div style={{
          background: "#fff",
          padding: 10,
          border: "1px solid #ccc",
          borderRadius: 6,
          boxShadow: "0 2px 6px rgba(0,0,0,0.15)"
        }}>
          <strong style={{ color: "#1f77b4" }}>{player["Player Name"]}</strong>
          <div>{scatterMetrics.x}: {player[scatterMetrics.x]}</div>
          <div>{scatterMetrics.y}: {player[scatterMetrics.y]}</div>
        </div>
      );
    }
    return null;
  };

  const attackingMetrics = metrics.filter(m => /goal|assist|shot|key pass|xg/i.test(m));
  const defendingMetrics = metrics.filter(m => /tackle|intercept|clearance|block|aerial|pressures/i.test(m));
  const ballCarryingMetrics = metrics.filter(m => /dribble|carry|progressive|pass/i.test(m));

  const goalkeeperMetrics = {
    Saving: metrics.filter(m => /save|shot stopped|penalty save/i.test(m)),
    Claims: metrics.filter(m => /claim|catch|cross/i.test(m)),
    Passing: metrics.filter(m => /pass|distribution/i.test(m))
  };

  const categoryScore = (player, categoryMetrics) => {
    const existingMetrics = categoryMetrics.filter(m => metrics.includes(m));
    if (existingMetrics.length === 0) return 0;

    const scores = existingMetrics.map(m => {
      const values = filteredPlayers.map(p => p[m]);
      let pct = (values.filter(v => v <= player[m]).length / values.length) * 100;
      if (lowerIsBetterMetrics.includes(m)) pct = 100 - pct;
      return pct;
    });

    return Math.round(scores.reduce((a,b) => a+b,0)/scores.length);
  };

  const renderCategoryPie = (label, score, color) => (
    <div style={{ textAlign: "center", margin: "10px auto" }}>
      <PieChart width={120} height={120}>
        <Pie
          data={[
            { name: "Player", value: score },
            { name: "Remaining", value: 100 - score }
          ]}
          startAngle={90}
          endAngle={-270}
          innerRadius={30}
          outerRadius={50}
          dataKey="value"
        >
          <Cell key="player" fill={color} />
          <Cell key="remaining" fill="#eee" />
        </Pie>
        <text
          x={60} y={60} textAnchor="middle" dominantBaseline="middle" fill="#000"
          fontSize={14} fontWeight="bold"
        >
          {score}
        </text>
      </PieChart>
      <div style={{ marginTop: -10, fontSize:12 }}>
        <strong>{label}</strong>
      </div>
    </div>
  );

    const pizzaData = selectedPlayer
      ? percentilesForPlayer(selectedPlayer).map(p => ({
          metric: p.metric,
          player: p.value,
          league: p.leagueAvg
        }))
      : [];
    const renderRadarLabel = ({ x, y, payload, type }) => {
      if (!payload) return null;

      const rawValue = type === "player" ? payload.player : payload.league;
      const value = `${Math.round(rawValue)}%`;

      const boxWidth = 34;
      const boxHeight = 18;

      const fillColor = type === "player" ? "#1a78cf" : "#ffd700";
      const textColor = type === "player" ? "#fff" : "#000";

      return (
        <g transform={`translate(${x - boxWidth / 2}, ${y - 26})`}>
          <rect
            width={boxWidth}
            height={boxHeight}
            fill={fillColor}
            stroke="#000"
            strokeWidth={2} // black border
            rx={3}
            ry={3}
          />
          <text
            x={boxWidth / 2}
            y={boxHeight / 2 + 5}
            textAnchor="middle"
            fill={textColor}
            fontSize={10}
            fontWeight={700}
          >
            {value}
          </text>
        </g>
      );
    };






  const uniquePositions = Array.from(new Set(players.map(p => p["Primary Position"]).filter(Boolean)));

  const RadarTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{ background:"#fff", padding:10, border:"1px solid #ccc", borderRadius:6 }}>
          <div><strong>{data.metric}</strong></div>
          <div>Player: {data.value}</div>
          <div>League Avg: {data.avg}</div>
        </div>
      );
    }
    return null;
  };

    const generateScoutSummaryData = (player) => {
      if (!player) return { strengths: [], weaknesses: [], clipsLink: "#" };

      const pcts = percentilesForPlayer(player);

      const strengths = pcts.filter(p => p.value >= 75).map(p => p.metric);
      const weaknesses = pcts.filter(p => p.value <= 35).map(p => p.metric);

      const clipsLink = player.ClipsLink || "#"; // default link or from CSV

      return { strengths, weaknesses, clipsLink };
    };



  return (
    <div style={{
      padding: 24,
      fontFamily: "Arial",
      background: "url('/scouting-world.png') center/cover no-repeat, linear-gradient(to bottom, #cceeff, #ffffff)",
      minHeight: "100vh"
    }}>
      <h1 style={{ color: "#1f77b4", textAlign:"center" }}>Create Scouting & Analysis Reports</h1>
      <input type="file" accept=".csv" onChange={handleUpload} style={{ marginTop: 10 }} />

      {/* Filters + Right Column */}
      <div style={{ marginTop: 20, display:"grid", gridTemplateColumns:"260px 1fr", gap:24, maxWidth:1400, margin:"auto" }}>
        {/* Filter Panel */}
        <div style={{ background:"rgba(255,255,255,0.95)", padding:16, borderRadius:12, boxShadow:"0 4px 12px rgba(0,0,0,0.08)", height:"fit-content" }}>
          <h3 style={{ marginTop:0, marginBottom:12, color:"#1f77b4" }}>Filters</h3>
          <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Min Minutes</label><input type="number" onChange={e => setFilters({...filters, minMinutes:Number(e.target.value)})} /></div>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Max Minutes</label><input type="number" onChange={e => setFilters({...filters, maxMinutes:Number(e.target.value)})} /></div>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Competition</label>
              <select onChange={e=>setFilters({...filters, competition:e.target.value})}>
                {competitions.map(c=><option key={c}>{c}</option>)}
              </select>
            </div>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Position</label>
              <select multiple style={{height:80}} onChange={e=>setFilters({...filters, positions:Array.from(e.target.selectedOptions, o=>o.value)})}>
                {uniquePositions.map(pos=><option key={pos}>{pos}</option>)}
              </select>
            </div>
            <hr style={{ opacity:0.2 }} />
            <div><label style={{ fontSize:12, fontWeight:600 }}>Scatter X</label>
              <select value={scatterMetrics.x} onChange={e=>setScatterMetrics({...scatterMetrics, x:e.target.value})}>
                {metrics.map(m=><option key={m}>{m}</option>)}
              </select>
            </div>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Scatter Y</label>
              <select value={scatterMetrics.y} onChange={e=>setScatterMetrics({...scatterMetrics, y:e.target.value})}>
                {metrics.map(m=><option key={m}>{m}</option>)}
              </select>
            </div>
            <button onClick={applyFilters} style={{marginTop:10, background:"#1f77b4", color:"#fff", border:"none", padding:"10px", borderRadius:6, fontWeight:600, cursor:"pointer"}}>Apply Filters</button>
          </div>
        </div>

        {/* Right Column: Explanation + Shortlist */}
        <div style={{ display:"flex", flexDirection:"column", gap:20 }}>
          {/* Explanation */}
          <div style={{ background:"rgba(255,255,255,0.95)", padding:16, borderRadius:12, boxShadow:"0 4px 12px rgba(0,0,0,0.08)", fontSize:13, lineHeight:1.6 }}>
            <h3 style={{ color:"#1f77b4", marginTop:0 }}>How Ratings Work</h3>
            <ul style={{ paddingLeft:18, margin:0 }}>
              <li>Each metric is converted into a <strong>percentile rank</strong> within the filtered dataset.</li>
              <li>Metrics where <em>lower is better</em> (e.g. fouls, turnovers) are inverted.</li>
              <li>The <strong>Average Rating</strong> is the mean of all metric percentiles.</li>
              <li>Filters redefine the comparison group in real time.</li>
            </ul>
            <div style={{ marginTop:10, color:"#555" }}>Current sample size: <strong>{filteredPlayers.length}</strong> players</div>
          </div>

          {/* Shortlist */}
          <div style={{ background:"rgba(255,255,255,0.95)", padding:16, borderRadius:12, boxShadow:"0 4px 12px rgba(0,0,0,0.08)" }}>
            <h3 style={{ color:"#1f77b4", marginTop:0 }}>Top Rated Players</h3>
            {filteredPlayers.map(p=>({p, rating:Number(averageRating(p))})).sort((a,b)=>b.rating-a.rating).slice(0,8).map((r,i)=>(
              <div key={i} onClick={()=>setSelectedPlayer(r.p)} style={{ display:"flex", justifyContent:"space-between", padding:"6px 8px", borderRadius:6, cursor:"pointer", background:selectedPlayer===r.p?"#eaf3fb":"transparent", fontSize:13 }}>
                <span>{i+1}. {r.p["Player Name"]}</span>
                <strong>{r.rating}</strong>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Player Selector */}
      <div style={{ marginTop:5, textAlign:"center" }}>
        <select onChange={e=>setSelectedPlayer(filteredPlayers[e.target.value])} style={{ padding:5, borderRadius:5 }}>
          <option>Select Player</option>
          {filteredPlayers.map((p,i)=><option key={i} value={i}>{p["Player Name"]}</option>)}
        </select>
      </div>

      {/* Export PDF */}
      {selectedPlayer && <div style={{ marginTop:7, textAlign:"center" }}>
        <button onClick={exportPDF} style={{background:"#1f77b4", color:"#fff", border:"none", padding:"5px 15px", borderRadius:5, cursor:"pointer"}}>Export PDF</button>
      </div>}
          
          <button
            style={{
              marginLeft: 10,
              padding: "8px 14px",
              borderRadius: 6,
              border: "none",
              background: "#1a78cf",
              color: "#fff",
              fontWeight: 700,
              cursor: "pointer",
            }}
            onClick={() => {
              if (!selectedPlayer) {
                alert("Select a player first");
                return;
              }

                const playerName = selectedPlayer["Player Name"];

                if (!playerName) {
                  alert("Invalid player data");
                  return;
                }

                setShadowSquad(prev => {
                  // Prevent duplicates
                  if (prev.some(p => p.playerName === playerName)) {
                    alert("Player already in Shadow Squad");
                    return prev;
                  }

                  const rawPosition = selectedPlayer["Primary Position"];
                  const mapped = POSITION_MAP[rawPosition] || "CF";

                  // 🔥 AUTO-ASSIGN TO FORMATION SLOTS
                  let finalPosition = mapped;

                  if (mapped === "CM") {
                    const cm1Taken = prev.some(p => p.position === "CM1");
                    finalPosition = cm1Taken ? "CM2" : "CM1";
                  }

                  if (mapped === "LW" || mapped === "RW") {
                    const lwTaken = prev.some(p => p.position === "LW");
                    finalPosition = lwTaken ? "RW" : "LW";
                  }

                  if (mapped === "CF" || mapped === "ST") {
                    finalPosition = "CF";
                  }

                  const playerToAdd = {
                    id: selectedPlayer["Player Id"] || playerName,

                    playerName,
                    team:
                      selectedPlayer["Team"] ||
                      selectedPlayer["Club"] ||
                      selectedPlayer["Current Team"] ||
                      selectedPlayer["Squad"] ||   // 👈 add this if needed
                      "Unknown",

                    position: finalPosition,        // ✅ FORMATION-SAFE
                    fullPosition: rawPosition,

                    reportUrl: selectedPlayer["Report URL"] || "",

                    raw: selectedPlayer,
                  };

                  console.log("✅ Added to Shadow Squad:", playerToAdd);

                  return [...prev, playerToAdd];
                });

            }}
          >
            ➕ Add to Shadow Squad
          </button>





      {/* Player Report */}
      {selectedPlayer && <div ref={exportRef}>
          {/* PLAYER BANNER */}
          <div
            style={{
              marginTop: 24,
              padding: "16px 20px",
              background: "#1f77b4",
              color: "#fff",
              borderRadius: 12,
              display: "flex",
              alignItems: "center",
              gap: 18
            }}
          >
            {/* PLAYER PHOTO */}
            <div
              style={{
                width: 92,
                height: 92,
                borderRadius: "50%",
                background: "#ddd",
                overflow: "hidden",
                flexShrink: 0,
                border: "3px solid #fff"
              }}
            >
             <img
               src={getPlayerPhoto(selectedPlayer)}
               alt={selectedPlayer["Player Name"]}
               style={{
                 width: "100%",
                 height: "100%",
                 objectFit: "cover",
                 display: "block"
               }}
               onError={(e) => {
                 e.currentTarget.onerror = null; // prevent infinite loop
                 e.currentTarget.src = "/placeholder-player.png";
               }}
             />

            </div>

            {/* PLAYER NAME + META */}
            <div style={{ lineHeight: 1.25 }}>
              <div style={{ fontSize: 26, fontWeight: 700 }}>
                {selectedPlayer["Player Name"]}
              </div>

              <div style={{ fontSize: 14, opacity: 0.9 }}>
                {selectedPlayer["Primary Position"]}
                {" • "}
                {selectedPlayer.Team}
              </div>
            </div>
          </div>

          {/* Info Panel + Pie Charts */}
          <div style={{ marginTop:20, display:"grid", gridTemplateColumns:"300px 1fr", gap:40 }}>
            <div style={{ background:"rgba(255,255,255,0.9)", padding:20, borderRadius:12, lineHeight:2.8 }}>
          
              <div>Team: <strong>{selectedPlayer.Team}</strong></div>
              <div>Position: <strong>{selectedPlayer["Primary Position"]}</strong></div>
              <div>Competition: <strong>{selectedPlayer["Competition Name"]}</strong></div>
              <div>Age: <strong>{selectedPlayer.Age}</strong></div>
              <div>Nationality: <strong>{selectedPlayer.Nationality}</strong></div>
              <div>Games Played: <strong>{selectedPlayer["Appearances"]}</strong></div>
              <div>Minutes Played: <strong>{selectedPlayer["Minutes Played"]}</strong></div>
              <div>Average Rating: <strong>{averageRating(selectedPlayer)}</strong></div>
              <div>Overall Rank: <strong>{overallRank(selectedPlayer)} / {filteredPlayers.length}</strong></div>

              {/* PIE CHARTS */}
              <div style={{ display:"flex", flexDirection:"column", gap:20, marginTop:20 }}>
                {selectedPlayer["Primary Position"]?.toLowerCase() === "goalkeeper" ? (
                  <>
                    {renderCategoryPie("Saving", categoryScore(selectedPlayer, goalkeeperMetrics.Saving), "#2ecc71")}
                    {renderCategoryPie("Claims", categoryScore(selectedPlayer, goalkeeperMetrics.Claims), "#e74c3c")}
                    {renderCategoryPie("Passing", categoryScore(selectedPlayer, goalkeeperMetrics.Passing), "#1f77b4")}
                  </>
                ) : (
                  <>
                    {renderCategoryPie("Attacking", categoryScore(selectedPlayer, attackingMetrics), "#2ecc71")}
                    {renderCategoryPie("Defending", categoryScore(selectedPlayer, defendingMetrics), "#e74c3c")}
                    {renderCategoryPie("Ball Carrying", categoryScore(selectedPlayer, ballCarryingMetrics), "#1f77b4")}
                  </>
                )}
              </div>
            </div>

            {/* Metric Percentiles */}
            <div style={{ background:"rgba(255,255,255,0.9)", padding:20, borderRadius:12 }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:10, gap:12 }}>
                <h3 style={{ color:"#1f77b4", margin:0 }}>Metric Percentiles</h3>

                {/* Legend */}
                <div style={{ display:"flex", alignItems:"center", gap:14 }}>
                  <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                    <div style={{ width:30, height:10, background:"#2ecc71", borderRadius:3 }} />
                    <span style={{ fontSize:12 }}>Top 25%</span>
                  </div>

                  <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                    <div style={{ width:30, height:10, background:"#f39c12", borderRadius:3 }} />
                    <span style={{ fontSize:12 }}>50%</span>
                  </div>

                  <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                    <div style={{ width:30, height:10, background:"#e74c3c", borderRadius:3 }} />
                    <span style={{ fontSize:12 }}>Below 25%</span>
                  </div>

                  <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                    <div style={{ width:30, height:10, background:"#ffd700", borderRadius:3 }} />
                    <span style={{ fontSize:12 }}>League Avg</span>
                  </div>
                </div>
              </div>

              {percentilesForPlayer(selectedPlayer).map(m => (
                <div key={m.metric} style={{ marginBottom:16 }}>
                  <div style={{ display:"flex", justifyContent:"space-between" }}>
                    <div>
                      <strong>{m.metric}</strong>
                      <div style={{ fontSize:12, color:"#666" }}>{m.raw}</div>
                    </div>
                    <strong>{m.value}%</strong>
                  </div>
                  <div style={{ height:10, background:"#e0e0e0", borderRadius:5, marginTop:4 }}>
                    <div style={{
                      width:`${m.value}%`,
                      height:"100%",
                      background: m.value >= 75 ? "#2ecc71" : m.value >= 50 ? "#f39c12" : "#e74c3c",
                      borderRadius:5
                    }}></div>
                  </div>
                  <div style={{ height:10, background:"#eee", borderRadius:5, marginTop:4 }}>
                    <div style={{
                      width:`${m.leagueAvg}%`,
                      height:"100%",
                      background:"#ffd700",
                      borderRadius:5,
                      textAlign:"right",
                      color:"#000",
                      fontSize:10,
                      lineHeight:"10px",
                      paddingRight:2
                    }}>
                      <span>{m.leagueAvg}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Charts */}
          <div
            style={{
              marginTop: 32,
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 32,
              pageBreakInside: "avoid"
            }}
          >
          {/* Pizza Chart */}
          <div
            style={{
              background: "rgba(255,255,255,0.95)",
              border: "1px solid #ddd",
              borderRadius: 12,
              padding: 16
            }}
          >
            <div style={{ textAlign: "center", fontWeight: 700, marginBottom: 10 }}>
              Pizza Chart
            </div>

            <ResponsiveContainer width="100%" height={380}>
              <RadarChart data={pizzaData}>
                {/* Grid */}
                <PolarGrid
                  gridType="polygon"
                  radialLines
                  stroke="#000"
                  strokeOpacity={0.4}
                />

                {/* Metric labels */}
                <PolarAngleAxis
                  dataKey="metric"
                  tick={{
                    fill: "#000",
                    fontSize: 10,
                    fontWeight: 700
                  }}
                />

                {/* TOOLTIP */}
                <Tooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload || payload.length === 0) return null;

                    const league = payload.find(p => p.dataKey === "league")?.value;
                    const player = payload.find(p => p.dataKey === "player")?.value;

                    return (
                      <div
                        style={{
                          background: "#fff",
                          border: "1.5px solid #000",
                          borderRadius: 6,
                          padding: "8px 10px",
                          fontSize: 12
                        }}
                      >
                        <div style={{ fontWeight: 700, marginBottom: 4 }}>
                          {label}
                        </div>

                        <div style={{ color: "#1a78cf", fontWeight: 600 }}>
                          Player: {player}%
                        </div>

                        <div style={{ color: "#000", fontWeight: 600 }}>
                          League Avg: {league}%
                        </div>
                      </div>
                    );
                  }}
                />

                {/* LEAGUE AVERAGE */}
                <Radar
                  dataKey="league"
                  stroke="#000"
                  strokeWidth={3}
                  fill="#ffd700"
                  fillOpacity={1}
                  name="League Avg"
                  label={({ x, y, value }) => (
                    <g>
                      <rect
                        x={x - 16}
                        y={y - 11}
                        width={32}
                        height={18}
                        fill="#ffd700"
                        stroke="#000"
                        strokeWidth={1.5}
                        rx={3}
                        ry={3}
                      />
                      <text
                        x={x}
                        y={y + 3}
                        textAnchor="middle"
                        fontSize={10}
                        fontWeight={700}
                        fill="#000"
                      >
                        {value}%
                      </text>
                    </g>
                  )}
                />

                {/* PLAYER */}
                <Radar
                  dataKey="player"
                  stroke="#000"
                  strokeWidth={2}
                  fill="#1a78cf"
                  fillOpacity={0.85}
                  name={selectedPlayer["Player Name"]}
                  label={({ x, y, value }) => (
                    <g>
                      <rect
                        x={x - 16}
                        y={y - 11}
                        width={32}
                        height={18}
                        fill="#1a78cf"
                        stroke="#000"
                        strokeWidth={1.5}
                        rx={3}
                        ry={3}
                      />
                      <text
                        x={x}
                        y={y + 3}
                        textAnchor="middle"
                        fontSize={10}
                        fontWeight={700}
                        fill="#fff"
                      >
                        {value}%
                      </text>
                    </g>
                  )}
                />

                {/* Legend */}
                <Legend
                  verticalAlign="bottom"
                  align="center"
                  iconType="none"
                  formatter={(value) => {
                    const color = value === "League Avg" ? "#ffd700" : "#1a78cf";
                    return (
                      <span
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 6,
                          fontWeight: 600,
                          color: "#000"
                        }}
                      >
                        <span
                          style={{
                            width: 14,
                            height: 14,
                            backgroundColor: color,
                            border: "1px solid #000",
                            display: "inline-block"
                          }}
                        />
                        {value}
                      </span>
                    );
                  }}
                  wrapperStyle={{
                    marginTop: 10,
                    fontSize: 12
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>






          <div
            style={{
              background: "rgba(255,255,255,0.95)",
              border: "1px solid #ddd",
              borderRadius: 12,
              padding: 16
            }}
          >

              <div style={{ textAlign:"center", fontWeight:600, marginBottom:8, color:"#1f77b4" }}>
                {scatterMetrics.x} vs {scatterMetrics.y}
              </div>

              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart>
                  <CartesianGrid stroke="#ddd" />
                  <XAxis
                    dataKey={scatterMetrics.x}
                    type="number"
                    label={{ value: scatterMetrics.x, position:"bottom", offset:-8, fontWeight:"bold", fontSize:14 }}
                  />
                  <YAxis
                    dataKey={scatterMetrics.y}
                    type="number"
                    label={{ value: scatterMetrics.y, angle:-90, position:"left", offset:-2, fontWeight:"bold", fontSize:14 }}
                  />
                  <Tooltip content={<CustomScatterTooltip />} cursor={{ strokeDasharray: "3 3" }} />
                  <Scatter
                    data={filteredPlayers}
                    dataKeyX={scatterMetrics.x}
                    dataKeyY={scatterMetrics.y}
                    fill="#555555"
                  />
                  <Scatter
                    data={[selectedPlayer]}
                    dataKeyX={scatterMetrics.x}
                    dataKeyY={scatterMetrics.y}
                    fill="#ff7f0e"
                    shape="circle"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

              
                      {/* AI Scout Summary */}
                      <div style={{ marginTop: 35, marginLeft: 8, }}>
                        <h3 style={{ color: "#1f77b4", marginBottom: 15 }}>
                          Scout Summary – Strengths & Weaknesses
                        </h3>

                        <div
                          style={{
                            background: "rgba(255,255,255,0.95)",
                            border: "1px solid #ddd",
                            borderRadius: 8,
                            padding: 20,
                            lineHeight: 1.6,
                          }}
                        >
                          {selectedPlayer ? (() => {
                            const summary = generateScoutSummaryData(selectedPlayer);

                            return (
                              <div>
                                    {/* Strengths */}
                                    {summary.strengths.length > 0 && (
                                      <div
                                        style={{
                                          display: "flex",
                                          alignItems: "flex-start",
                                          marginBottom: 10,
                                        }}
                                      >
                                        <span
                                          style={{
                                            marginRight: 10,
                                            fontSize: 22,
                                            fontWeight: 700,
                                            lineHeight: "22px",
                                            color: "#2ecc71", // GREEN
                                            flexShrink: 0,
                                          }}
                                        >
                                         ＋
                                        </span>
                                        <div style={{ color: "#000", lineHeight: 1.6 }}>
                                          {summary.strengths.join(", ")}
                                        </div>
                                      </div>
                                    )}

                                    {/* Weaknesses */}
                                    {summary.weaknesses.length > 0 && (
                                      <div
                                        style={{
                                          display: "flex",
                                          alignItems: "flex-start",
                                          marginBottom: 10,
                                        }}
                                      >
                                        <span
                                          style={{
                                            marginRight: 10,
                                            fontSize: 22,
                                            fontWeight: 700,
                                            lineHeight: "22px",
                                            color: "#e74c3c", // RED
                                            flexShrink: 0,
                                          }}
                                        >
                                          −
                                        </span>
                                        <div style={{ color: "#000", lineHeight: 1.6 }}>
                                          {summary.weaknesses.join(", ")}
                                        </div>
                                      </div>
                                    )}



                                {/* Clips */}
                                <div
                                  style={{
                                    marginTop: 14,
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 6,
                                  }}
                                >
                                  <span role="img" aria-label="clips" style={{ fontSize: 18 }}>
                                    🎬
                                  </span>
                                    <input
                                      type="text"
                                      placeholder="Insert Player Clips Here"
                                      value={clipsLink}
                                      onChange={(e) => setClipsLink(e.target.value)}
                                      style={{
                                        border: "1px solid #ccc",
                                        borderRadius: 4,
                                        padding: "4px 8px",
                                        fontSize: 14,
                                        width: 220,
                                      }}
                                    />

                                </div>
                              </div>
                            );
                          })() : (
                            <div>No player selected</div>
                          )}
                        </div>
                      </div>



      </div>}

      <style>
        {`.no-export { display: block; } @media print { .no-export { display: none !important; } }`}
      </style>
    </div>
  );
}

export default ScoutReportPage;

