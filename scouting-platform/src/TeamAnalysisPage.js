import { ClubBadge } from "./ClubBadge";
import { useAuth } from "./AuthContext";
import React, { useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
} from "recharts";

const normalise = (value) => String(value ?? "").trim();
const displayName = (value) =>
  normalise(value)
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\s+/g, " ");
const pdfTitle = (value) => String(value ?? "").replace(/\s+/g, "\u00a0");
const number = (value) => {
  const parsed = Number(
    String(value ?? "")
      .replace(/%/g, "")
      .replace(/,/g, ""),
  );
  return Number.isFinite(parsed) ? parsed : null;
};
const lowerIsBetter = (metric) =>
  /conceded|faced|against|ppda|turnovers lost|errors?/.test(
    String(metric || "").toLowerCase(),
  );
const scoreColour = (score) =>
  score >= 75
    ? "#15c77a"
    : score >= 50
      ? "#ff9f1a"
      : score >= 25
        ? "#ffd21c"
        : "#ff4740";
const rankColour = (rank) =>
  rank <= 6
    ? "#15c77a"
    : rank <= 13
      ? "#ffd21c"
      : rank <= 19
        ? "#ff9f1a"
        : "#ff4740";
const rankLabel = (rank) => {
  if (!rank) return "-";
  const suffix = rank % 100 >= 11 && rank % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[rank % 10] || "th");
  return `${rank}${suffix}`;
};
const openTeamAnalysisDb = () => new Promise((resolve, reject) => {
  if (!window.indexedDB) return reject(new Error("Browser storage is unavailable"));
  const request = window.indexedDB.open("scoutpro-team-analysis", 1);
  request.onupgradeneeded = () => {
    if (!request.result.objectStoreNames.contains("uploads")) {
      request.result.createObjectStore("uploads");
    }
  };
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error || new Error("Could not open browser storage"));
});
const readTeamAnalysisUpload = async (key) => {
  const db = await openTeamAnalysisDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction("uploads", "readonly").objectStore("uploads").get(key);
    request.onsuccess = () => { db.close(); resolve(request.result || null); };
    request.onerror = () => { db.close(); reject(request.error || new Error("Could not read saved upload")); };
  });
};
const writeTeamAnalysisUpload = async (key, value) => {
  const db = await openTeamAnalysisDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction("uploads", "readwrite").objectStore("uploads").put(value, key);
    request.onsuccess = () => { db.close(); resolve(); };
    request.onerror = () => { db.close(); reject(request.error || new Error("Could not save uploaded data")); };
  });
};
const metricGroup = (metric) => {
  const label = metric.toLowerCase();
  if (/set.?piece|corner|free.?kick|dead.?ball|throw.?in|dfk/.test(label))
    return "Set-Pieces";
  if (
    /tackle|intercept|clearance|pressure|regain|defensive|duel|block|conceded|faced|against|ppda|aerial|defend|offside|aggressive actions/.test(
      label,
    )
  )
    return "Out Of Possession";
  return "In Possession";
};
const styleGroup = (metric) => {
  const label = String(metric || "").toLowerCase();
  if (/set.?piece|corner|free.?kick|dead.?ball|throw.?in|dfk/.test(label))
    return "Set-Pieces";
  if (
    /tackle|intercept|clearance|pressure|regain|defensive|duel|block|conceded|faced|against|ppda|aerial|defend|offside|aggressive actions/.test(
      label,
    )
  )
    return "Defensive Work";
  if (
    /xg|expected goals|assist|chance|key pass|scoring contribution|touch(es)? in box|creation/.test(
      label,
    )
  )
    return "Chance Creation";
  if (/goal|finish|conversion|shot/.test(label)) return "Finishing";
  if (/xg|expected|carry|dribble|possession|pass/.test(label))
    return "Build-up & Possession";
  return "Attacking Output";
};
class ChartBoundary extends React.Component {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <div style={{ padding: 24, color: "#667" }}>
        Chart unavailable for this upload.
      </div>
    ) : (
      this.props.children
    );
  }
}

const DEFAULT_TACTICAL_HOME = [
  [90, 320], [245, 90], [245, 245], [245, 395], [245, 550],
  [420, 170], [420, 320], [420, 470], [610, 145], [610, 320], [610, 495],
].map(([x, y], index) => ({ id: `home-${index + 1}`, side: "home", number: index + 1, x, y }));
const DEFAULT_TACTICAL_AWAY = [
  [910, 320], [755, 90], [755, 245], [755, 395], [755, 550],
  [580, 170], [580, 320], [580, 470], [390, 145], [390, 320], [390, 495],
].map(([x, y], index) => ({ id: `away-${index + 1}`, side: "away", number: index + 1, x, y }));
const freshTacticalBoard = (homeName = "Home Team", awayName = "Opposition") => ({
  name: "New Tactic",
  homeName: homeName || "Home Team",
  awayName: awayName || "Opposition",
  players: [...DEFAULT_TACTICAL_HOME, ...DEFAULT_TACTICAL_AWAY],
  arrows: [],
});

const drawTacticalVideoFrame = (canvas, board) => {
  if (!canvas) return;
  const width = canvas.width;
  const height = canvas.height;
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, width, height);
  context.save();
  context.scale(width / 1000, height / 640);
  context.fillStyle = "#24734e";
  context.fillRect(12, 12, 976, 616);
  context.strokeStyle = "#e8f4e9";
  context.lineWidth = 3;
  context.strokeRect(35, 35, 930, 570);
  context.beginPath(); context.moveTo(500, 35); context.lineTo(500, 605); context.stroke();
  context.beginPath(); context.arc(500, 320, 83, 0, Math.PI * 2); context.stroke();
  context.fillStyle = "#e8f4e9";
  [[500, 320], [145, 320], [855, 320]].forEach(([x, y]) => { context.beginPath(); context.arc(x, y, 4, 0, Math.PI * 2); context.fill(); });
  context.strokeRect(35, 180, 150, 280); context.strokeRect(35, 245, 58, 150);
  context.strokeRect(815, 180, 150, 280); context.strokeRect(907, 245, 58, 150);
  context.beginPath(); context.arc(145, 320, 82, -0.83, 0.83); context.stroke();
  context.beginPath(); context.arc(855, 320, 82, Math.PI - 0.83, Math.PI + 0.83); context.stroke();
  context.fillStyle = "#fff";
  context.font = "bold 24px Arial, sans-serif";
  context.textAlign = "left";
  context.fillText(board.homeName || "Home Team", 42, 76);
  context.textAlign = "right";
  context.fillText(board.awayName || "Opposition", 958, 76);
  board.arrows.forEach((arrow) => {
    const angle = Math.atan2(arrow.to.y - arrow.from.y, arrow.to.x - arrow.from.x);
    context.strokeStyle = "#ffe36e"; context.fillStyle = "#ffe36e"; context.lineWidth = 6; context.lineCap = "round";
    context.beginPath(); context.moveTo(arrow.from.x, arrow.from.y); context.lineTo(arrow.to.x, arrow.to.y); context.stroke();
    context.beginPath(); context.moveTo(arrow.to.x, arrow.to.y);
    context.lineTo(arrow.to.x - 20 * Math.cos(angle - Math.PI / 6), arrow.to.y - 20 * Math.sin(angle - Math.PI / 6));
    context.lineTo(arrow.to.x - 20 * Math.cos(angle + Math.PI / 6), arrow.to.y - 20 * Math.sin(angle + Math.PI / 6));
    context.closePath(); context.fill();
  });
  board.players.forEach((player) => {
    context.beginPath(); context.arc(player.x, player.y + 3, 27, 0, Math.PI * 2); context.fillStyle = "#05223f70"; context.fill();
    context.beginPath(); context.arc(player.x, player.y, 24, 0, Math.PI * 2); context.fillStyle = player.side === "home" ? "#1485dd" : "#e64b50"; context.fill();
    context.strokeStyle = "#f3f8ff"; context.lineWidth = 3; context.stroke();
    context.fillStyle = "#fff"; context.font = "bold 17px Arial, sans-serif"; context.textAlign = "center"; context.textBaseline = "middle";
    context.fillText(String(player.number), player.x, player.y + 1);
  });
  context.restore();
};

function TacticalPad({ storageKey, homeTeamName, oppositionName }) {
  const [board, setBoard] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
      return saved?.players?.length ? { ...freshTacticalBoard(homeTeamName, oppositionName), ...saved } : freshTacticalBoard(homeTeamName, oppositionName);
    } catch { return freshTacticalBoard(homeTeamName, oppositionName); }
  });
  const [mode, setMode] = useState("move");
  const [draggingId, setDraggingId] = useState(null);
  const [arrowStart, setArrowStart] = useState(null);
  const [arrowPreview, setArrowPreview] = useState(null);
  const [selectedToken, setSelectedToken] = useState(null);
  const [videoRecording, setVideoRecording] = useState(false);
  const [videoStatus, setVideoStatus] = useState("");
  const pitchRef = useRef(null);
  const videoCanvasRef = useRef(null);
  const boardRef = useRef(board);
  const videoRecorderRef = useRef(null);
  const videoChunksRef = useRef([]);
  const videoFrameRef = useRef(null);

  useEffect(() => { boardRef.current = board; }, [board]);

  useEffect(() => {
    const saveTimer = window.setTimeout(() => {
      try { localStorage.setItem(storageKey, JSON.stringify(board)); } catch {}
    }, 250);
    return () => window.clearTimeout(saveTimer);
  }, [board, storageKey]);
  useEffect(() => () => {
    if (videoFrameRef.current) window.cancelAnimationFrame(videoFrameRef.current);
    const recorder = videoRecorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);

  const pointFromEvent = (event) => {
    const rect = pitchRef.current?.getBoundingClientRect();
    if (!rect) return { x: 500, y: 320 };
    return {
      x: Math.max(24, Math.min(976, ((event.clientX - rect.left) / rect.width) * 1000)),
      y: Math.max(24, Math.min(616, ((event.clientY - rect.top) / rect.height) * 640)),
    };
  };
  const beginPlayerDrag = (event, player) => {
    if (mode !== "move") return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setDraggingId(player.id);
    setSelectedToken(player.id);
  };
  const handlePitchPointerDown = (event) => {
    if (mode !== "arrow") return;
    pitchRef.current?.setPointerCapture?.(event.pointerId);
    const point = pointFromEvent(event);
    setArrowStart(point);
    setArrowPreview(point);
  };
  const handlePitchPointerMove = (event) => {
    const point = pointFromEvent(event);
    if (draggingId) {
      setBoard((current) => ({ ...current, players: current.players.map((player) => player.id === draggingId ? { ...player, ...point } : player) }));
    }
    if (arrowStart) setArrowPreview(point);
  };
  const handlePitchPointerUp = (event) => {
    if (arrowStart) {
      const end = pointFromEvent(event);
      const distance = Math.hypot(end.x - arrowStart.x, end.y - arrowStart.y);
      if (distance > 12) setBoard((current) => ({ ...current, arrows: [...current.arrows, { id: `${Date.now()}-${current.arrows.length}`, from: arrowStart, to: end }] }));
      setArrowStart(null);
      setArrowPreview(null);
    }
    setDraggingId(null);
  };
  const addPlayer = (side) => {
    setBoard((current) => {
      const sidePlayers = current.players.filter((player) => player.side === side);
      if (sidePlayers.length >= 16) return current;
      const number = sidePlayers.reduce((max, player) => Math.max(max, Number(player.number) || 0), 0) + 1;
      const player = { id: `${side}-${Date.now()}`, side, number, x: side === "home" ? 350 : 650, y: 320 + (sidePlayers.length % 5) * 18 };
      return { ...current, players: [...current.players, player] };
    });
  };
  const removeSelectedPlayer = () => {
    if (!selectedToken) return;
    setBoard((current) => ({ ...current, players: current.players.filter((player) => player.id !== selectedToken) }));
    setSelectedToken(null);
  };
  const startVideoRecording = () => {
    const canvas = videoCanvasRef.current;
    if (!canvas?.captureStream || typeof MediaRecorder === "undefined") {
      setVideoStatus("Video recording is not supported by this browser. Try Chrome, Edge, or Safari.");
      return;
    }
    drawTacticalVideoFrame(canvas, boardRef.current);
    const stream = canvas.captureStream(30);
    const formats = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
    const mimeType = formats.find((format) => MediaRecorder.isTypeSupported?.(format));
    let recorder;
    try {
      recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    } catch (recordError) {
      stream.getTracks().forEach((track) => track.stop());
      setVideoStatus(recordError?.message || "Could not start video recording in this browser.");
      return;
    }
    videoChunksRef.current = [];
    recorder.ondataavailable = (event) => { if (event.data?.size) videoChunksRef.current.push(event.data); };
    recorder.onstop = () => {
      const blob = new Blob(videoChunksRef.current, { type: recorder.mimeType || mimeType || "video/webm" });
      if (blob.size) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        const extension = blob.type.includes("mp4") ? "mp4" : "webm";
        const filename = (boardRef.current.name || "tactical-movement").trim().replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "tactical-movement";
        link.href = url; link.download = `${filename}.${extension}`; link.click();
        window.setTimeout(() => URL.revokeObjectURL(url), 1500);
        setVideoStatus("Video downloaded. Move the discs while recording to show the play.");
      } else setVideoStatus("No video frames were recorded. Please try again.");
      stream.getTracks().forEach((track) => track.stop());
      videoRecorderRef.current = null;
    };
    recorder.onerror = () => setVideoStatus("Video recording stopped because of a browser error.");
    videoRecorderRef.current = recorder;
    recorder.start(250);
    setVideoRecording(true);
    setVideoStatus("Recording. Move the discs to show the play, then stop and download.");
    const drawFrame = () => {
      drawTacticalVideoFrame(videoCanvasRef.current, boardRef.current);
      videoFrameRef.current = window.requestAnimationFrame(drawFrame);
    };
    videoFrameRef.current = window.requestAnimationFrame(drawFrame);
  };
  const stopVideoRecording = () => {
    const recorder = videoRecorderRef.current;
    if (videoFrameRef.current) window.cancelAnimationFrame(videoFrameRef.current);
    videoFrameRef.current = null;
    setVideoRecording(false);
    if (recorder && recorder.state !== "inactive") recorder.stop();
  };

  return (
    <section className="sr-tactical-pad">
      <div className="sr-tactical-toolbar">
        <div className="sr-tactical-title-fields">
          <label><span>Tactic name</span><input value={board.name} onChange={(event) => setBoard((current) => ({ ...current, name: event.target.value }))} /></label>
          <label><span>Home team</span><input value={board.homeName} onChange={(event) => setBoard((current) => ({ ...current, homeName: event.target.value }))} /></label>
          <label><span>Opposition</span><input value={board.awayName} onChange={(event) => setBoard((current) => ({ ...current, awayName: event.target.value }))} /></label>
        </div>
        <div className="sr-tactical-tools" role="toolbar" aria-label="Tactical pad tools">
          <button type="button" className={mode === "move" ? "selected" : ""} onClick={() => setMode("move")}>Move players</button>
          <button type="button" className={mode === "arrow" ? "selected" : ""} onClick={() => setMode("arrow")}>Draw arrows</button>
          <button type="button" onClick={() => addPlayer("home")}>+ Home disc</button>
          <button type="button" onClick={() => addPlayer("away")}>+ Opponent disc</button>
          <button type="button" onClick={removeSelectedPlayer} disabled={!selectedToken}>Remove selected</button>
          <button type="button" onClick={() => setBoard((current) => ({ ...current, arrows: [] }))} disabled={!board.arrows.length}>Clear arrows</button>
          <button type="button" onClick={() => { setBoard(freshTacticalBoard(homeTeamName, oppositionName)); setSelectedToken(null); }}>Reset pitch</button>
          <button type="button" className={videoRecording ? "sr-tactical-recording" : "sr-tactical-video"} onClick={videoRecording ? stopVideoRecording : startVideoRecording}>{videoRecording ? "Stop & download video" : "Record video"}</button>
        </div>
      </div>
      {videoStatus && <p className="sr-tactical-video-status" role="status">{videoStatus}</p>}
      <canvas ref={videoCanvasRef} width="1280" height="819" className="sr-tactical-video-canvas" aria-hidden="true" />

      <div className={`sr-tactical-pitch-wrap ${mode === "arrow" ? "drawing" : "moving"}`}>
        <svg ref={pitchRef} className="sr-tactical-pitch" viewBox="0 0 1000 640" role="img" aria-label="Tactical football pitch. Drag the discs to position players." onPointerDown={handlePitchPointerDown} onPointerMove={handlePitchPointerMove} onPointerUp={handlePitchPointerUp} onPointerCancel={handlePitchPointerUp}>
          <defs><marker id="sr-tactical-arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#ffe36e" /></marker></defs>
          <rect x="12" y="12" width="976" height="616" rx="7" className="sr-tactical-grass" />
          <rect x="35" y="35" width="930" height="570" className="sr-tactical-line" />
          <line x1="500" y1="35" x2="500" y2="605" className="sr-tactical-line" />
          <circle cx="500" cy="320" r="83" className="sr-tactical-line" />
          <circle cx="500" cy="320" r="4" className="sr-tactical-mark" />
          <rect x="35" y="180" width="150" height="280" className="sr-tactical-line" />
          <rect x="35" y="245" width="58" height="150" className="sr-tactical-line" />
          <rect x="815" y="180" width="150" height="280" className="sr-tactical-line" />
          <rect x="907" y="245" width="58" height="150" className="sr-tactical-line" />
          <path d="M185 250 A82 82 0 0 1 185 390 M815 250 A82 82 0 0 0 815 390" className="sr-tactical-line" />
          <circle cx="145" cy="320" r="4" className="sr-tactical-mark" />
          <circle cx="855" cy="320" r="4" className="sr-tactical-mark" />
          <path d="M35 55 A20 20 0 0 0 55 35 M945 35 A20 20 0 0 0 965 55 M35 585 A20 20 0 0 1 55 605 M945 605 A20 20 0 0 1 965 585" className="sr-tactical-line" />
          {board.arrows.map((arrow) => <line key={arrow.id} x1={arrow.from.x} y1={arrow.from.y} x2={arrow.to.x} y2={arrow.to.y} className="sr-tactical-arrow" markerEnd="url(#sr-tactical-arrowhead)" />)}
          {arrowPreview && arrowStart && <line x1={arrowStart.x} y1={arrowStart.y} x2={arrowPreview.x} y2={arrowPreview.y} className="sr-tactical-arrow sr-tactical-arrow-preview" markerEnd="url(#sr-tactical-arrowhead)" />}
          {board.players.map((player) => <g key={player.id} className={`sr-tactical-player ${player.side} ${selectedToken === player.id ? "selected" : ""}`} transform={`translate(${player.x} ${player.y})`} onPointerDown={(event) => beginPlayerDrag(event, player)} onClick={() => setSelectedToken(player.id)}>
            <circle r="27" className="sr-tactical-disc-shadow" />
            <circle r="24" className="sr-tactical-disc" />
            <text y="1">{player.number}</text>
          </g>)}
        </svg>
      </div>
      <div className="sr-tactical-legend"><span><i className="home" />{board.homeName}</span><span><i className="away" />{board.awayName}</span><small>Drag discs to move players. Select a disc, then remove it if needed. Draw arrows to show runs or passes. This board saves automatically on this device.</small></div>
    </section>
  );
}

export default function TeamAnalysisPage() {
  const { user, loading: authLoading } = useAuth();
  const [rows, setRows] = useState([]);
  const [selectedTeam, setSelectedTeam] = useState("");
  const [comparisonTeam, setComparisonTeam] = useState("");
  const [selectedMetrics, setSelectedMetrics] = useState(null);
  const [comparisonMetrics, setComparisonMetrics] = useState(null);
  const [error, setError] = useState("");
  const [storageWarning, setStorageWarning] = useState("");
  const [dataStorageKey, setDataStorageKey] = useState("");
  const [isRestoring, setIsRestoring] = useState(true);
  const [moreOpen, setMoreOpen] = useState(false);
  const [tacticalPadOpen, setTacticalPadOpen] = useState(false);
  const dashboardRef = useRef(null);
  const hydrated = useRef(false);
  const storageKey = `scoutpro-team-analysis-uploaded-data:${user?.id || "anonymous"}`;

  useEffect(() => {
    hydrated.current = false;
    setIsRestoring(true);
    if (authLoading) return undefined;
    let cancelled = false;
    const restore = async () => {
      let parsed = null;
      try {
        const saved = window.localStorage.getItem(storageKey);
        parsed = saved ? JSON.parse(saved) : null;
      } catch {
        // Fall back to IndexedDB if localStorage is full or unavailable.
      }
      if (!parsed) {
        try {
          parsed = await readTeamAnalysisUpload(storageKey);
        } catch {
          // No saved upload is available in this browser.
        }
      }
      if (cancelled) return;
      if (parsed && Array.isArray(parsed.rows)) {
        setRows(parsed.rows);
        setSelectedTeam(parsed.selectedTeam || "");
        setComparisonTeam(parsed.comparisonTeam || "");
        const restoreMetrics = (value) => {
          if (!Array.isArray(value)) return null;
          if (parsed.filterVersion !== 2 && value.length === 0) return null;
          return value;
        };
        setSelectedMetrics(restoreMetrics(parsed.selectedMetrics));
        setComparisonMetrics(restoreMetrics(parsed.comparisonMetrics));
      } else {
        setRows([]);
        setSelectedTeam("");
        setComparisonTeam("");
        setSelectedMetrics(null);
        setComparisonMetrics(null);
      }
      setDataStorageKey(storageKey);
      hydrated.current = true;
      setIsRestoring(false);
    };
    restore();
    return () => { cancelled = true; };
  }, [storageKey, authLoading]);

  useEffect(() => {
    if (authLoading || !hydrated.current || dataStorageKey !== storageKey || !rows.length) return;
    const savedUpload = { rows, selectedTeam, comparisonTeam, selectedMetrics, comparisonMetrics, filterVersion: 2 };
    let localStorageSaved = true;
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(savedUpload));
    } catch {
      localStorageSaved = false;
    }
    writeTeamAnalysisUpload(storageKey, savedUpload).then(() => {
      setStorageWarning("");
    }).catch(() => {
      if (!localStorageSaved) setStorageWarning("Your data is loaded, but this browser could not save it for next time. Try a smaller file or clear some browser storage.");
    });
  }, [rows, selectedTeam, comparisonTeam, selectedMetrics, comparisonMetrics, storageKey, dataStorageKey, authLoading]);

  const teamKey = useMemo(() => {
    const keys = Object.keys(rows[0] || {});
    return (
      keys.find((key) =>
        /^(team|club|squad|side|team name|club name)$/i.test(key),
      ) ||
      keys[0] ||
      "Team"
    );
  }, [rows]);

  const teamRows = useMemo(
    () =>
      rows.filter(
        (row) =>
          !/^(league average|league avg|average|avg)$/i.test(
            normalise(row[teamKey]),
          ),
      ),
    [rows, teamKey],
  );
  const teams = useMemo(
    () => [
      ...new Set(
        teamRows.map((row) => normalise(row[teamKey])).filter(Boolean),
      ),
    ],
    [teamRows, teamKey],
  );
  const metrics = useMemo(() => {
    const keys = Object.keys(rows[0] || {});
    return keys.filter(
      (key) =>
        key !== teamKey &&
        !/^(name|player|league|competition|position|country|season|id|rank|games?|games played|matches?|appearances?)$/i.test(
          key,
        ) &&
        rows.filter((row) => number(row[key]) !== null).length >=
          Math.max(3, Math.floor(rows.length * 0.6)),
    );
  }, [rows, teamKey]);

  const activeMetrics = selectedMetrics !== null
    ? metrics.filter((metric) => selectedMetrics.includes(metric))
    : metrics;
  const comparisonActiveMetrics = comparisonMetrics !== null
    ? metrics.filter((metric) => comparisonMetrics.includes(metric))
    : metrics;

  const selected =
    teamRows.find((row) => normalise(row[teamKey]) === selectedTeam) ||
    teamRows[0] ||
    null;
  const selectedTeamName = displayName(selected?.[teamKey]);
  const gamesKey = Object.keys(selected || {}).find((key) =>
    /^(games?|games played|matches?|appearances?)$/i.test(key),
  );
  const chartData = activeMetrics.map((metric) => {
    const values = teamRows
      .map((row) => number(row[metric]))
      .filter((value) => value !== null);
    const value = number(selected?.[metric]);
    const average = values.length
      ? values.reduce((sum, item) => sum + item, 0) / values.length
      : 0;
    const maximum = values.length ? Math.max(...values) : 0;
    const percentile =
      value === null || !values.length
        ? 0
        : ((lowerIsBetter(metric)
            ? values.filter((item) => item >= value).length
            : values.filter((item) => item <= value).length) /
            values.length) *
          100;
    return {
      metric,
      percentile: Math.round(percentile),
      teamValue: value ?? 0,
      leagueAverage: Number(average.toFixed(2)),
      leagueAveragePct: maximum ? Math.round((average / maximum) * 100) : 0,
    };
  });
  const styleData = [
    "Build-up & Possession",
    "Chance Creation",
    "Finishing",
    "Defensive Work",
    "Pressing & Regains",
    "Set-Pieces",
  ].map((style) => {
    const matching = chartData.filter(
      (item) =>
        styleGroup(item.metric) === style ||
        (style === "Pressing & Regains" &&
          /pressure|regain/i.test(item.metric)),
    );
    return {
      style,
      score: matching.length
        ? Math.round(
            matching.reduce((sum, item) => sum + item.percentile, 0) /
              matching.length,
          )
        : 0,
      leagueAverage: matching.length
        ? Math.round(
            matching.reduce((sum, item) => sum + item.leagueAveragePct, 0) /
              matching.length,
          )
        : 0,
    };
  });
  const comparison =
    teamRows.find((row) => normalise(row[teamKey]) === comparisonTeam) ||
    teamRows.find((row) => normalise(row[teamKey]) !== selectedTeam) ||
    null;
  const comparisonTeamName = displayName(comparison?.[teamKey]);
  const comparisonData = comparisonActiveMetrics.map((metric) => {
    const first = number(selected?.[metric]) ?? 0;
    const second = number(comparison?.[metric]) ?? 0;
    const scale = Math.max(
      ...rows.map((row) => number(row[metric]) ?? 0),
      first,
      second,
      1,
    );
    const difference = Number((first - second).toFixed(2));
    return {
      metric,
      first,
      second,
      firstPct: Math.round((first / scale) * 100),
      secondPct: Math.round((second / scale) * 100),
      difference,
      differencePct: Math.min(
        100,
        Math.round((Math.abs(difference) / scale) * 100),
      ),
    };
  });
  const metricColumns = ["In Possession", "Out Of Possession", "Set-Pieces"];
  const renderMetric = (item) => (
    <div key={item.metric} style={{ marginBottom: 14, breakInside: "avoid" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "end",
          gap: 10,
        }}
      >
        <div>
          <strong style={{ display: "block", fontSize: 14, color: "#fff" }}>
            {item.metric}
          </strong>
          <small style={{ color: "#d2e5fa" }}>{item.teamValue}</small>
        </div>
        <strong style={{ color: "#fff", fontSize: 13 }}>
          {item.percentile}%
        </strong>
      </div>
      <div
        style={{
          height: 11,
          marginTop: 4,
          borderRadius: 8,
          background: "#365b80",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${item.percentile}%`,
            height: "100%",
            background: scoreColour(item.percentile),
            borderRadius: 8,
          }}
        />
      </div>
      <div
        style={{
          height: 11,
          marginTop: 4,
          borderRadius: 8,
          background: "#284d73",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${item.leagueAveragePct}%`,
            height: "100%",
            background: "#ffd21c",
            borderRadius: 8,
          }}
        />
        <span
          style={{
            position: "relative",
            display: "block",
            marginTop: -11,
            textAlign: "center",
            fontSize: 9,
            fontWeight: 800,
            color: "#111",
          }}
        >
          {item.leagueAveragePct}%
        </span>
      </div>
    </div>
  );

  const upload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError("");
    const done = (data) => {
      const clean = (data || []).filter(
        (row) =>
          row &&
          typeof row === "object" &&
          Object.values(row).some((value) => normalise(value)),
      );
      if (!clean.length) {
        setRows([]);
        setError("No readable team data was found in that file.");
        return;
      }
      setRows(clean);
      setDataStorageKey(storageKey);
      const keys = Object.keys(clean[0] || {});
      const detectedTeamKey =
        keys.find((key) =>
          /^(team|club|squad|side|team name|club name)$/i.test(key),
        ) ||
        keys[0] ||
        "Team";
      const uploadedTeams = clean.filter(
        (row) =>
          !/^(league average|league avg|average|avg)$/i.test(
            normalise(row[detectedTeamKey]),
          ),
      );
      setSelectedTeam(normalise(uploadedTeams[0]?.[detectedTeamKey]) || "");
      setComparisonTeam(normalise(uploadedTeams[1]?.[detectedTeamKey]) || "");
      setSelectedMetrics(null);
      setComparisonMetrics(null);
      try {
        window.localStorage.setItem(storageKey, JSON.stringify({ rows: clean, selectedTeam: normalise(uploadedTeams[0]?.[detectedTeamKey]) || "", comparisonTeam: normalise(uploadedTeams[1]?.[detectedTeamKey]) || "", selectedMetrics: null, comparisonMetrics: null, filterVersion: 2 }));
        setStorageWarning("");
      } catch {
        setStorageWarning("Your data is loaded, but this browser could not save it for next time. Try a smaller file or clear some browser storage.");
      }
    };
    if (/\.xlsx?$/.test(file.name.toLowerCase())) {
      file
        .arrayBuffer()
        .then((buffer) => {
          const workbook = XLSX.read(buffer, { type: "array" });
          const sheet = workbook.Sheets[workbook.SheetNames[0]];
          done(XLSX.utils.sheet_to_json(sheet, { defval: "" }));
        })
        .catch(() => setError("Could not read that spreadsheet."));
    } else {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (result) => done(result.data),
        error: () => setError("Could not read that CSV file."),
      });
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
    pdf.rect(
      0,
      0,
      pdf.internal.pageSize.getWidth(),
      pdf.internal.pageSize.getHeight(),
      "F",
    );
    const ratio = Math.min(
      pageWidth / canvas.width,
      pageHeight / canvas.height,
    );
    const width = canvas.width * ratio;
    const height = canvas.height * ratio;
    pdf.addImage(
      canvas.toDataURL("image/png"),
      "PNG",
      margin + (pageWidth - width) / 2,
      margin + (pageHeight - height) / 2,
      width,
      height,
    );
    pdf.save(`${normalise(selected?.[teamKey]) || "team"}-analysis-report.pdf`);
  };

  const moreMenu = (
    <div className="sr-team-analysis-more">
      <button type="button" className="sr-team-analysis-more-trigger" aria-haspopup="menu" aria-expanded={moreOpen} onClick={() => setMoreOpen((open) => !open)}>More <span aria-hidden="true">▾</span></button>
      {moreOpen && <div className="sr-team-analysis-more-menu" role="menu">
        <button type="button" role="menuitem" onClick={() => { setTacticalPadOpen((open) => !open); setMoreOpen(false); }}>{tacticalPadOpen ? "Team Analysis" : "Tactical Pad"}</button>
      </div>}
    </div>
  );
  const pageHeader = (title, description, showBack = false) => (
    <header className="sr-team-analysis-header">
      <div>
        <div style={{ color: "#6bd7fa", fontSize: 14, letterSpacing: 1 }}>SCOUTPRO PLATFORM</div>
        <h1 style={{ margin: "10px 0", fontSize: 48, fontWeight: 800, color: "#62dcff", textTransform: "uppercase", fontFamily: 'Impact,"Arial Narrow",sans-serif' }}>{title}</h1>
        <p style={{ color: "#d7e8f8", fontSize: 18, margin: 0 }}>{description}</p>
        {showBack && <button type="button" className="sr-team-analysis-back" onClick={() => setTacticalPadOpen(false)}>‹ Back to Team Analysis</button>}
      </div>
      {moreMenu}
    </header>
  );
  const pageStyle = {
    minHeight: "calc(100vh - 80px)",
    background: "linear-gradient(135deg,#062c63 0%,#063d74 100%)",
    color: "#fff",
    padding: "42px clamp(22px,5vw,72px) 70px",
    boxSizing: "border-box",
  };

  if (tacticalPadOpen) {
    return <main style={pageStyle}>
      <div style={{ maxWidth: 1400, margin: "0 auto" }}>
        {pageHeader("Tactical Pad", "Set up a shape, move players and draw runs or passes.", true)}
        <TacticalPad storageKey={`scoutpro-tactical-board:${user?.id || "anonymous"}`} homeTeamName={selectedTeamName} oppositionName={comparisonTeamName} />
      </div>
    </main>;
  }

  return (
    <main style={pageStyle}>
      <style>{`[data-old-metric-section]{display:none!important}[data-grouped-metrics]{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;order:3!important}[data-grouped-metrics] span{color:#fff!important}section[style*="order: 3"]{order:2!important}section[style*="order: 4"]{display:none!important}@media(max-width:800px){[data-grouped-metrics]{grid-template-columns:1fr}}`}</style>
      <div style={{ maxWidth: 1400, margin: "0 auto" }}>
        {pageHeader("Team Analysis", "Upload league team data to compare performance across every available playing metric.")}
        <section
          style={{
            marginTop: 28,
            padding: 24,
            borderRadius: 14,
            background: "rgba(255,255,255,.12)",
            border: "1px solid rgba(255,255,255,.35)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 18,
              flexWrap: "wrap",
            }}
          >
            <label style={{ fontWeight: 700 }}>Upload team data</label>
            <button
              type="button"
              onClick={exportPDF}
              disabled={!rows.length}
              style={{
                border: "1px solid #b9eaff",
                borderRadius: 7,
                padding: "11px 18px",
                fontWeight: 700,
                cursor: rows.length ? "pointer" : "not-allowed",
                background: rows.length ? "#62dcff" : "#47718e",
                color: "#063d63",
              }}
            >
              Export one-page PDF
            </button>
          </div>
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={upload}
            disabled={isRestoring}
            style={{ color: "#fff" }}
          />
          {error && <p style={{ color: "#ffd0d0" }}>{error}</p>}
          {storageWarning && <p role="status" style={{ color: "#ffe9a8", marginBottom: 0 }}>{storageWarning}</p>}
        </section>
        {isRestoring ? (
          <div style={{ padding: "90px 20px", textAlign: "center", color: "#d7e8f8", fontSize: 20 }}>
            Restoring your saved team data…
          </div>
        ) : !rows.length ? (
          <div
            style={{
              padding: "90px 20px",
              textAlign: "center",
              color: "#d7e8f8",
              fontSize: 20,
            }}
          >
            Upload a league team spreadsheet to build the dashboard.
          </div>
        ) : (
          <>
            <section style={{ marginTop: 28, display: "grid", gap: 16 }}>
              <div
                style={{
                  display: "flex",
                  gap: 16,
                  alignItems: "end",
                  flexWrap: "wrap",
                }}
              >
                <label style={{ display: "grid", gap: 8, fontWeight: 700 }}>
                  Team
                  <select
                    value={selectedTeam}
                    onChange={(event) => setSelectedTeam(event.target.value)}
                    style={{
                      minWidth: 280,
                      padding: 12,
                      borderRadius: 8,
                      fontSize: 16,
                    }}
                  >
                    {teams.map((team) => (
                      <option key={team}>{team}</option>
                    ))}
                  </select>
                </label>
                <label style={{ display: "grid", gap: 8, fontWeight: 700 }}>
                  Compare with
                  <select
                    value={comparisonTeam}
                    onChange={(event) => setComparisonTeam(event.target.value)}
                    style={{
                      minWidth: 280,
                      padding: 12,
                      borderRadius: 8,
                      fontSize: 16,
                    }}
                  >
                    {teams
                      .filter((team) => team !== selectedTeam)
                      .map((team) => (
                        <option key={team}>{team}</option>
                      ))}
                  </select>
                </label>
                <div style={{ color: "#d7e8f8", paddingBottom: 12 }}>
                  {teams.length} teams · {activeMetrics.length} of{" "}
                  {metrics.length} metrics selected
                </div>
                <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid rgba(255,255,255,.28)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 8 }}>
                    <strong>Comparison chart metrics</strong>
                    <button type="button" onClick={() => setComparisonMetrics(null)} style={{ padding: "5px 9px", borderRadius: 6, border: "1px solid #b9eaff", background: "#62dcff", color: "#063d63", fontWeight: 700 }}>Use all metrics</button>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 16px" }}>
                    {metrics.map((metric) => <label key={metric} style={{ display: "flex", gap: 6, alignItems: "center", color: "#fff", fontSize: 13 }}><input type="checkbox" checked={comparisonMetrics === null || comparisonMetrics.includes(metric)} onChange={() => setComparisonMetrics((current) => { const base = current === null ? [...metrics] : current; return base.includes(metric) ? base.filter((item) => item !== metric) : [...base, metric]; })} />{metric}</label>)}
                  </div>
                </div>
              </div>
              <div
                style={{
                  padding: 18,
                  borderRadius: 12,
                  background: "rgba(255,255,255,.12)",
                  border: "1px solid rgba(255,255,255,.28)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 12,
                    flexWrap: "wrap",
                    marginBottom: 12,
                  }}
                >
                  <strong>Select metrics to include</strong>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button
                      type="button"
                      onClick={() => setSelectedMetrics(null)}
                      style={{
                        padding: "6px 10px",
                        borderRadius: 6,
                        border: "1px solid #b9eaff",
                        background: "#62dcff",
                        color: "#063d63",
                        fontWeight: 700,
                      }}
                    >
                      Use all
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setSelectedMetrics(
                          metrics.filter(
                            (metric) => metricGroup(metric) === "In Possession",
                          ),
                        )
                      }
                      style={{
                        padding: "6px 10px",
                        borderRadius: 6,
                        border: "1px solid #b9eaff",
                        background: "transparent",
                        color: "#fff",
                        fontWeight: 700,
                      }}
                    >
                      In Possession only
                    </button>
                  </div>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3,minmax(0,1fr))",
                    gap: 16,
                  }}
                >
                  {["In Possession", "Out Of Possession", "Set-Pieces"].map(
                    (group) => (
                      <div key={group}>
                        <div
                          style={{
                            color: "#62dcff",
                            fontWeight: 800,
                            marginBottom: 8,
                          }}
                        >
                          {group}
                        </div>
                        <div style={{ display: "grid", gap: 6 }}>
                          {metrics
                            .filter((metric) => metricGroup(metric) === group)
                            .map((metric) => (
                              <label
                                key={metric}
                                style={{
                                  display: "flex",
                                  gap: 7,
                                  alignItems: "center",
                                  color: "#fff",
                                  fontSize: 13,
                                }}
                              >
                                <input
                                  type="checkbox"
                                  checked={
                                    selectedMetrics === null ||
                                    selectedMetrics.includes(metric)
                                  }
                                  onChange={() =>
                                    setSelectedMetrics((current) => {
                                      const base = current === null ? [...metrics] : current;
                                      return base.includes(metric)
                                        ? base.filter((item) => item !== metric)
                                        : [...base, metric];
                                    })
                                  }
                                />
                                {metric}
                              </label>
                            ))}
                        </div>
                      </div>
                    ),
                  )}
                </div>
              </div>
            </section>
            <div
              ref={dashboardRef}
              style={{
                background: "#062c63",
                padding: 18,
                borderRadius: 16,
                width: "100%",
                boxSizing: "border-box",
                display: "flex",
                flexDirection: "column",
              }}
            >
              <section
                style={{
                  marginTop: 28,
                  padding: "22px 28px",
                  borderRadius: 14,
                  background: "linear-gradient(110deg,#0b3c73,#155b91)",
                  border: "2px solid #78b4d8",
                  textAlign: "center",
                }}
              >
                <h2 style={{ margin: "0 0 7px", color: "#fff", fontSize: 26, whiteSpace: "nowrap", wordSpacing: "0.18em", display: "flex", justifyContent: "center", alignItems: "center", gap: 10 }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}><ClubBadge club={selectedTeamName} size={28} />{pdfTitle(selectedTeamName)}</span>
                  <span>{pdfTitle("Team Analysis")}</span>
                </h2>
                <div style={{ color: "#d2e5fa", fontSize: 14 }}>
                  {gamesKey
                    ? `Games Played: ${normalise(selected?.[gamesKey]) || "-"}`
                    : ""}{" "}
                  · {metrics.length} metrics available · All metrics per 90
                </div>
              </section>
              <section
                data-old-metric-section
                style={{
                  order: 2,
                  marginTop: 24,
                  background: "#173f70",
                  color: "#fff",
                  borderRadius: 14,
                  padding: "24px 28px",
                  border: "2px solid #6a96bd",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 16,
                    flexWrap: "wrap",
                    marginBottom: 22,
                  }}
                >
                  <h2 style={{ color: "#62dcff", margin: 0, wordSpacing: "0.2em" }}>
                    {pdfTitle("Metric Percentiles")}
                  </h2>
                  <div
                    style={{
                      display: "flex",
                      gap: 18,
                      flexWrap: "wrap",
                      fontSize: 12,
                      fontWeight: 700,
                    }}
                  >
                    <span>
                      <i
                        style={{
                          display: "inline-block",
                          width: 13,
                          height: 13,
                          borderRadius: 4,
                          background: "#15c77a",
                          marginRight: 6,
                          verticalAlign: "-2px",
                        }}
                      />
                      Top 25%
                    </span>
                    <span>
                      <i
                        style={{
                          display: "inline-block",
                          width: 13,
                          height: 13,
                          borderRadius: 4,
                          background: "#ff9f1a",
                          marginRight: 6,
                          verticalAlign: "-2px",
                        }}
                      />
                      50%
                    </span>
                    <span>
                      <i
                        style={{
                          display: "inline-block",
                          width: 13,
                          height: 13,
                          borderRadius: 4,
                          background: "#ff4740",
                          marginRight: 6,
                          verticalAlign: "-2px",
                        }}
                      />
                      Below 25%
                    </span>
                    <span>
                      <i
                        style={{
                          display: "inline-block",
                          width: 13,
                          height: 13,
                          borderRadius: 4,
                          background: "#ffd21c",
                          marginRight: 6,
                          verticalAlign: "-2px",
                        }}
                      />
                      League Average
                    </span>
                  </div>
                </div>
                <div style={{ columnCount: 2, columnGap: 30 }}>
                  {chartData.map((item) => (
                    <div
                      key={item.metric}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "150px minmax(0,1fr)",
                        alignItems: "center",
                        gap: 12,
                        marginBottom: 14,
                        breakInside: "avoid",
                      }}
                    >
                      <div>
                        <strong
                          style={{
                            display: "block",
                            fontSize: 14,
                            color: "#fff",
                          }}
                        >
                          {item.metric}
                        </strong>
                        <small style={{ color: "#d2e5fa" }}>
                          {item.teamValue}
                        </small>
                      </div>
                      <div>
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "flex-end",
                            fontSize: 12,
                            fontWeight: 800,
                            color: "#fff",
                            marginBottom: 4,
                          }}
                        >
                          <span>{item.percentile}%</span>
                        </div>
                        <div
                          style={{
                            height: 11,
                            borderRadius: 8,
                            background: "#365b80",
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              width: `${item.percentile}%`,
                              height: "100%",
                              borderRadius: 8,
                              background: scoreColour(item.percentile),
                            }}
                          />
                        </div>
                        <div
                          style={{
                            height: 11,
                            marginTop: 4,
                            borderRadius: 8,
                            background: "#284d73",
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              width: `${item.leagueAveragePct}%`,
                              height: "100%",
                              borderRadius: 8,
                              background: "#ffd21c",
                            }}
                          />
                          <span
                            style={{
                              position: "relative",
                              display: "block",
                              marginTop: -11,
                              textAlign: "center",
                              fontSize: 9,
                              fontWeight: 800,
                              color: "#111",
                            }}
                          >
                            {item.leagueAveragePct}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
              <section
                style={{
                  order: 1,
                  marginTop: 28,
                  background: "#173f70",
                  color: "#fff",
                  border: "2px solid #6a96bd",
                  borderRadius: 14,
                  padding: "24px 28px",
                }}
              >
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3,minmax(180px,1fr))",
                    gap: 30,
                    maxWidth: 760,
                    margin: "0 auto",
                  }}
                >
                  {["In Possession", "Out Of Possession", "Set-Pieces"].map(
                    (group) => {
                      const groupMetrics = chartData.filter(
                        (item) => metricGroup(item.metric) === group,
                      );
                      const score = groupMetrics.length
                        ? Math.round(
                            groupMetrics.reduce(
                              (sum, item) => sum + item.percentile,
                              0,
                            ) / groupMetrics.length,
                          )
                        : 0;
                      return (
                        <div key={group} style={{ textAlign: "center" }}>
                          <svg
                            viewBox="0 0 140 140"
                            width="150"
                            height="150"
                            role="img"
                            aria-label={`${group}: ${score}%`}
                          >
                            <circle
                              cx="70"
                              cy="70"
                              r="56"
                              fill="none"
                              stroke="#365b80"
                              strokeWidth="12"
                            />
                            <circle
                              cx="70"
                              cy="70"
                              r="56"
                              fill="none"
                              stroke={scoreColour(score)}
                              strokeWidth="12"
                              strokeLinecap="round"
                              strokeDasharray={`${score * 3.518} 351.8`}
                              transform="rotate(-90 70 70)"
                            />
                            <text
                              x="70"
                              y="78"
                              textAnchor="middle"
                              fontSize="27"
                              fontWeight="800"
                              fill="#fff"
                            >
                              {score}%
                            </text>
                          </svg>
                          <div
                            style={{
                              color: "#fff",
                              fontWeight: 800,
                              fontSize: 17,
                            }}
                          >
                            {group}
                          </div>
                          <div
                            style={{
                              color: "#d2e5fa",
                              fontSize: 12,
                              marginTop: 5,
                            }}
                          >
                            {groupMetrics.length} metrics combined
                          </div>
                        </div>
                      );
                    },
                  )}
                </div>
              </section>
              <section
                data-grouped-metrics
                style={{
                  order: 2.5,
                  marginTop: 24,
                  background: "#173f70",
                  color: "#fff",
                  border: "2px solid #6a96bd",
                  borderRadius: 14,
                  padding: "24px 28px",
                }}
              >
                <div
                  style={{
                    gridColumn: "1 / -1",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 16,
                    flexWrap: "wrap",
                    marginBottom: 8,
                  }}
                >
                  <h2 style={{ color: "#62dcff", margin: 0, wordSpacing: "0.2em" }}>
                    {pdfTitle("Metric Percentiles")}
                  </h2>
                  <div
                    style={{
                      display: "flex",
                      gap: 18,
                      flexWrap: "wrap",
                      fontSize: 12,
                      fontWeight: 700,
                    }}
                  >
                    <span>
                      <i
                        style={{
                          display: "inline-block",
                          width: 13,
                          height: 13,
                          borderRadius: 4,
                          background: "#15c77a",
                          marginRight: 6,
                          verticalAlign: "-2px",
                        }}
                      />
                      Top 25%
                    </span>
                    <span>
                      <i
                        style={{
                          display: "inline-block",
                          width: 13,
                          height: 13,
                          borderRadius: 4,
                          background: "#ff9f1a",
                          marginRight: 6,
                          verticalAlign: "-2px",
                        }}
                      />
                      50%
                    </span>
                    <span>
                      <i
                        style={{
                          display: "inline-block",
                          width: 13,
                          height: 13,
                          borderRadius: 4,
                          background: "#ff4740",
                          marginRight: 6,
                          verticalAlign: "-2px",
                        }}
                      />
                      Below 25%
                    </span>
                    <span>
                      <i
                        style={{
                          display: "inline-block",
                          width: 13,
                          height: 13,
                          borderRadius: 4,
                          background: "#ffd21c",
                          marginRight: 6,
                          verticalAlign: "-2px",
                        }}
                      />
                      League Average
                    </span>
                  </div>
                </div>
                {metricColumns.map((group) => (
                  <div key={group}>
                    <h3
                      style={{
                        color: "#fff",
                        fontSize: 18,
                        margin: "0 0 16px",
                        borderBottom: "1px solid #6a96bd",
                        paddingBottom: 8,
                      }}
                    >
                      {group}
                    </h3>
                    {chartData
                      .filter((item) => metricGroup(item.metric) === group)
                      .map(renderMetric)}
                  </div>
                ))}
              </section>
              <ChartBoundary>
                <section
                  style={{
                    order: 3,
                    marginTop: 24,
                    background: "#173f70",
                    color: "#fff",
                    border: "2px solid #6a96bd",
                    borderRadius: 14,
                    padding: "20px 28px",
                  }}
                >
                  <h2
                    style={{
                      color: "#62dcff",
                      margin: "0 0 4px",
                      textAlign: "center",
                      wordSpacing: "0.2em",
                    }}
                  >
                    {pdfTitle("Team Visual Comparison")}
                  </h2>
                  <p
                    style={{
                      color: "#d2e5fa",
                      marginTop: 0,
                      textAlign: "center",
                    }}
                  >
                    Compare the selected teams across the uploaded metrics.
                  </p>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(2,minmax(0,1fr))",
                      gap: 24,
                      alignItems: "stretch",
                    }}
                  >
                    <section
                      style={{
                        background: "#123761",
                        borderRadius: 10,
                        padding: 16,
                      }}
                    >
                      <h3
                        style={{
                          margin: "0 0 4px",
                          color: "#fff",
                          fontSize: 18,
                          textAlign: "center",
                        }}
                      >
                        Team Style
                      </h3>
                      <p
                        style={{
                          margin: "0 0 8px",
                          color: "#d2e5fa",
                          fontSize: 12,
                          textAlign: "center",
                        }}
                      >
                        Compared with league average
                      </p>
                      <ResponsiveContainer width="100%" height={300}>
                        <RadarChart
                          data={styleData}
                          cx="50%"
                          cy="50%"
                          outerRadius="68%"
                        >
                          <PolarGrid stroke="#6a96bd" />
                          <PolarAngleAxis
                            dataKey="style"
                            tick={{ fontSize: 12, fill: "#d2e5fa" }}
                          />
                          <Radar
                            name="League Average"
                            dataKey="leagueAverage"
                            stroke="#ffd21c"
                            fill="#ffd21c"
                            fillOpacity={0.28}
                          />
                          <Radar
                            name={normalise(selected?.[teamKey]) || "Team"}
                            dataKey="score"
                            stroke="#62dcff"
                            fill="#62dcff"
                            fillOpacity={0.5}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "center",
                          gap: 16,
                          fontSize: 12,
                          fontWeight: 700,
                        }}
                      >
                        <span>
                          <i
                            style={{
                              display: "inline-block",
                              width: 11,
                              height: 11,
                              borderRadius: 3,
                              background: "#62dcff",
                              marginRight: 5,
                            }}
                          />
                          {selectedTeamName}
                        </span>
                        <span>
                          <i
                            style={{
                              display: "inline-block",
                              width: 11,
                              height: 11,
                              borderRadius: 3,
                              background: "#ffd21c",
                              marginRight: 5,
                            }}
                          />
                          League Average
                        </span>
                      </div>
                    </section>
                    <section
                      style={{
                        background: "#123761",
                        borderRadius: 10,
                        padding: 16,
                      }}
                    >
                      <h3
                        style={{
                          margin: "0 0 4px",
                          color: "#fff",
                          fontSize: 18,
                          textAlign: "center",
                        }}
                      >
                        Team Difference
                      </h3>
                      <p
                        style={{
                          margin: "0 0 14px",
                          color: "#d2e5fa",
                          fontSize: 12,
                          textAlign: "center",
                        }}
                      >
                        <ClubBadge club={selectedTeamName} size={18} /> {selectedTeamName} vs{" "}
                        <ClubBadge club={comparisonTeamName} size={18} /> {comparisonTeamName || "second team"}
                      </p>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-around",
                          color: "#fff",
                          fontSize: 11,
                          marginBottom: 10,
                        }}
                      >
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}><ClubBadge club={selectedTeamName} size={15} />◀ {selectedTeamName}</span>
                        <span style={{ color: "#ffd21c" }}>
                          <ClubBadge club={comparisonTeamName} size={15} /> {comparisonTeamName || "Comparison"} ▶
                        </span>
                      </div>
                      {comparisonData.slice(0, 10).map((item) => (
                        <div key={item.metric} style={{ marginBottom: 10 }}>
                          <div
                            style={{
                              display: "grid",
                              gridTemplateColumns: "1fr 1.5fr 1fr",
                              gap: 6,
                              alignItems: "center",
                              fontSize: 10,
                              color: "#fff",
                            }}
                          >
                            <span style={{ textAlign: "left" }}>
                              {item.first}
                            </span>
                            <strong
                              style={{ textAlign: "center", fontSize: 10 }}
                            >
                              {item.metric}
                            </strong>
                            <span style={{ textAlign: "right" }}>
                              {item.second}
                            </span>
                          </div>
                          <div
                            style={{
                              display: "grid",
                              gridTemplateColumns: "1fr 2px 1fr",
                              gap: 3,
                              alignItems: "center",
                              marginTop: 3,
                            }}
                          >
                            <div
                              style={{
                                height: 9,
                                background: "#365b80",
                                borderRadius: "6px 0 0 6px",
                                overflow: "hidden",
                                display: "flex",
                                justifyContent: "flex-end",
                              }}
                            >
                              <div
                                style={{
                                  width: `${item.firstPct}%`,
                                  height: "100%",
                                  background: "#62dcff",
                                }}
                              />
                            </div>
                            <div style={{ height: 13, background: "#fff" }} />
                            <div
                              style={{
                                height: 9,
                                background: "#284d73",
                                borderRadius: "0 6px 6px 0",
                                overflow: "hidden",
                              }}
                            >
                              <div
                                style={{
                                  width: `${item.secondPct}%`,
                                  height: "100%",
                                  background: "#ffd21c",
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </section>
                  </div>
                </section>
              </ChartBoundary>
              <section
                style={{
                  order: 4,
                  marginTop: 24,
                  background: "#173f70",
                  color: "#fff",
                  border: "2px solid #6a96bd",
                  borderRadius: 14,
                  padding: "20px 28px",
                }}
              >
                <h2 style={{ color: "#62dcff", margin: "0 0 16px", wordSpacing: "0.2em" }}>
                  {pdfTitle("Team Rankings")}
                </h2>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(2,minmax(0,1fr))",
                    gap: "10px 28px",
                  }}
                >
                  {activeMetrics.map((metric) => {
                    const value = number(selected?.[metric]);
                    const ranked = rows
                      .map((row) => number(row[metric]))
                      .filter((item) => item !== null)
                      .sort((a, b) => (lowerIsBetter(metric) ? a - b : b - a));
                    const rank =
                      value === null
                        ? null
                        : ranked.findIndex((item) => item === value) + 1;
                    return (
                      <div
                        key={metric}
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          gap: 12,
                          borderBottom: "1px solid #365b80",
                          paddingBottom: 7,
                        }}
                      >
                        <span>{metric}</span>
                        <strong>
                          {rankLabel(rank)}
                        </strong>
                      </div>
                    );
                  })}
                </div>
              </section>
              <section
                style={{
                  order: 5,
                  marginTop: 24,
                  background: "#173f70",
                  color: "#fff",
                  border: "2px solid #6a96bd",
                  borderRadius: 14,
                  padding: "20px 28px",
                }}
              >
                <h2 style={{ color: "#62dcff", margin: "0 0 16px", wordSpacing: "0.2em" }}>
                  {pdfTitle("Team Rankings")}
                </h2>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3,minmax(0,1fr))",
                    gap: 24,
                  }}
                >
                  {metricColumns.map((group) => (
                    <div key={group}>
                      <h3
                        style={{
                          margin: "0 0 10px",
                          color: "#fff",
                          fontSize: 17,
                          borderBottom: "1px solid #6a96bd",
                          paddingBottom: 8,
                        }}
                      >
                        {group}
                      </h3>
                      {activeMetrics
                        .filter((metric) => metricGroup(metric) === group)
                        .map((metric) => {
                          const value = number(selected?.[metric]);
                          const ranked = teamRows
                            .map((row) => number(row[metric]))
                            .filter((item) => item !== null)
                            .sort((a, b) =>
                              lowerIsBetter(metric) ? a - b : b - a,
                            );
                          const rank =
                            value === null
                              ? null
                              : ranked.findIndex((item) => item === value) + 1;
                          return (
                            <div
                              key={metric}
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                gap: 8,
                                borderBottom: "1px solid #365b80",
                                padding: "7px 0",
                                fontSize: 13,
                              }}
                            >
                              <span>{metric}</span>
                              <strong
                                style={{
                                  color: rank ? rankColour(rank) : "#fff",
                                  whiteSpace: "nowrap",
                                }}
                              >
                                {rankLabel(rank)}
                              </strong>
                            </div>
                          );
                        })}
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
