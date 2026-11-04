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
const HALF_PITCH_VIEW_HEIGHT = 580;
const DEFAULT_TACTICAL_AWAY = [
  [910, 320], [755, 90], [755, 245], [755, 395], [755, 550],
  [580, 170], [580, 320], [580, 470], [390, 145], [390, 320], [390, 495],
].map(([x, y], index) => ({ id: `away-${index + 1}`, side: "away", number: index + 1, x, y }));
const freshTacticalBoard = (homeName = "Home Team", awayName = "Opposition") => ({
  name: "New Tactic",
  homeName: homeName || "Home Team",
  awayName: awayName || "Opposition",
  players: [...DEFAULT_TACTICAL_HOME, ...DEFAULT_TACTICAL_AWAY],
  ball: { x: 500, y: 320 },
  arrows: [],
  shapes: [],
  pitchMode: "full",
});

const pathLength = (points = []) => points.slice(1).reduce((total, point, index) => total + Math.hypot(point.x - points[index].x, point.y - points[index].y), 0);
const tacticalClipDuration = (clip) => Math.max(3400, Math.min(9000, (pathLength(clip.ballPath || []) / 0.18) || 3400));
const pointAlongPath = (points, progress, start, end) => {
  const path = points?.length > 1 ? points : [start, end];
  const lengths = path.slice(1).map((point, index) => Math.hypot(point.x - path[index].x, point.y - path[index].y));
  const totalLength = lengths.reduce((sum, length) => sum + length, 0);
  if (totalLength < 0.001) return { x: start.x + (end.x - start.x) * progress, y: start.y + (end.y - start.y) * progress };
  let remaining = totalLength * Math.max(0, Math.min(1, progress));
  for (let index = 0; index < lengths.length; index += 1) {
    const segmentLength = lengths[index];
    if (remaining <= segmentLength || index === lengths.length - 1) {
      const fraction = segmentLength < 0.001 ? 1 : Math.max(0, Math.min(1, remaining / segmentLength));
      return {
        x: path[index].x + (path[index + 1].x - path[index].x) * fraction,
        y: path[index].y + (path[index + 1].y - path[index].y) * fraction,
      };
    }
    remaining -= segmentLength;
  }
  return { ...end };
};
const interpolateTacticalShapes = (fromShapes = [], toShapes = [], progress = 0) => {
  const fromById = new Map(fromShapes.map((shape) => [shape.id, shape]));
  const toById = new Map(toShapes.map((shape) => [shape.id, shape]));
  const ids = new Set([...fromById.keys(), ...toById.keys()]);
  return [...ids].map((id) => {
    const from = fromById.get(id);
    const to = toById.get(id);
    if (from && to) {
      return {
        ...to,
        x: from.x + (to.x - from.x) * progress,
        y: from.y + (to.y - from.y) * progress,
        width: from.width + (to.width - from.width) * progress,
        height: from.height + (to.height - from.height) * progress,
        fontSize: (from.fontSize || 22) + ((to.fontSize || 22) - (from.fontSize || 22)) * progress,
        opacity: (from.opacity ?? (from.type === "text" ? 0.85 : 0.35)) + ((to.opacity ?? (to.type === "text" ? 0.85 : 0.35)) - (from.opacity ?? (from.type === "text" ? 0.85 : 0.35))) * progress,
        _playbackOpacity: 1,
      };
    }
    const shape = to || from;
    const appearing = Boolean(to);
    const amount = appearing ? progress : 1 - progress;
    const width = Math.max(0.01, shape.width * amount);
    const height = Math.max(0.01, shape.height * amount);
    return {
      ...shape,
      x: shape.x + (shape.width - width) / 2,
      y: shape.y + (shape.height - height) / 2,
      width,
      height,
      _playbackOpacity: amount,
    };
  });
};
const interpolateTacticalArrows = (fromArrows = [], toArrows = [], progress = 0) => {
  const fromById = new Map(fromArrows.map((arrow) => [arrow.id, arrow]));
  const toById = new Map(toArrows.map((arrow) => [arrow.id, arrow]));
  const ids = new Set([...fromById.keys(), ...toById.keys()]);
  return [...ids].map((id) => {
    const from = fromById.get(id);
    const to = toById.get(id);
    if (from && to) {
      return {
        ...to,
        from: { x: from.from.x + (to.from.x - from.from.x) * progress, y: from.from.y + (to.from.y - from.from.y) * progress },
        to: { x: from.to.x + (to.to.x - from.to.x) * progress, y: from.to.y + (to.to.y - from.to.y) * progress },
        opacity: (from.opacity ?? 1) + ((to.opacity ?? 1) - (from.opacity ?? 1)) * progress,
        strokeWidth: (from.strokeWidth || 6) + ((to.strokeWidth || 6) - (from.strokeWidth || 6)) * progress,
      };
    }
    const arrow = to || from;
    const amount = to ? progress : 1 - progress;
    return {
      ...arrow,
      to: { x: arrow.from.x + (arrow.to.x - arrow.from.x) * amount, y: arrow.from.y + (arrow.to.y - arrow.from.y) * amount },
      opacity: (arrow.opacity ?? 1) * amount,
    };
  });
};
const resizeTacticalShape = (shape, corner, delta) => {
  const minSize = 24;
  let left = shape.x;
  let top = shape.y;
  let right = shape.x + shape.width;
  let bottom = shape.y + shape.height;
  if (corner.includes("w")) left = Math.max(0, Math.min(right - minSize, left + delta.x));
  else right = Math.min(1000, Math.max(left + minSize, right + delta.x));
  if (corner.includes("n")) top = Math.max(0, Math.min(bottom - minSize, top + delta.y));
  else bottom = Math.min(640, Math.max(top + minSize, bottom + delta.y));
  if (["square", "circle"].includes(shape.type)) {
    const side = Math.max(minSize, Math.max(right - left, bottom - top));
    left = corner.includes("w") ? Math.max(0, shape.x + shape.width - side) : shape.x;
    top = corner.includes("n") ? Math.max(0, shape.y + shape.height - side) : shape.y;
    const boundedSide = Math.min(side, 1000 - left, 640 - top);
    right = left + boundedSide;
    bottom = top + boundedSide;
  }
  return { x: left, y: top, width: right - left, height: bottom - top };
};
const resizePortraitTextBox = (shape, corner, delta) => {
  const scaleX = 750 / 640;
  const screenDeltaX = delta.y * scaleX;
  const screenDeltaY = -delta.x;
  const growsEast = corner.endsWith("e");
  const growsSouth = corner.endsWith("s");
  const width = Math.max(40, Math.min(1000, shape.width + (growsEast ? screenDeltaX : -screenDeltaX) / scaleX));
  const height = Math.max(30, Math.min(640, shape.height + (growsSouth ? screenDeltaY : -screenDeltaY)));
  const centerViewX = scaleX * (shape.y + shape.height / 2) + screenDeltaX / 2;
  const centerViewY = 1000 - (shape.x + shape.width / 2) + screenDeltaY / 2;
  const centerY = centerViewX / scaleX;
  const centerX = 1000 - centerViewY;
  return {
    x: Math.max(0, Math.min(1000 - width, centerX - width / 2)),
    y: Math.max(0, Math.min(640 - height, centerY - height / 2)),
    width,
    height,
  };
};

function TacticalToolIcon({ type }) {
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" };
  return <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" {...common}>
    {type === "move" && <><path d="M5 3 19 13l-6 .7-3.2 5.3L5 3Z" /><path d="m13 14 3 6" /></>}
    {type === "multiple" && <><circle cx="7" cy="7" r="3" /><circle cx="17" cy="7" r="3" /><circle cx="7" cy="17" r="3" /><circle cx="17" cy="17" r="3" /></>}
    {type === "arrow" && <><path d="M4 18 18 6" /><path d="M11 6h7v7" /></>}
    {type === "line" && <path d="M4 19 20 5" />}
    {type === "rectangle" && <rect x="3" y="6" width="18" height="12" rx="1" />}
    {type === "square" && <rect x="5" y="5" width="14" height="14" rx="1" />}
    {type === "circle" && <circle cx="12" cy="12" r="8" />}
    {type === "triangle" && <path d="m12 4 9 16H3L12 4Z" />}
    {type === "text" && <><path d="M4 6h16M12 6v13M8 19h8" /><path d="M7 6h10" /></>}
  </svg>;
}

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
  (board.shapes || []).forEach((shape) => {
    const opacity = Number(shape.opacity);
    const baseOpacity = Math.max(0, Math.min(1, Number.isFinite(opacity) ? opacity : 0.35));
    context.globalAlpha = baseOpacity * (shape._playbackOpacity ?? 1);
    context.fillStyle = shape.type === "text" ? shape.backgroundColor || "#0c3769" : shape.color || "#62dcff";
    context.strokeStyle = shape.color || "#62dcff";
    context.lineWidth = Number(shape.strokeWidth) || 3;
    if (shape.type === "line") {
      context.beginPath(); context.moveTo(shape.x, shape.y); context.lineTo(shape.x + shape.width, shape.y + shape.height); context.stroke();
    } else if (shape.type === "circle") {
      context.beginPath(); context.ellipse(shape.x + shape.width / 2, shape.y + shape.height / 2, shape.width / 2, shape.height / 2, 0, 0, Math.PI * 2); context.fill(); context.stroke();
    } else if (shape.type === "triangle") {
      context.beginPath(); context.moveTo(shape.x + shape.width / 2, shape.y); context.lineTo(shape.x + shape.width, shape.y + shape.height); context.lineTo(shape.x, shape.y + shape.height); context.closePath(); context.fill(); context.stroke();
    } else {
      context.fillRect(shape.x, shape.y, shape.width, shape.height);
      context.strokeRect(shape.x, shape.y, shape.width, shape.height);
      if (shape.type === "text") {
        context.globalAlpha = shape._playbackOpacity ?? 1;
        context.fillStyle = shape.textColor || "#fff";
        context.font = `bold ${Number(shape.fontSize) || 22}px Arial, sans-serif`;
        context.textAlign = "left";
        context.textBaseline = "middle";
        context.fillText(shape.text || "Text", shape.x + 10, shape.y + shape.height / 2, Math.max(20, shape.width - 20));
      }
    }
  });
  context.globalAlpha = 1;
  context.fillStyle = "#fff";
  context.font = "bold 24px Arial, sans-serif";
  context.textAlign = "left";
  context.fillText(board.homeName || "Home Team", 42, 76);
  context.textAlign = "right";
  context.fillText(board.awayName || "Opposition", 958, 76);
  (board.arrows || []).forEach((arrow) => {
    const angle = Math.atan2(arrow.to.y - arrow.from.y, arrow.to.x - arrow.from.x);
    context.setLineDash(arrow.lineStyle === "dashed" ? [14, 9] : arrow.lineStyle === "dotted" ? [2, 8] : []);
    context.globalAlpha = arrow.opacity ?? 1;
    context.strokeStyle = arrow.color || "#ffe36e"; context.fillStyle = arrow.color || "#ffe36e"; context.lineWidth = Number(arrow.strokeWidth) || 6; context.lineCap = "round";
    context.beginPath(); context.moveTo(arrow.from.x, arrow.from.y); context.lineTo(arrow.to.x, arrow.to.y); context.stroke();
    context.beginPath(); context.moveTo(arrow.to.x, arrow.to.y);
    context.lineTo(arrow.to.x - 20 * Math.cos(angle - Math.PI / 6), arrow.to.y - 20 * Math.sin(angle - Math.PI / 6));
    context.lineTo(arrow.to.x - 20 * Math.cos(angle + Math.PI / 6), arrow.to.y - 20 * Math.sin(angle + Math.PI / 6));
    context.closePath(); context.fill();
    context.globalAlpha = 1;
  });
  context.setLineDash([]);
  board.players.forEach((player) => {
    context.beginPath(); context.arc(player.x, player.y + 3, 27, 0, Math.PI * 2); context.fillStyle = "#05223f70"; context.fill();
    context.beginPath(); context.arc(player.x, player.y, 24, 0, Math.PI * 2); context.fillStyle = player.side === "home" ? "#1485dd" : "#e64b50";
    context.shadowColor = player.moving ? "#ffe36e" : "transparent"; context.shadowBlur = player.moving ? 22 : 0; context.fill(); context.shadowBlur = 0;
    context.strokeStyle = player.moving ? "#ffe36e" : "#f3f8ff"; context.lineWidth = player.moving ? 5 : 3; context.stroke();
    context.fillStyle = "#fff"; context.font = "bold 17px Arial, sans-serif"; context.textAlign = "center"; context.textBaseline = "middle";
    context.fillText(String(player.number), player.x, player.y + 1);
  });
  const ball = board.ball || { x: 500, y: 320 };
  context.beginPath(); context.arc(ball.x, ball.y + 2, 11, 0, Math.PI * 2); context.fillStyle = "#05223f70"; context.fill();
  context.beginPath(); context.arc(ball.x, ball.y, 10, 0, Math.PI * 2); context.fillStyle = "#f7f8f3"; context.fill();
  context.strokeStyle = "#142234"; context.lineWidth = 2; context.stroke();
  context.fillStyle = "#142234"; context.beginPath(); context.moveTo(ball.x, ball.y - 5); context.lineTo(ball.x + 4, ball.y - 2); context.lineTo(ball.x + 3, ball.y + 3); context.lineTo(ball.x - 3, ball.y + 3); context.lineTo(ball.x - 4, ball.y - 2); context.closePath(); context.fill();
  context.restore();
};

function TacticalPad({ storageKey, homeTeamName, oppositionName }) {
  const clipStorageKey = `${storageKey}:movement-clips`;
  const [board, setBoard] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
      return saved?.players?.length ? { ...freshTacticalBoard(homeTeamName, oppositionName), ...saved } : freshTacticalBoard(homeTeamName, oppositionName);
    } catch { return freshTacticalBoard(homeTeamName, oppositionName); }
  });
  const [mode, setMode] = useState("move");
  const [draggingIds, setDraggingIds] = useState([]);
  const [draggingBall, setDraggingBall] = useState(false);
  const [draggingShapeId, setDraggingShapeId] = useState(null);
  const [draggingArrowId, setDraggingArrowId] = useState(null);
  const [resizingShapeId, setResizingShapeId] = useState(null);
  const [resizingArrowEnd, setResizingArrowEnd] = useState(null);
  const [multiSelectMode, setMultiSelectMode] = useState(false);
  const [arrowStart, setArrowStart] = useState(null);
  const [arrowPreview, setArrowPreview] = useState(null);
  const [selectedTokens, setSelectedTokens] = useState([]);
  const [selectedShapeId, setSelectedShapeId] = useState(null);
  const [editingTextId, setEditingTextId] = useState(null);
  const [copiedTextBox, setCopiedTextBox] = useState(null);
  const [selectedArrowId, setSelectedArrowId] = useState(null);
  const [enteringGraphicIds, setEnteringGraphicIds] = useState([]);
  const [newShapeOpacity, setNewShapeOpacity] = useState(0.35);
  const [newShapeColour, setNewShapeColour] = useState("#62dcff");
  const [newShapeStrokeWidth, setNewShapeStrokeWidth] = useState(4);
  const [newArrowLineStyle, setNewArrowLineStyle] = useState("solid");
  const [newTextColor, setNewTextColor] = useState("#ffffff");
  const [newTextBoxColor, setNewTextBoxColor] = useState("#0c3769");
  const [dockPanel, setDockPanel] = useState("shapes");
  const [movementClips, setMovementClips] = useState(() => {
    try {
      const savedClips = JSON.parse(localStorage.getItem(clipStorageKey) || "[]");
      return Array.isArray(savedClips) ? savedClips : [];
    } catch { return []; }
  });
  const movementClipsRef = useRef(movementClips);
  const [pitchMode, setPitchMode] = useState(() => board.pitchMode || "full");
  const [activeClipStart, setActiveClipStart] = useState(null);
  const [videoRecording, setVideoRecording] = useState(false);
  const [finishingVideo, setFinishingVideo] = useState(false);
  const [videoStatus, setVideoStatus] = useState("");
  const [playbackBoard, setPlaybackBoard] = useState(null);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const pitchRef = useRef(null);
  const videoCanvasRef = useRef(null);
  const boardRef = useRef(board);
  const activeClipStartRef = useRef(null);
  const activeClipBallPathRef = useRef([]);
  const dragStartRef = useRef(null);
  const dragTargetsRef = useRef({});
  const ballDragTargetRef = useRef(null);
  const videoRecorderRef = useRef(null);
  const videoChunksRef = useRef([]);
  const videoFrameRef = useRef(null);
  const videoFinishTimerRef = useRef(null);
  const clipPreviewFrameRef = useRef(null);
  const loadedClipStorageKeyRef = useRef(clipStorageKey);

  const persistMovementClips = (nextOrUpdater) => {
    const next = typeof nextOrUpdater === "function" ? nextOrUpdater(movementClipsRef.current) : nextOrUpdater;
    movementClipsRef.current = next;
    setMovementClips(next);
    try { localStorage.setItem(clipStorageKey, JSON.stringify(next)); } catch {}
    writeTeamAnalysisUpload(clipStorageKey, next).catch(() => {});
  };

  const markGraphicEntering = (id) => setEnteringGraphicIds((current) => [...current, id]);
  const finishGraphicEntering = (id) => setEnteringGraphicIds((current) => current.filter((item) => item !== id));

  useEffect(() => { boardRef.current = board; }, [board]);

  useEffect(() => {
    const saveTimer = window.setTimeout(() => {
      try { localStorage.setItem(storageKey, JSON.stringify(board)); } catch {}
    }, 250);
    return () => window.clearTimeout(saveTimer);
  }, [board, storageKey]);
  useEffect(() => {
    try { localStorage.setItem(clipStorageKey, JSON.stringify(movementClips)); } catch {}
  }, [movementClips, clipStorageKey]);
  useEffect(() => {
    let cancelled = false;
    if (loadedClipStorageKeyRef.current !== clipStorageKey) {
      loadedClipStorageKeyRef.current = clipStorageKey;
      let clipsForThisUser = [];
      try {
        const stored = JSON.parse(localStorage.getItem(clipStorageKey) || "[]");
        if (Array.isArray(stored)) clipsForThisUser = stored;
      } catch {}
      movementClipsRef.current = clipsForThisUser;
      setMovementClips(clipsForThisUser);
    }
    readTeamAnalysisUpload(clipStorageKey).then((savedClips) => {
      if (cancelled || movementClipsRef.current.length || !Array.isArray(savedClips) || !savedClips.length) return;
      movementClipsRef.current = savedClips;
      setMovementClips(savedClips);
    }).catch(() => {});
    const flushClips = () => {
      try { localStorage.setItem(clipStorageKey, JSON.stringify(movementClipsRef.current)); } catch {}
    };
    window.addEventListener("pagehide", flushClips);
    return () => { cancelled = true; flushClips(); window.removeEventListener("pagehide", flushClips); };
  }, [clipStorageKey]);
  useEffect(() => () => {
    if (videoFrameRef.current) window.cancelAnimationFrame(videoFrameRef.current);
    if (clipPreviewFrameRef.current) window.cancelAnimationFrame(clipPreviewFrameRef.current);
    if (videoFinishTimerRef.current) window.clearTimeout(videoFinishTimerRef.current);
    const recorder = videoRecorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);
  useEffect(() => () => { if (recordedVideo?.url) URL.revokeObjectURL(recordedVideo.url); }, [recordedVideo]);

  const pointFromEvent = (event) => {
    const rect = pitchRef.current?.getBoundingClientRect();
    if (!rect) return { x: 500, y: 320 };
    const activePitchMode = videoPlaying ? (playbackBoard?.pitchMode || pitchMode) : pitchMode;
    const viewX = ((event.clientX - rect.left) / rect.width) * (activePitchMode === "full" ? 1000 : 750);
    const viewY = ((event.clientY - rect.top) / rect.height) * (activePitchMode === "full" ? 640 : activePitchMode === "half" ? HALF_PITCH_VIEW_HEIGHT : 1000);
    return activePitchMode === "full"
      ? { x: Math.max(24, Math.min(976, viewX)), y: Math.max(24, Math.min(616, viewY)) }
      : { x: Math.max(24, Math.min(976, 1000 - viewY)), y: Math.max(24, Math.min(616, viewX * (640 / 750))) };
  };
  const shapeForDrag = (start, end, type) => {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    if (["square", "circle"].includes(type)) {
      const side = Math.max(Math.abs(dx), Math.abs(dy));
      dx = (dx < 0 ? -1 : 1) * side;
      dy = (dy < 0 ? -1 : 1) * side;
    }
    return { x: Math.min(start.x, start.x + dx), y: Math.min(start.y, start.y + dy), width: Math.abs(dx), height: Math.abs(dy) };
  };
  const textBoxForDrag = (start, end) => {
    if (pitchMode === "full") return shapeForDrag(start, end, "rectangle");
    const centerX = (start.x + end.x) / 2;
    const centerY = (start.y + end.y) / 2;
    const width = Math.abs(end.y - start.y);
    const height = Math.abs(end.x - start.x);
    return { x: centerX - width / 2, y: centerY - height / 2, width, height };
  };
  const beginPlayerDrag = (event, player) => {
    if (mode !== "move" || finishingVideo) return;
    event.stopPropagation();
    if (multiSelectMode) {
      setSelectedTokens((current) => current.includes(player.id) ? current.filter((id) => id !== player.id) : [...current, player.id]);
      return;
    }
    pitchRef.current?.setPointerCapture?.(event.pointerId);
    const moving = selectedTokens.includes(player.id) && selectedTokens.length > 1 ? selectedTokens : [player.id];
    setSelectedTokens(moving);
    setDraggingIds(moving);
    const point = pointFromEvent(event);
    const positions = Object.fromEntries(boardRef.current.players.filter((item) => moving.includes(item.id)).map((item) => [item.id, { x: item.x, y: item.y }]));
    dragStartRef.current = { pointer: point, positions };
  };
  const beginBallDrag = (event) => {
    if (mode !== "move" || multiSelectMode || finishingVideo) return;
    event.stopPropagation();
    pitchRef.current?.setPointerCapture?.(event.pointerId);
    const point = pointFromEvent(event);
    const ball = boardRef.current.ball || { x: 500, y: 320 };
    dragStartRef.current = { pointer: point, positions: { "tactical-ball": { x: ball.x, y: ball.y } } };
    setDraggingBall(true);
  };
  const beginShapeDrag = (event, shape) => {
    if (mode !== "move" || finishingVideo) return;
    event.stopPropagation();
    pitchRef.current?.setPointerCapture?.(event.pointerId);
    const point = pointFromEvent(event);
    dragStartRef.current = { pointer: point, positions: { [shape.id]: { x: shape.x, y: shape.y } } };
    setDraggingShapeId(shape.id);
    setEditingTextId(null);
    setSelectedShapeId(shape.id);
    setSelectedArrowId(null);
  };
  const beginShapeResize = (event, shape, corner) => {
    if (mode !== "move" || finishingVideo) return;
    event.stopPropagation();
    pitchRef.current?.setPointerCapture?.(event.pointerId);
    const point = pointFromEvent(event);
    dragStartRef.current = { pointer: point, resizeShape: { ...shape }, resizeCorner: corner };
    setResizingShapeId(shape.id);
    setEditingTextId(null);
    setSelectedShapeId(shape.id);
    setSelectedArrowId(null);
  };
  const beginArrowEndResize = (event, arrow, endpoint) => {
    if (mode !== "move" || finishingVideo) return;
    event.stopPropagation();
    pitchRef.current?.setPointerCapture?.(event.pointerId);
    const point = pointFromEvent(event);
    dragStartRef.current = { pointer: point, resizeArrow: { ...arrow, from: { ...arrow.from }, to: { ...arrow.to } }, resizeArrowEndpoint: endpoint };
    setResizingArrowEnd({ id: arrow.id, endpoint });
    setSelectedArrowId(arrow.id);
    setSelectedShapeId(null);
  };
  const beginArrowDrag = (event, arrow) => {
    if (mode !== "move" || finishingVideo) return;
    event.stopPropagation();
    pitchRef.current?.setPointerCapture?.(event.pointerId);
    const point = pointFromEvent(event);
    dragStartRef.current = { pointer: point, dragArrow: { ...arrow, from: { ...arrow.from }, to: { ...arrow.to } } };
    setDraggingArrowId(arrow.id);
    setSelectedArrowId(arrow.id);
    setSelectedShapeId(null);
  };
  const handlePitchPointerDown = (event) => {
    if (!(["arrow", "rectangle", "square", "circle", "triangle", "line", "text"].includes(mode)) || finishingVideo) return;
    pitchRef.current?.setPointerCapture?.(event.pointerId);
    const point = pointFromEvent(event);
    setArrowStart(point);
    setArrowPreview(point);
      setSelectedShapeId(null);
      setSelectedArrowId(null);
  };
  const handlePitchPointerMove = (event) => {
    const point = pointFromEvent(event);
    if (draggingIds.length && dragStartRef.current) {
      const delta = { x: point.x - dragStartRef.current.pointer.x, y: point.y - dragStartRef.current.pointer.y };
      const targets = Object.fromEntries(draggingIds.map((id) => [id, {
        x: Math.max(24, Math.min(976, dragStartRef.current.positions[id].x + delta.x)),
        y: Math.max(24, Math.min(616, dragStartRef.current.positions[id].y + delta.y)),
      }]));
      dragTargetsRef.current = targets;
      setBoard((current) => ({ ...current, players: current.players.map((player) => targets[player.id] ? { ...player, ...targets[player.id] } : player) }));
    }
    if (draggingBall && dragStartRef.current) {
      const origin = dragStartRef.current.positions["tactical-ball"];
      const delta = { x: point.x - dragStartRef.current.pointer.x, y: point.y - dragStartRef.current.pointer.y };
      const target = { x: Math.max(24, Math.min(976, origin.x + delta.x)), y: Math.max(24, Math.min(616, origin.y + delta.y)) };
      ballDragTargetRef.current = target;
      if (activeClipStartRef.current) {
        const path = activeClipBallPathRef.current;
        const last = path[path.length - 1];
        if (path.length < 600 && (!last || Math.hypot(target.x - last.x, target.y - last.y) >= 4)) path.push({ ...target });
      }
      setBoard((current) => ({ ...current, ball: target }));
    }
    if (draggingShapeId && dragStartRef.current) {
      const origin = dragStartRef.current.positions[draggingShapeId];
      const delta = { x: point.x - dragStartRef.current.pointer.x, y: point.y - dragStartRef.current.pointer.y };
      const target = { x: Math.max(0, Math.min(1000, origin.x + delta.x)), y: Math.max(0, Math.min(640, origin.y + delta.y)) };
      setBoard((current) => ({ ...current, shapes: (current.shapes || []).map((shape) => shape.id === draggingShapeId ? { ...shape, ...target } : shape) }));
    }
    if (draggingArrowId && dragStartRef.current?.dragArrow) {
      const start = dragStartRef.current;
      const deltaX = point.x - start.pointer.x;
      const deltaY = point.y - start.pointer.y;
      const minDeltaX = -Math.min(start.dragArrow.from.x, start.dragArrow.to.x);
      const maxDeltaX = 1000 - Math.max(start.dragArrow.from.x, start.dragArrow.to.x);
      const minDeltaY = -Math.min(start.dragArrow.from.y, start.dragArrow.to.y);
      const maxDeltaY = 640 - Math.max(start.dragArrow.from.y, start.dragArrow.to.y);
      const dx = Math.max(minDeltaX, Math.min(maxDeltaX, deltaX));
      const dy = Math.max(minDeltaY, Math.min(maxDeltaY, deltaY));
      setBoard((current) => ({ ...current, arrows: (current.arrows || []).map((arrow) => arrow.id === draggingArrowId ? { ...arrow, from: { x: start.dragArrow.from.x + dx, y: start.dragArrow.from.y + dy }, to: { x: start.dragArrow.to.x + dx, y: start.dragArrow.to.y + dy } } : arrow) }));
    }
    if (resizingShapeId && dragStartRef.current?.resizeShape) {
      const start = dragStartRef.current;
      const delta = { x: point.x - start.pointer.x, y: point.y - start.pointer.y };
      const size = pitchMode !== "full" && start.resizeShape.type === "text"
        ? resizePortraitTextBox(start.resizeShape, start.resizeCorner, delta)
        : resizeTacticalShape(start.resizeShape, start.resizeCorner, delta);
      setBoard((current) => ({ ...current, shapes: (current.shapes || []).map((shape) => shape.id === resizingShapeId ? { ...shape, ...size } : shape) }));
    }
    if (resizingArrowEnd && dragStartRef.current?.resizeArrow) {
      const endpoint = { x: point.x, y: point.y };
      setBoard((current) => ({ ...current, arrows: (current.arrows || []).map((arrow) => arrow.id === resizingArrowEnd.id ? { ...arrow, [resizingArrowEnd.endpoint]: endpoint } : arrow) }));
    }
    if (arrowStart) setArrowPreview(point);
  };
  const handlePitchPointerUp = (event) => {
    if (draggingIds.length && dragStartRef.current) {
      const finalPoint = pointFromEvent(event);
      const delta = { x: finalPoint.x - dragStartRef.current.pointer.x, y: finalPoint.y - dragStartRef.current.pointer.y };
      const finalTargets = Object.fromEntries(draggingIds.map((id) => [id, {
        x: Math.max(24, Math.min(976, dragStartRef.current.positions[id].x + delta.x)),
        y: Math.max(24, Math.min(616, dragStartRef.current.positions[id].y + delta.y)),
      }]));
      dragTargetsRef.current = finalTargets;
      setBoard((current) => ({ ...current, players: current.players.map((player) => finalTargets[player.id] ? { ...player, ...finalTargets[player.id] } : player) }));
    }
    if (draggingBall && dragStartRef.current) {
      const finalPoint = pointFromEvent(event);
      const origin = dragStartRef.current.positions["tactical-ball"];
      const delta = { x: finalPoint.x - dragStartRef.current.pointer.x, y: finalPoint.y - dragStartRef.current.pointer.y };
      const finalBall = { x: Math.max(24, Math.min(976, origin.x + delta.x)), y: Math.max(24, Math.min(616, origin.y + delta.y)) };
      ballDragTargetRef.current = finalBall;
      if (activeClipStartRef.current) {
        const path = activeClipBallPathRef.current;
        const last = path[path.length - 1];
        if (!last || Math.hypot(finalBall.x - last.x, finalBall.y - last.y) > 0.5) path.push({ ...finalBall });
      }
      setBoard((current) => ({ ...current, ball: finalBall }));
    }
    if (draggingShapeId && dragStartRef.current) {
      const finalPoint = pointFromEvent(event);
      const origin = dragStartRef.current.positions[draggingShapeId];
      const delta = { x: finalPoint.x - dragStartRef.current.pointer.x, y: finalPoint.y - dragStartRef.current.pointer.y };
      const finalPosition = { x: Math.max(0, Math.min(1000, origin.x + delta.x)), y: Math.max(0, Math.min(640, origin.y + delta.y)) };
      setBoard((current) => ({ ...current, shapes: (current.shapes || []).map((shape) => shape.id === draggingShapeId ? { ...shape, ...finalPosition } : shape) }));
    }
    if (draggingArrowId && dragStartRef.current?.dragArrow) {
      const start = dragStartRef.current;
      const finalPoint = pointFromEvent(event);
      const deltaX = finalPoint.x - start.pointer.x;
      const deltaY = finalPoint.y - start.pointer.y;
      const minDeltaX = -Math.min(start.dragArrow.from.x, start.dragArrow.to.x);
      const maxDeltaX = 1000 - Math.max(start.dragArrow.from.x, start.dragArrow.to.x);
      const minDeltaY = -Math.min(start.dragArrow.from.y, start.dragArrow.to.y);
      const maxDeltaY = 640 - Math.max(start.dragArrow.from.y, start.dragArrow.to.y);
      const dx = Math.max(minDeltaX, Math.min(maxDeltaX, deltaX));
      const dy = Math.max(minDeltaY, Math.min(maxDeltaY, deltaY));
      setBoard((current) => ({ ...current, arrows: (current.arrows || []).map((arrow) => arrow.id === draggingArrowId ? { ...arrow, from: { x: start.dragArrow.from.x + dx, y: start.dragArrow.from.y + dy }, to: { x: start.dragArrow.to.x + dx, y: start.dragArrow.to.y + dy } } : arrow) }));
    }
    if (resizingShapeId && dragStartRef.current?.resizeShape) {
      const start = dragStartRef.current;
      const finalPoint = pointFromEvent(event);
      const delta = { x: finalPoint.x - start.pointer.x, y: finalPoint.y - start.pointer.y };
      const size = pitchMode !== "full" && start.resizeShape.type === "text"
        ? resizePortraitTextBox(start.resizeShape, start.resizeCorner, delta)
        : resizeTacticalShape(start.resizeShape, start.resizeCorner, delta);
      setBoard((current) => ({ ...current, shapes: (current.shapes || []).map((shape) => shape.id === resizingShapeId ? { ...shape, ...size } : shape) }));
    }
    if (resizingArrowEnd && dragStartRef.current?.resizeArrow) {
      const finalPoint = pointFromEvent(event);
      const endpoint = { x: finalPoint.x, y: finalPoint.y };
      setBoard((current) => ({ ...current, arrows: (current.arrows || []).map((arrow) => arrow.id === resizingArrowEnd.id ? { ...arrow, [resizingArrowEnd.endpoint]: endpoint } : arrow) }));
    }
    if (arrowStart) {
      const end = pointFromEvent(event);
      const distance = Math.hypot(end.x - arrowStart.x, end.y - arrowStart.y);
      if (distance > 12 && mode === "arrow") {
        const newArrow = { id: `arrow-${Date.now()}`, from: arrowStart, to: end, color: newShapeColour, opacity: newShapeOpacity, strokeWidth: newShapeStrokeWidth, lineStyle: newArrowLineStyle };
        setBoard((current) => ({ ...current, arrows: [...current.arrows, newArrow] }));
        markGraphicEntering(newArrow.id);
        setSelectedArrowId(newArrow.id);
        setSelectedShapeId(null);
      }
      if (distance > 12 && ["rectangle", "square", "circle", "triangle", "line"].includes(mode)) {
        const box = shapeForDrag(arrowStart, end, mode);
        const newShape = { id: `shape-${Date.now()}`, type: mode, ...box, color: newShapeColour, opacity: newShapeOpacity, strokeWidth: newShapeStrokeWidth };
        setBoard((current) => ({ ...current, shapes: [...(current.shapes || []), newShape] }));
        markGraphicEntering(newShape.id);
        setSelectedShapeId(newShape.id);
        setSelectedArrowId(null);
      }
      if (mode === "text") {
        const drawnBox = distance > 12 ? textBoxForDrag(arrowStart, end) : pitchMode === "full"
          ? { x: arrowStart.x, y: arrowStart.y, width: 220, height: 48 }
          : { x: arrowStart.x - 110, y: arrowStart.y - 24, width: 220, height: 48 };
        const width = Math.max(140, drawnBox.width);
        const height = Math.max(48, drawnBox.height);
        const centerX = drawnBox.x + drawnBox.width / 2;
        const centerY = drawnBox.y + drawnBox.height / 2;
        const x = Math.max(0, Math.min(1000 - width, centerX - width / 2));
        const y = Math.max(0, Math.min(640 - height, centerY - height / 2));
        const newTextBox = { id: `text-${Date.now()}`, type: "text", x, y, width, height, text: "Text", fontSize: 22, color: newShapeColour, textColor: newTextColor, backgroundColor: newTextBoxColor, opacity: 0.85 };
        setBoard((current) => ({ ...current, shapes: [...(current.shapes || []), newTextBox] }));
        markGraphicEntering(newTextBox.id);
        setSelectedShapeId(newTextBox.id);
        setSelectedArrowId(null);
        setMode("move");
        setMultiSelectMode(false);
        setEditingTextId(newTextBox.id);
      }
      setArrowStart(null);
      setArrowPreview(null);
    }
    dragStartRef.current = null;
    dragTargetsRef.current = {};
    ballDragTargetRef.current = null;
    setDraggingIds([]);
    setDraggingBall(false);
    setDraggingShapeId(null);
    setDraggingArrowId(null);
    setResizingShapeId(null);
    setResizingArrowEnd(null);
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
    if (!selectedTokens.length) return;
    setBoard((current) => ({ ...current, players: current.players.filter((player) => !selectedTokens.includes(player.id)) }));
    setSelectedTokens([]);
  };
  const selectedShape = (board.shapes || []).find((shape) => shape.id === selectedShapeId);
  const selectedArrow = (board.arrows || []).find((arrow) => arrow.id === selectedArrowId);
  const selectedGraphic = selectedShape || selectedArrow;
  useEffect(() => {
    const handleTextBoxShortcuts = (event) => {
      const key = event.key.toLowerCase();
      const targetTag = event.target?.tagName?.toLowerCase();
      if (["delete", "backspace"].includes(key) && !["input", "textarea"].includes(targetTag) && !event.target?.isContentEditable) {
        if (selectedShape) {
          event.preventDefault();
          setBoard((current) => ({ ...current, shapes: (current.shapes || []).filter((shape) => shape.id !== selectedShape.id) }));
          setSelectedShapeId(null);
          setEditingTextId(null);
        } else if (selectedArrow) {
          event.preventDefault();
          setBoard((current) => ({ ...current, arrows: (current.arrows || []).filter((arrow) => arrow.id !== selectedArrow.id) }));
          setSelectedArrowId(null);
        }
        return;
      }
      if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
      if (key === "c" && selectedShape?.type === "text") {
        event.preventDefault();
        setCopiedTextBox({ ...selectedShape });
      } else if (key === "v" && copiedTextBox) {
        event.preventDefault();
        const base = selectedShape?.type === "text" ? selectedShape : copiedTextBox;
        const width = copiedTextBox.width || 220;
        const height = copiedTextBox.height || 48;
        const pasted = {
          ...copiedTextBox,
          id: `text-${Date.now()}`,
          x: Math.max(0, Math.min(1000 - width, base.x + 24)),
          y: Math.max(0, Math.min(640 - height, base.y + 24)),
        };
        setBoard((current) => ({ ...current, shapes: [...(current.shapes || []), pasted] }));
        setEnteringGraphicIds((current) => [...current, pasted.id]);
        setSelectedShapeId(pasted.id);
        setSelectedArrowId(null);
        setEditingTextId(null);
        setMode("move");
      }
    };
    window.addEventListener("keydown", handleTextBoxShortcuts);
    return () => window.removeEventListener("keydown", handleTextBoxShortcuts);
  }, [selectedShape, selectedArrow, copiedTextBox]);
  const updateSelectedShape = (changes) => {
    if (!selectedShapeId) return;
    setBoard((current) => ({ ...current, shapes: (current.shapes || []).map((shape) => shape.id === selectedShapeId ? { ...shape, ...changes } : shape) }));
  };
  const updateSelectedGraphic = (changes) => {
    if (selectedShape) updateSelectedShape(changes);
    else if (selectedArrowId) setBoard((current) => ({ ...current, arrows: (current.arrows || []).map((arrow) => arrow.id === selectedArrowId ? { ...arrow, ...changes } : arrow) }));
  };
  const displayedBoard = playbackBoard || board;
  const pitchViewMode = videoPlaying ? (displayedBoard.pitchMode || pitchMode) : pitchMode;
  const beginOrSaveMovementClip = () => {
    if (!activeClipStartRef.current) {
      const start = JSON.parse(JSON.stringify(boardRef.current));
      activeClipStartRef.current = start;
      activeClipBallPathRef.current = [{ ...(start.ball || { x: 500, y: 320 }) }];
      setActiveClipStart(start);
      setVideoStatus(`Movement clip ${movementClips.length + 1} started. Move the players or ball, then save the clip.`);
      return;
    }
    const from = activeClipStartRef.current;
    const to = JSON.parse(JSON.stringify(boardRef.current));
    const movedPlayers = to.players.filter((player) => {
      const origin = from.players.find((item) => item.id === player.id);
      return origin && Math.hypot(player.x - origin.x, player.y - origin.y) > 2;
    }).length;
    const fromShapes = from.shapes || [];
    const toShapes = to.shapes || [];
    const movedShapes = new Set([...fromShapes, ...toShapes].filter((shape, index, all) => all.findIndex((item) => item.id === shape.id) === index).filter((shape) => {
      const origin = fromShapes.find((item) => item.id === shape.id);
      const destination = toShapes.find((item) => item.id === shape.id);
      if (!origin || !destination) return true;
      return Math.max(Math.abs(destination.x - origin.x), Math.abs(destination.y - origin.y), Math.abs(destination.width - origin.width), Math.abs(destination.height - origin.height)) > 2;
    })).size;
    const fromArrows = from.arrows || [];
    const toArrows = to.arrows || [];
    const movedArrows = new Set([...fromArrows, ...toArrows].filter((arrow, index, all) => all.findIndex((item) => item.id === arrow.id) === index).filter((arrow) => {
      const origin = fromArrows.find((item) => item.id === arrow.id);
      const destination = toArrows.find((item) => item.id === arrow.id);
      if (!origin || !destination) return true;
      return Math.max(Math.abs(destination.from.x - origin.from.x), Math.abs(destination.from.y - origin.from.y), Math.abs(destination.to.x - origin.to.x), Math.abs(destination.to.y - origin.to.y), Math.abs((destination.strokeWidth || 6) - (origin.strokeWidth || 6))) > 2 || destination.lineStyle !== origin.lineStyle || destination.color !== origin.color;
    })).size;
    const fromBall = from.ball || { x: 500, y: 320 };
    const toBall = to.ball || { x: 500, y: 320 };
    const ballPath = [...(activeClipBallPathRef.current.length ? activeClipBallPathRef.current : [fromBall])];
    const lastPathPoint = ballPath[ballPath.length - 1];
    if (!lastPathPoint || Math.hypot(toBall.x - lastPathPoint.x, toBall.y - lastPathPoint.y) > 0.5) ballPath.push({ ...toBall });
    const ballMoved = pathLength(ballPath) > 2 || Math.hypot(toBall.x - fromBall.x, toBall.y - fromBall.y) > 2;
    const movedCount = movedPlayers + (ballMoved ? 1 : 0) + movedShapes + movedArrows;
    if (!movedCount) {
      setVideoStatus("No movement recorded yet. Move at least one disc or the ball, then save this clip.");
      return;
    }
    const clip = { id: `movement-${Date.now()}`, from, to, ballPath, movedCount };
    persistMovementClips((current) => [...current, clip]);
    activeClipStartRef.current = null;
    activeClipBallPathRef.current = [];
    setActiveClipStart(null);
    setVideoStatus(`Movement clip ${movementClips.length + 1} saved. Add another clip or play all clips.`);
  };
  const cancelMovementClip = () => {
    activeClipStartRef.current = null;
    activeClipBallPathRef.current = [];
    setActiveClipStart(null);
    setVideoStatus("Movement clip cancelled.");
  };
  const previewMovementClip = (clip, clipNumber) => {
    if (finishingVideo || activeClipStartRef.current) return;
    if (clipPreviewFrameRef.current) window.cancelAnimationFrame(clipPreviewFrameRef.current);
    const startedAt = performance.now() + 50;
    const duration = tacticalClipDuration(clip);
    setFinishingVideo(true);
    setVideoPlaying(true);
    setPlaybackBoard(clip.from);
    setVideoStatus(`Previewing movement clip ${clipNumber} on the pitch…`);
    const animateFrame = () => {
      const progress = Math.min(1, Math.max(0, (performance.now() - startedAt) / duration));
      const eased = progress < 0.5 ? 4 * progress * progress * progress : 1 - Math.pow(-2 * progress + 2, 3) / 2;
      const players = clip.to.players.map((player) => {
        const origin = clip.from.players.find((item) => item.id === player.id);
        return origin ? { ...player, x: origin.x + (player.x - origin.x) * eased, y: origin.y + (player.y - origin.y) * eased } : player;
      });
      const fromBall = clip.from.ball || { x: 500, y: 320 };
      const toBall = clip.to.ball || { x: 500, y: 320 };
      const ball = pointAlongPath(clip.ballPath, eased, fromBall, toBall);
      const shapes = interpolateTacticalShapes(clip.from.shapes || [], clip.to.shapes || [], eased);
      const arrows = interpolateTacticalArrows(clip.from.arrows || [], clip.to.arrows || [], eased);
      setPlaybackBoard({
        ...clip.to,
        name: boardRef.current.name,
        homeName: boardRef.current.homeName,
        awayName: boardRef.current.awayName,
        arrows,
        shapes,
        players,
        ball,
      });
      if (progress < 1) clipPreviewFrameRef.current = window.requestAnimationFrame(animateFrame);
      else {
        clipPreviewFrameRef.current = null;
        setPlaybackBoard(null);
        setVideoPlaying(false);
        setFinishingVideo(false);
        setVideoStatus(`Clip ${clipNumber} preview finished.`);
      }
    };
    clipPreviewFrameRef.current = window.requestAnimationFrame(animateFrame);
  };
  const playAllMovementClips = () => {
    const clips = JSON.parse(JSON.stringify(movementClips));
    if (!clips.length || finishingVideo) return;
    const canvas = videoCanvasRef.current;
    if (!canvas?.captureStream || typeof MediaRecorder === "undefined") {
      setVideoStatus("Video recording is not supported by this browser. Try Chrome, Edge, or Safari.");
      return;
    }
    drawTacticalVideoFrame(canvas, clips[0].from);
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
    if (recordedVideo?.url) URL.revokeObjectURL(recordedVideo.url);
    setRecordedVideo(null);
    setPlaybackBoard(clips[0].from);
    const clipDurations = clips.map(tacticalClipDuration);
    const pauseBetweenClips = 450;
    const timelineDuration = clipDurations.reduce((total, duration) => total + duration, 0) + Math.max(0, clips.length - 1) * pauseBetweenClips;
    const startedAt = performance.now() + 450;
    recorder.ondataavailable = (event) => { if (event.data?.size) videoChunksRef.current.push(event.data); };
    recorder.onstop = () => {
      const blob = new Blob(videoChunksRef.current, { type: recorder.mimeType || mimeType || "video/webm" });
      if (blob.size) {
        const url = URL.createObjectURL(blob);
        const extension = blob.type.includes("mp4") ? "mp4" : "webm";
        const filename = (boardRef.current.name || "tactical-movement").trim().replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "tactical-movement";
        setRecordedVideo({ url, filename: `${filename}.${extension}` });
        setVideoStatus("Combined video ready. Play it here or download it when you're finished.");
      } else setVideoStatus("No video frames were recorded. Please try again.");
      stream.getTracks().forEach((track) => track.stop());
      videoRecorderRef.current = null;
      setVideoRecording(false);
      setFinishingVideo(false);
    };
    recorder.onerror = () => {
      setVideoStatus("Video recording stopped because of a browser error.");
      stream.getTracks().forEach((track) => track.stop());
      setVideoRecording(false);
      setFinishingVideo(false);
    };
    videoRecorderRef.current = recorder;
    recorder.start(250);
    setVideoRecording(true);
    setFinishingVideo(true);
    setVideoPlaying(true);
    setVideoStatus(`Playing ${clips.length} movement clip${clips.length === 1 ? "" : "s"} in sequence and recording one video…`);
    const drawFrame = () => {
      const now = performance.now();
      const elapsed = Math.max(0, now - startedAt);
      if (elapsed >= timelineDuration) {
        const lastBoard = clips[clips.length - 1].to;
        const finalFrame = { ...lastBoard, name: boardRef.current.name, homeName: boardRef.current.homeName, awayName: boardRef.current.awayName, arrows: lastBoard.arrows || [], shapes: lastBoard.shapes || [] };
        drawTacticalVideoFrame(canvas, finalFrame);
        setPlaybackBoard(null);
        setVideoPlaying(false);
        setBoard((current) => ({ ...current, players: finalFrame.players, ball: finalFrame.ball }));
        videoFrameRef.current = null;
        videoFinishTimerRef.current = window.setTimeout(() => { if (recorder.state !== "inactive") recorder.stop(); }, 500);
        return;
      }
      let clipIndex = 0;
      let clipStart = 0;
      while (clipIndex < clips.length - 1 && elapsed >= clipStart + clipDurations[clipIndex] + pauseBetweenClips) {
        clipStart += clipDurations[clipIndex] + pauseBetweenClips;
        clipIndex += 1;
      }
      const clip = clips[clipIndex];
      const clipElapsed = Math.max(0, elapsed - clipStart);
      const progress = Math.min(1, clipElapsed / clipDurations[clipIndex]);
      const eased = progress < 0.5 ? 4 * progress * progress * progress : 1 - Math.pow(-2 * progress + 2, 3) / 2;
      const players = clip.to.players.map((player) => {
        const origin = clip.from.players.find((item) => item.id === player.id);
        if (!origin) return player;
        return { ...player, x: origin.x + (player.x - origin.x) * eased, y: origin.y + (player.y - origin.y) * eased };
      });
      const startBall = clip.from.ball || { x: 500, y: 320 };
      const endBall = clip.to.ball || { x: 500, y: 320 };
      const ball = pointAlongPath(clip.ballPath, eased, startBall, endBall);
      const shapes = interpolateTacticalShapes(clip.from.shapes || [], clip.to.shapes || [], eased);
      const arrows = interpolateTacticalArrows(clip.from.arrows || [], clip.to.arrows || [], eased);
      const frameBoard = {
        ...clip.to,
        name: boardRef.current.name,
        homeName: boardRef.current.homeName,
        awayName: boardRef.current.awayName,
        arrows,
        shapes,
        players,
        ball,
      };
      drawTacticalVideoFrame(canvas, frameBoard);
      setPlaybackBoard(frameBoard);
      videoFrameRef.current = window.requestAnimationFrame(drawFrame);
    };
    videoFrameRef.current = window.requestAnimationFrame(drawFrame);
  };
  return (
    <section className="sr-tactical-pad">
      <div className="sr-tactical-toolbar">
        <div className="sr-tactical-title-fields">
          <label><span>Tactic name</span><input value={board.name} onChange={(event) => setBoard((current) => ({ ...current, name: event.target.value }))} /></label>
          <label><span>Home team</span><input value={board.homeName} onChange={(event) => setBoard((current) => ({ ...current, homeName: event.target.value }))} /></label>
          <label><span>Opposition</span><input value={board.awayName} onChange={(event) => setBoard((current) => ({ ...current, awayName: event.target.value }))} /></label>
        </div>
      </div>
      {videoStatus && <p className="sr-tactical-video-status" role="status">{videoStatus}</p>}
      <canvas ref={videoCanvasRef} width="1280" height="819" className="sr-tactical-video-canvas" aria-hidden="true" />

      <div className={`sr-tactical-pitch-wrap ${mode === "arrow" ? "drawing" : "moving"} ${pitchViewMode !== "full" ? "portrait" : "landscape"}`}>
        <svg ref={pitchRef} className="sr-tactical-pitch" preserveAspectRatio="none" style={{ width: pitchViewMode === "full" ? undefined : pitchViewMode === "half" ? "92%" : "auto", height: pitchViewMode === "full" ? undefined : pitchViewMode === "half" ? "min(44vh, 460px)" : "min(45vh, 480px)", minHeight: pitchViewMode === "full" ? undefined : 0, maxWidth: pitchViewMode === "full" ? undefined : "100%", maxHeight: pitchViewMode === "full" ? undefined : pitchViewMode === "half" ? "min(44vh, 460px)" : "min(45vh, 480px)", aspectRatio: pitchViewMode === "full" ? "25 / 16" : pitchViewMode === "half" ? `750 / ${HALF_PITCH_VIEW_HEIGHT}` : "3 / 4" }} viewBox={pitchViewMode === "full" ? "0 0 1000 640" : pitchViewMode === "half" ? `0 0 750 ${HALF_PITCH_VIEW_HEIGHT}` : "0 0 750 1000"} role="img" aria-label={`${pitchViewMode === "full" ? "Full" : pitchViewMode === "half" ? "Half pitch plus a little beyond halfway" : "18-yard box"} tactical football pitch. Drag the discs to position players.`} onPointerDown={handlePitchPointerDown} onPointerMove={handlePitchPointerMove} onPointerUp={handlePitchPointerUp} onPointerCancel={handlePitchPointerUp}>
          <defs>{(displayedBoard.arrows || []).map((arrow) => <marker key={`head-${arrow.id}`} id={`sr-tactical-arrowhead-${arrow.id}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill={arrow.color || "#ffe36e"} fillOpacity={arrow.opacity ?? 1} /></marker>)}<marker id="sr-tactical-arrowhead-preview" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill={newShapeColour} /></marker></defs>
          {pitchViewMode === "full" ? <>
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
          </> : <g transform="scale(1.171875 1)">
            <rect x="12" y="12" width="616" height="976" rx="7" className="sr-tactical-grass" />
              {pitchViewMode === "half" ? <>
              <rect x="35" y="35" width="570" height="930" className="sr-tactical-line" />
              <line x1="35" y1="470" x2="605" y2="470" className="sr-tactical-line" />
              <circle cx="320" cy="470" r="83" className="sr-tactical-line" />
              <circle cx="320" cy="470" r="4" className="sr-tactical-mark" />
              <rect x="180" y="35" width="280" height="150" className="sr-tactical-line" />
              <rect x="245" y="35" width="150" height="58" className="sr-tactical-line" />
              <path d="M250 185 A82 82 0 0 0 390 185" className="sr-tactical-line" />
              <circle cx="320" cy="145" r="4" className="sr-tactical-mark" />
            </> : <>
              <rect x="35" y="35" width="570" height="930" className="sr-tactical-line" />
              <rect x="125" y="35" width="390" height="360" className="sr-tactical-line" />
              <rect x="220" y="35" width="200" height="150" className="sr-tactical-line" />
              <path d="M240 395 A82 82 0 0 0 400 395" className="sr-tactical-line" />
              <circle cx="320" cy="260" r="4" className="sr-tactical-mark" />
              <line x1="270" y1="20" x2="370" y2="20" className="sr-tactical-line" />
            </>}
          </g>}
          <g transform={pitchViewMode === "full" ? undefined : "scale(1.171875 1) translate(0 1000) rotate(-90)"}>
          {(displayedBoard.shapes || []).map((shape) => {
            const selected = !videoPlaying && selectedShapeId === shape.id;
            const entering = enteringGraphicIds.includes(shape.id);
            const graphicClass = `sr-tactical-shape ${selected ? "selected" : ""} ${entering ? "entering" : ""}`;
            const graphicEvents = {
              className: graphicClass,
              style: { opacity: shape._playbackOpacity ?? 1 },
              onAnimationEnd: () => finishGraphicEntering(shape.id),
              onPointerDown: (event) => beginShapeDrag(event, shape),
            };
            const fill = shape.color || "#62dcff";
            const opacity = shape.opacity ?? 0.35;
            const strokeWidth = Number(shape.strokeWidth) || 3;
            let shapeBody;
            if (shape.type === "text") {
              shapeBody = <><rect x={shape.x} y={shape.y} width={shape.width} height={shape.height} rx="5" fill={shape.backgroundColor || "#0c3769"} fillOpacity={shape.opacity ?? 0.85} stroke={selected ? "#ffe36e" : fill} strokeOpacity={1} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} />{editingTextId === shape.id && !videoPlaying ? <foreignObject x={shape.x + 5} y={shape.y + 4} width={Math.max(20, shape.width - 10)} height={Math.max(24, shape.height - 8)}><textarea xmlns="http://www.w3.org/1999/xhtml" className="sr-tactical-inline-text" style={{ fontSize: `${shape.fontSize || 22}px`, color: shape.textColor || "#fff", backgroundColor: shape.backgroundColor || "#0c3769" }} autoFocus aria-label="Edit text on pitch" value={shape.text || ""} onPointerDown={(event) => event.stopPropagation()} onChange={(event) => setBoard((current) => ({ ...current, shapes: (current.shapes || []).map((item) => item.id === shape.id ? { ...item, text: event.target.value } : item) }))} onBlur={() => setEditingTextId(null)} onKeyDown={(event) => { if (event.key === "Escape") { event.preventDefault(); setEditingTextId(null); event.currentTarget.blur(); } }} /></foreignObject> : <foreignObject x={shape.x + 5} y={shape.y + 4} width={Math.max(20, shape.width - 10)} height={Math.max(24, shape.height - 8)}><div xmlns="http://www.w3.org/1999/xhtml" className="sr-tactical-text-render" style={{ fontSize: `${shape.fontSize || 22}px`, color: shape.textColor || "#fff" }} onDoubleClick={(event) => { event.stopPropagation(); setSelectedShapeId(shape.id); setEditingTextId(shape.id); }}>{shape.text || "Text"}</div></foreignObject>}</>;
            } else if (shape.type === "line") shapeBody = <line x1={shape.x} y1={shape.y} x2={shape.x + shape.width} y2={shape.y + shape.height} stroke={selected ? "#ffe36e" : fill} strokeOpacity={opacity} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} pointerEvents="stroke" />;
            else if (shape.type === "circle") shapeBody = <ellipse cx={shape.x + shape.width / 2} cy={shape.y + shape.height / 2} rx={shape.width / 2} ry={shape.height / 2} fill={fill} fillOpacity={opacity} stroke={selected ? "#ffe36e" : fill} strokeOpacity={selected ? 1 : opacity} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} />;
            else if (shape.type === "triangle") shapeBody = <polygon points={`${shape.x + shape.width / 2},${shape.y} ${shape.x + shape.width},${shape.y + shape.height} ${shape.x},${shape.y + shape.height}`} fill={fill} fillOpacity={opacity} stroke={selected ? "#ffe36e" : fill} strokeOpacity={selected ? 1 : opacity} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} />;
            else shapeBody = <rect x={shape.x} y={shape.y} width={shape.width} height={shape.height} rx="2" fill={fill} fillOpacity={opacity} stroke={selected ? "#ffe36e" : fill} strokeOpacity={selected ? 1 : opacity} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} />;
            const handles = selected && !videoPlaying ? [["nw", shape.x, shape.y], ["ne", shape.x + shape.width, shape.y], ["sw", shape.x, shape.y + shape.height], ["se", shape.x + shape.width, shape.y + shape.height]] : [];
            const keepTextReadable = shape.type === "text" && pitchViewMode !== "full";
            const counterRotation = keepTextReadable ? `rotate(90 ${shape.x + shape.width / 2} ${shape.y + shape.height / 2})` : undefined;
            return <g key={`${shape.id}-${pitchViewMode}`} {...graphicEvents}><g transform={counterRotation}>{shapeBody}{handles.map(([corner, x, y]) => <circle key={corner} className="sr-tactical-resize-handle" style={{ cursor: corner === "nw" || corner === "se" ? "nwse-resize" : "nesw-resize" }} cx={x} cy={y} r="7" onPointerDown={(event) => beginShapeResize(event, shape, corner)} />)}</g></g>;
          })}
          {arrowPreview && arrowStart && mode === "arrow" && <line x1={arrowStart.x} y1={arrowStart.y} x2={arrowPreview.x} y2={arrowPreview.y} className="sr-tactical-arrow sr-tactical-arrow-preview" stroke={newShapeColour} markerEnd="url(#sr-tactical-arrowhead-preview)" />}
          {arrowPreview && arrowStart && ["rectangle", "square", "circle", "triangle", "line", "text"].includes(mode) && (() => {
            const shape = mode === "text" ? textBoxForDrag(arrowStart, arrowPreview) : shapeForDrag(arrowStart, arrowPreview, mode);
            if (mode === "text") return <g transform={pitchViewMode === "full" ? undefined : `rotate(90 ${shape.x + shape.width / 2} ${shape.y + shape.height / 2})`}><rect {...shape} rx="5" fill={newTextBoxColor} fillOpacity="0.85" stroke={newShapeColour} strokeWidth={2} className="sr-tactical-shape-preview" /></g>;
            if (mode === "line") return <line x1={arrowStart.x} y1={arrowStart.y} x2={arrowPreview.x} y2={arrowPreview.y} stroke={newShapeColour} strokeOpacity={newShapeOpacity} strokeWidth={3} className="sr-tactical-shape-preview" />;
            if (mode === "circle") return <ellipse cx={shape.x + shape.width / 2} cy={shape.y + shape.height / 2} rx={shape.width / 2} ry={shape.height / 2} fill={newShapeColour} fillOpacity={newShapeOpacity} stroke={newShapeColour} strokeOpacity={newShapeOpacity} strokeWidth={3} className="sr-tactical-shape-preview" />;
            if (mode === "triangle") return <polygon points={`${shape.x + shape.width / 2},${shape.y} ${shape.x + shape.width},${shape.y + shape.height} ${shape.x},${shape.y + shape.height}`} fill={newShapeColour} fillOpacity={newShapeOpacity} stroke={newShapeColour} strokeOpacity={newShapeOpacity} strokeWidth={3} className="sr-tactical-shape-preview" />;
            return <rect {...shape} fill={newShapeColour} fillOpacity={newShapeOpacity} stroke={newShapeColour} strokeOpacity={newShapeOpacity} strokeWidth={3} className="sr-tactical-shape-preview" />;
          })()}
          {(displayedBoard.arrows || []).map((arrow) => {
            const selected = !videoPlaying && selectedArrowId === arrow.id;
            const strokeWidth = Number(arrow.strokeWidth) || 6;
            const dashPattern = arrow.lineStyle === "dashed" ? "14 9" : arrow.lineStyle === "dotted" ? "2 8" : undefined;
            return <g key={arrow.id}>
              <line x1={arrow.from.x} y1={arrow.from.y} x2={arrow.to.x} y2={arrow.to.y} stroke={arrow.color || "#ffe36e"} strokeOpacity={arrow.opacity ?? 1} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} strokeDasharray={dashPattern} strokeLinecap="round" className={`sr-tactical-arrow ${selected ? "selected" : ""} ${draggingArrowId === arrow.id ? "dragging" : ""} ${enteringGraphicIds.includes(arrow.id) ? "entering" : ""}`} markerEnd={`url(#sr-tactical-arrowhead-${arrow.id})`} onAnimationEnd={() => finishGraphicEntering(arrow.id)} onPointerDown={(event) => beginArrowDrag(event, arrow)} />
              {selected && [["from", arrow.from], ["to", arrow.to]].map(([endpoint, point]) => <circle key={endpoint} className="sr-tactical-arrow-resize-handle" cx={point.x} cy={point.y} r="7" onPointerDown={(event) => beginArrowEndResize(event, arrow, endpoint)} />)}
            </g>;
          })}
          <g className={`sr-tactical-ball ${draggingBall ? "dragging" : ""} ${videoPlaying ? "video-playing" : ""}`} transform={`translate(${displayedBoard.ball?.x ?? 500} ${displayedBoard.ball?.y ?? 320})`} onPointerDown={beginBallDrag}>
            <circle r="13" className="sr-tactical-ball-shadow" /><circle r="10" className="sr-tactical-ball-shell" />
            <path d="M0 -5 L4 -2 L3 3 L-3 3 L-4 -2 Z" className="sr-tactical-ball-patch" />
          </g>
          {displayedBoard.players.map((player) => <g key={player.id} className={`sr-tactical-player ${player.side} ${!videoPlaying && selectedTokens.includes(player.id) ? "selected" : ""} ${draggingIds.includes(player.id) ? "dragging" : ""} ${videoPlaying ? "video-playing" : ""}`} transform={`translate(${player.x} ${player.y})`} onPointerDown={(event) => beginPlayerDrag(event, player)}>
            <circle r="27" className="sr-tactical-disc-shadow" />
            <circle r="24" className="sr-tactical-disc" />
            <text y="1" transform={pitchViewMode === "full" ? undefined : "rotate(90)"}>{player.number}</text>
          </g>)}
          </g>
        </svg>
      </div>
      <div className="sr-tactical-dock">
        <div className="sr-tactical-dock-tabs" role="tablist" aria-label="Tactical pad toolbar">
          <button type="button" role="tab" aria-selected={dockPanel === "animation"} className={dockPanel === "animation" ? "active" : ""} onClick={() => setDockPanel("animation")}>▶ Animation</button>
          <button type="button" role="tab" aria-selected={dockPanel === "shapes"} className={dockPanel === "shapes" ? "active" : ""} onClick={() => setDockPanel("shapes")}>◇ Shapes</button>
          <button type="button" role="tab" aria-selected={dockPanel === "text"} className={dockPanel === "text" ? "active" : ""} onClick={() => { setDockPanel("text"); if (mode !== "text" && selectedShape?.type !== "text") setMode("move"); }}>T Text</button>
          <button type="button" role="tab" aria-selected={dockPanel === "tools"} className={dockPanel === "tools" ? "active" : ""} onClick={() => setDockPanel("tools")}>⚙ Tools</button>
        </div>
        {dockPanel === "animation" && <div className="sr-tactical-dock-panel" role="tabpanel">
          <div className="sr-tactical-tools" role="toolbar" aria-label="Movement clip tools">
            <button type="button" className={activeClipStart ? "sr-tactical-recording" : "sr-tactical-video"} onClick={beginOrSaveMovementClip} disabled={finishingVideo}>{activeClipStart ? "Save movement clip" : "Start movement clip"}</button>
            {activeClipStart && <button type="button" onClick={cancelMovementClip} disabled={finishingVideo}>Cancel clip</button>}
            <button type="button" className={videoRecording ? "sr-tactical-recording" : "sr-tactical-video"} onClick={playAllMovementClips} disabled={!movementClips.length || finishingVideo || Boolean(activeClipStart)}>{finishingVideo ? "Playing clips…" : `Play all clips${movementClips.length ? ` (${movementClips.length})` : ""}`}</button>
            <small>Start a clip, move the discs or ball, then save it. Repeat to build a sequence.</small>
          </div>
          {movementClips.length > 0 && <div className="sr-tactical-clip-list"><strong>Movement clips — select one to preview</strong><ol>{movementClips.map((clip, index) => <li key={clip.id}><button type="button" className="sr-tactical-clip-preview-button" onClick={() => previewMovementClip(clip, index + 1)} disabled={finishingVideo || Boolean(activeClipStart)}>Clip {index + 1}: {clip.movedCount} piece{clip.movedCount === 1 ? "" : "s"} moved <span>Preview on pitch</span></button><button type="button" onClick={() => persistMovementClips((current) => current.filter((item) => item.id !== clip.id))} disabled={finishingVideo}>Remove</button></li>)}</ol><div className="sr-tactical-clip-list-actions"><button type="button" onClick={() => persistMovementClips((current) => current.slice(0, -1))} disabled={finishingVideo}>Undo last clip</button><button type="button" onClick={() => persistMovementClips([])} disabled={finishingVideo}>Clear clips</button></div></div>}
        </div>}
        {dockPanel === "shapes" && <div className="sr-tactical-dock-panel" role="tabpanel">
          <div className="sr-tactical-tools sr-tactical-shape-palette" role="toolbar" aria-label="Shape and drawing tools">
            <button type="button" className={`sr-tactical-icon-tool ${mode === "move" && !multiSelectMode ? "selected" : ""}`} onClick={() => { setMode("move"); setMultiSelectMode(false); }}><TacticalToolIcon type="move" /><span>Select</span></button>
            <button type="button" className={`sr-tactical-icon-tool ${multiSelectMode ? "selected" : ""}`} onClick={() => { setMode("move"); setMultiSelectMode((value) => !value); }}><TacticalToolIcon type="multiple" /><span>Multi</span></button>
            <button type="button" className={`sr-tactical-icon-tool ${mode === "arrow" ? "selected" : ""}`} onClick={() => { setMode("arrow"); setSelectedShapeId(null); setSelectedArrowId(null); }}><TacticalToolIcon type="arrow" /><span>Arrow</span></button>
            {[["line", "Line"], ["rectangle", "Rectangle"], ["square", "Square"], ["circle", "Circle"], ["triangle", "Triangle"]].map(([tool, label]) => <button key={tool} type="button" className={`sr-tactical-icon-tool ${mode === tool ? "selected" : ""}`} onClick={() => { setMode(tool); setSelectedShapeId(null); setSelectedArrowId(null); }}><TacticalToolIcon type={tool} /><span>{label}</span></button>)}
            <label className="sr-tactical-color-control"><span>Colour</span><input type="color" value={selectedGraphic?.color || newShapeColour} onChange={(event) => { setNewShapeColour(event.target.value); updateSelectedGraphic({ color: event.target.value }); }} /></label>
            <label className="sr-tactical-opacity-control"><span>Transparency</span><input aria-label="Shape transparency" type="range" min="0" max="95" step="5" value={100 - Math.round((selectedGraphic?.opacity ?? newShapeOpacity) * 100)} onChange={(event) => { const opacity = 1 - Number(event.target.value) / 100; if (selectedGraphic) updateSelectedGraphic({ opacity }); else setNewShapeOpacity(opacity); }} /><output>{100 - Math.round((selectedGraphic?.opacity ?? newShapeOpacity) * 100)}%</output></label>
            <label className="sr-tactical-stroke-width-control"><span>Line width</span><input aria-label="Line width" type="range" min="1" max="16" step="1" value={selectedGraphic?.strokeWidth || newShapeStrokeWidth} onChange={(event) => { const strokeWidth = Number(event.target.value); setNewShapeStrokeWidth(strokeWidth); updateSelectedGraphic({ strokeWidth }); }} /><output>{selectedGraphic?.strokeWidth || newShapeStrokeWidth}px</output></label>
            {selectedArrow && <label className="sr-tactical-line-style-control"><span>Arrow line</span><select aria-label="Arrow line style" value={selectedArrow.lineStyle || "solid"} onChange={(event) => { setNewArrowLineStyle(event.target.value); updateSelectedGraphic({ lineStyle: event.target.value }); }}><option value="solid">Solid</option><option value="dashed">Dashed</option><option value="dotted">Dotted</option></select></label>}
            <button type="button" onClick={() => { if (selectedShape) setBoard((current) => ({ ...current, shapes: (current.shapes || []).filter((shape) => shape.id !== selectedShapeId) })); else if (selectedArrowId) setBoard((current) => ({ ...current, arrows: (current.arrows || []).filter((arrow) => arrow.id !== selectedArrowId) })); setSelectedShapeId(null); setSelectedArrowId(null); }} disabled={!selectedGraphic}>Delete selected</button>
            <button type="button" onClick={() => { setBoard((current) => ({ ...current, shapes: [], arrows: [] })); setSelectedShapeId(null); setSelectedArrowId(null); }} disabled={!(board.shapes || []).length && !board.arrows.length}>Clear shapes</button>
          </div>
          <small>{selectedArrow ? "Drag either endpoint to resize the arrow. Change its colour or line width above, or press Delete to remove it." : selectedShape ? "Drag a corner handle to resize. Change colour or line width above, or press Delete to remove it." : "Choose a tool, then draw directly on the pitch. Select a drawn item to edit it."}</small>
        </div>}
        {dockPanel === "text" && <div className="sr-tactical-dock-panel sr-tactical-text-panel" role="tabpanel">
          <button type="button" className={`sr-tactical-icon-tool ${mode === "text" ? "selected" : ""}`} onClick={() => { setMode("text"); setSelectedShapeId(null); setSelectedArrowId(null); setEditingTextId(null); }}><TacticalToolIcon type="text" /><span>Draw text box</span></button>
          {selectedShape?.type === "text" && <><label><span>Edit selected text</span><input value={selectedShape.text || ""} onChange={(event) => updateSelectedShape({ text: event.target.value })} /></label><label className="sr-tactical-font-size-control"><span>Text size</span><input aria-label="Text size" type="range" min="12" max="42" step="1" value={selectedShape.fontSize || 22} onChange={(event) => updateSelectedShape({ fontSize: Number(event.target.value) })} /><output>{selectedShape.fontSize || 22}px</output></label></>}
          <div className="sr-tactical-text-colours"><label><span>Text colour</span><input aria-label="Text colour" type="color" value={selectedShape?.type === "text" ? selectedShape.textColor || "#ffffff" : newTextColor} onChange={(event) => { setNewTextColor(event.target.value); if (selectedShape?.type === "text") updateSelectedShape({ textColor: event.target.value }); }} /></label><label><span>Box colour</span><input aria-label="Text box colour" type="color" value={selectedShape?.type === "text" ? selectedShape.backgroundColor || "#0c3769" : newTextBoxColor} onChange={(event) => { setNewTextBoxColor(event.target.value); if (selectedShape?.type === "text") updateSelectedShape({ backgroundColor: event.target.value }); }} /></label></div>
          <small>{selectedShape?.type === "text" ? "Drag the box to move it or its corner handles to resize. Double-click it to edit on the pitch. Use Cmd/Ctrl+C and Cmd/Ctrl+V to copy and paste boxes." : "Choose Draw text box, then drag on the pitch to set its size. New boxes switch back to Select automatically."}</small>
        </div>}
        {dockPanel === "tools" && <div className="sr-tactical-dock-panel" role="tabpanel">
          <div className="sr-tactical-tools" role="toolbar" aria-label="Player and pitch tools">
            <label className="sr-tactical-pitch-mode"><span>Pitch view</span><select aria-label="Pitch view" value={pitchMode} onChange={(event) => { const nextMode = event.target.value; setPitchMode(nextMode); setBoard((current) => ({ ...current, pitchMode: nextMode })); }}><option value="full">Full pitch · landscape</option><option value="half">Half pitch · portrait</option><option value="box">18-yard box · set-pieces</option></select></label>
            <button type="button" onClick={() => addPlayer("home")}>+ Home disc</button>
            <button type="button" onClick={() => addPlayer("away")}>+ Opponent disc</button>
            <button type="button" onClick={removeSelectedPlayer} disabled={!selectedTokens.length}>Remove selected player</button>
            <button type="button" onClick={() => { setBoard((current) => ({ ...current, arrows: [] })); setSelectedArrowId(null); }} disabled={!board.arrows.length}>Clear arrows</button>
            <button type="button" onClick={() => { setBoard({ ...freshTacticalBoard(homeTeamName, oppositionName), pitchMode }); setSelectedTokens([]); setSelectedShapeId(null); setSelectedArrowId(null); activeClipStartRef.current = null; setActiveClipStart(null); }}>Reset pitch</button>
          </div>
        </div>}
      </div>
      {recordedVideo && <div className="sr-tactical-video-preview"><button type="button" className="sr-tactical-video-close" aria-label="Close video preview" title="Close video preview" onClick={() => { setRecordedVideo(null); setPlaybackBoard(null); setVideoPlaying(false); setVideoStatus(""); setMode("move"); setMultiSelectMode(false); }}>×</button><video controls playsInline preload="metadata" src={recordedVideo.url}>Your browser cannot play this recording.</video><a href={recordedVideo.url} download={recordedVideo.filename}>Download video</a></div>}
      <div className="sr-tactical-legend"><span><i className="home" />{board.homeName}</span><span><i className="away" />{board.awayName}</span><span><i className="ball" />Ball</span><small>Start a movement clip, move players or the ball, and save it. Repeat for each movement, then play all clips as one video. The board saves automatically on this device.</small></div>
    </section>
  );
}

export default function TeamAnalysisPage() {
  const { user, loading: authLoading } = useAuth();
  const tacticalViewStorageKey = `scoutpro-team-analysis-view:${user?.id || "anonymous"}`;
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
  const [tacticalPadOpen, setTacticalPadOpen] = useState(() => {
    try { return localStorage.getItem(tacticalViewStorageKey) === "tactical-pad"; } catch { return false; }
  });
  const tacticalViewOwner = useRef(tacticalViewStorageKey);
  const dashboardRef = useRef(null);
  const hydrated = useRef(false);
  const storageKey = `scoutpro-team-analysis-uploaded-data:${user?.id || "anonymous"}`;

  useEffect(() => {
    if (tacticalViewOwner.current !== tacticalViewStorageKey) {
      tacticalViewOwner.current = tacticalViewStorageKey;
      try { setTacticalPadOpen(localStorage.getItem(tacticalViewStorageKey) === "tactical-pad"); } catch { setTacticalPadOpen(false); }
      return;
    }
    try { localStorage.setItem(tacticalViewStorageKey, tacticalPadOpen ? "tactical-pad" : "dashboard"); } catch {}
  }, [tacticalPadOpen, tacticalViewStorageKey]);

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
