import { ClubBadge } from "./ClubBadge";
import { clubBadgePathFor } from "./clubBadgePaths";
import { supabase } from "./supabaseClient";
import { useAuth } from "./AuthContext";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
const HALF_PITCH_VIEW_HEIGHT = 650;
const BOX_PITCH_VIEW = { x: -75, y: 0, width: 900, height: 520 };
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
const SET_PIECE_BLUE_TWO_FIX_VERSION = 1;

const pathLength = (points = []) => points.slice(1).reduce((total, point, index) => total + Math.hypot(point.x - points[index].x, point.y - points[index].y), 0);
const tacticalClipDuration = (clip) => {
  if (Number(clip.durationMs)) return Math.max(2000, Math.min(12000, Number(clip.durationMs)));
  return Math.max(3400, Math.min(9000, (pathLength(clip.ballPath || []) / 0.18) || 3400));
};
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
        headSize: (from.headSize || 20) + ((to.headSize || 20) - (from.headSize || 20)) * progress,
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

const wrapTacticalCanvasText = (context, value, maxWidth) => {
  const wrapped = [];
  String(value || "Text").split("\n").forEach((rawLine) => {
    if (!rawLine) { wrapped.push(""); return; }
    const bullet = rawLine.match(/^([•\-*]\s*)/);
    const firstPrefix = bullet?.[0] || "";
    const continuationPrefix = bullet ? "  " : "";
    const words = rawLine.slice(firstPrefix.length).split(/\s+/);
    let line = firstPrefix;
    let prefix = firstPrefix;
    words.forEach((word) => {
      const candidate = line === prefix ? `${line}${word}` : `${line} ${word}`;
      if (line !== prefix && context.measureText(candidate).width > maxWidth) {
        wrapped.push(line);
        prefix = continuationPrefix;
        line = `${prefix}${word}`;
      } else line = candidate;
    });
    wrapped.push(line || firstPrefix);
  });
  return wrapped;
};

const drawTacticalVideoFrame = (canvas, board, viewMode = board.pitchMode || "full", { playerScale = 1, boxPdfStyle = false } = {}) => {
  if (!canvas) return;
  const width = canvas.width;
  const height = canvas.height;
  const context = canvas.getContext("2d");
  if (!context) return;
  const portrait = viewMode !== "full";
  // Keep 18-yard-box exports focused on the attacking end instead of
  // including the unused length of pitch down to the distant goalkeeper.
  const boxVideoCrop = viewMode === "box" ? { left: BOX_PITCH_VIEW.x, width: BOX_PITCH_VIEW.width, height: BOX_PITCH_VIEW.height } : null;
  const portraitCropWidth = boxVideoCrop?.width || 750;
  const portraitCropLeft = boxVideoCrop?.left || 0;
  context.clearRect(0, 0, width, height);
  if (portrait) {
    context.save();
    const viewHeight = viewMode === "half" ? HALF_PITCH_VIEW_HEIGHT : boxVideoCrop?.height || 1000;
    context.scale(width / portraitCropWidth, height / viewHeight);
    context.translate(-portraitCropLeft, 0);
    context.fillStyle = "#24734e";
    context.fillRect(12, 12, 726, 976);
    for (let y = 12; y < 988; y += 80) {
      if (Math.floor((y - 12) / 80) % 2 === 0) {
        context.fillStyle = "#3a8b59";
        context.globalAlpha = 0.38;
        context.fillRect(12, y, 726, 40);
      }
    }
    context.globalAlpha = 1;
    context.strokeStyle = "#e8f4e9";
    context.lineWidth = 3;
    context.strokeRect(41, 35, 668, 930);
    if (viewMode === "box") {
      context.beginPath(); context.ellipse(64.4375, 55, 23.4375, 20, 0, Math.PI, Math.PI * 1.5); context.stroke();
      context.beginPath(); context.ellipse(685.5625, 55, 23.4375, 20, 0, -Math.PI / 2, 0); context.stroke();
    }
    if (viewMode === "half") {
      context.beginPath(); context.moveTo(41, 470); context.lineTo(709, 470); context.stroke();
      context.beginPath(); context.arc(375, 470, 83, 0, Math.PI * 2); context.stroke();
      context.strokeRect(182, 35, 387, 120);
      context.strokeRect(275, 35, 199, 50);
      context.beginPath(); context.arc(375, 155, 96, Math.PI, 0, true); context.stroke();
      context.beginPath(); context.arc(375, 145, 4, 0, Math.PI * 2); context.fillStyle = "#e8f4e9"; context.fill();
    } else {
      context.strokeRect(146, 35, 458, 160);
      context.strokeRect(258, 35, 234, 60);
      context.beginPath(); context.arc(375, 145, 4, 0, Math.PI * 2); context.fillStyle = "#e8f4e9"; context.fill();
      if (boxPdfStyle) {
        const dRadius = 82;
        const dHalfChord = Math.sqrt(dRadius * dRadius - 50 * 50);
        context.beginPath();
        context.arc(375, 145, dRadius, Math.atan2(50, -dHalfChord), Math.atan2(50, dHalfChord), true);
        context.stroke();
        context.fillStyle = "rgba(232,244,233,0.18)";
        context.fillRect(316, 5, 118, 30);
        context.strokeStyle = "#e8f4e9";
        context.lineWidth = 3;
        context.strokeRect(316, 5, 118, 30);
        context.strokeStyle = "rgba(232,244,233,0.62)";
        context.lineWidth = 1;
        context.beginPath();
        [336, 356, 376, 396, 416].forEach((x) => { context.moveTo(x, 5); context.lineTo(x, 35); });
        [15, 25].forEach((y) => { context.moveTo(316, y); context.lineTo(434, y); });
        context.stroke();
      } else {
        context.beginPath(); context.arc(375, 195, 96, Math.PI, 0, true); context.stroke();
        context.beginPath(); context.moveTo(316, 20); context.lineTo(434, 20); context.stroke();
      }
    }
    context.restore();
    context.save();
    context.scale(width / portraitCropWidth, height / viewHeight);
    context.translate(-portraitCropLeft, 0);
    context.scale(1.171875, 1);
    context.translate(0, 1000);
    context.rotate(-Math.PI / 2);
  } else {
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
  }
  (board.shapes || []).forEach((shape) => {
    const opacity = Number(shape.opacity);
    const defaultOpacity = shape.type === "text" ? 0.85 : 0.35;
    const baseOpacity = Math.max(0, Math.min(1, Number.isFinite(opacity) ? opacity : defaultOpacity));
    context.globalAlpha = baseOpacity * (shape._playbackOpacity ?? 1);
    context.fillStyle = shape.type === "text" ? shape.backgroundColor || "#0c3769" : shape.color || "#62dcff";
    context.strokeStyle = shape.borderColor || shape.color || "#62dcff";
    context.lineWidth = Number(shape.strokeWidth) || 3;
    if (shape.type === "text") {
      const fontSize = Number(shape.fontSize) || 22;
      const lineHeight = fontSize * 1.2;
      context.font = `${shape.italic ? "italic " : ""}${shape.bold === false ? "normal" : "bold"} ${fontSize}px Arial, sans-serif`;
      context.textAlign = "left";
      context.textBaseline = "middle";
      const textLines = wrapTacticalCanvasText(context, shape.text, Math.max(20, shape.width - 20))
        .slice(0, Math.max(1, Math.floor((shape.height - 8) / lineHeight)));
      const firstLineY = -((textLines.length - 1) * lineHeight) / 2;
      const textAlign = ["left", "center", "right"].includes(shape.textAlign) ? shape.textAlign : "left";
      context.textAlign = textAlign;
      const textXFor = (left, center, right) => textAlign === "center" ? center : textAlign === "right" ? right : left;
      const drawTextLine = (line, x, y, maxWidth) => {
        context.fillText(line, x, y, maxWidth);
        if (shape.underline) {
          const lineWidth = Math.min(maxWidth, context.measureText(line).width);
          const lineLeft = textAlign === "center" ? x - lineWidth / 2 : textAlign === "right" ? x - lineWidth : x;
          context.beginPath(); context.moveTo(lineLeft, y + fontSize * 0.38); context.lineTo(lineLeft + lineWidth, y + fontSize * 0.38); context.strokeStyle = shape.textColor || "#fff"; context.lineWidth = Math.max(1, fontSize / 16); context.stroke();
        }
      };
      if (portrait) {
        const centerX = shape.x + shape.width / 2;
        const centerY = shape.y + shape.height / 2;
        context.save();
        context.translate(centerX, centerY);
        context.rotate(Math.PI / 2);
        context.fillRect(-shape.width / 2, -shape.height / 2, shape.width, shape.height);
        context.strokeStyle = shape.borderColor || shape.color || "#62dcff";
        context.strokeRect(-shape.width / 2, -shape.height / 2, shape.width, shape.height);
        context.globalAlpha = shape._playbackOpacity ?? 1;
        context.fillStyle = shape.textColor || "#fff";
        textLines.forEach((line, index) => drawTextLine(line, textXFor(-shape.width / 2 + 10, 0, shape.width / 2 - 10), firstLineY + index * lineHeight, Math.max(20, shape.width - 20)));
        context.restore();
      } else {
        context.fillRect(shape.x, shape.y, shape.width, shape.height);
        context.strokeStyle = shape.borderColor || shape.color || "#62dcff";
        context.strokeRect(shape.x, shape.y, shape.width, shape.height);
        context.globalAlpha = shape._playbackOpacity ?? 1;
        context.fillStyle = shape.textColor || "#fff";
        textLines.forEach((line, index) => drawTextLine(line, textXFor(shape.x + 10, shape.x + shape.width / 2, shape.x + shape.width - 10), shape.y + shape.height / 2 + firstLineY + index * lineHeight, Math.max(20, shape.width - 20)));
      }
    } else if (shape.type === "line") {
      context.beginPath(); context.moveTo(shape.x, shape.y); context.lineTo(shape.x + shape.width, shape.y + shape.height); context.stroke();
    } else if (shape.type === "circle") {
      context.beginPath(); context.ellipse(shape.x + shape.width / 2, shape.y + shape.height / 2, shape.width / 2, shape.height / 2, 0, 0, Math.PI * 2); context.fill(); context.stroke();
    } else if (shape.type === "triangle") {
      context.beginPath(); context.moveTo(shape.x + shape.width / 2, shape.y); context.lineTo(shape.x + shape.width, shape.y + shape.height); context.lineTo(shape.x, shape.y + shape.height); context.closePath(); context.fill(); context.stroke();
    } else {
      context.fillRect(shape.x, shape.y, shape.width, shape.height);
      context.strokeRect(shape.x, shape.y, shape.width, shape.height);
    }
  });
  context.globalAlpha = 1;
  (board.arrows || []).forEach((arrow) => {
    const angle = Math.atan2(arrow.to.y - arrow.from.y, arrow.to.x - arrow.from.x);
    context.setLineDash(arrow.lineStyle === "dashed" ? [14, 9] : arrow.lineStyle === "dotted" ? [2, 8] : []);
    context.globalAlpha = arrow.opacity ?? 1;
    context.strokeStyle = arrow.color || "#ffe36e"; context.fillStyle = arrow.color || "#ffe36e"; context.lineWidth = Number(arrow.strokeWidth) || 6; context.lineCap = "round";
    context.beginPath(); context.moveTo(arrow.from.x, arrow.from.y); context.lineTo(arrow.to.x, arrow.to.y); context.stroke();
    if (arrow.headStyle !== "none") {
      const headSize = Number(arrow.headSize) || 20;
      context.beginPath(); context.moveTo(arrow.to.x, arrow.to.y);
      context.lineTo(arrow.to.x - headSize * Math.cos(angle - Math.PI / 6), arrow.to.y - headSize * Math.sin(angle - Math.PI / 6));
      context.lineTo(arrow.to.x - headSize * Math.cos(angle + Math.PI / 6), arrow.to.y - headSize * Math.sin(angle + Math.PI / 6));
      if (arrow.headStyle === "open") {
        context.strokeStyle = arrow.color || "#ffe36e";
        context.lineWidth = Math.max(2, Number(arrow.strokeWidth) || 6);
        context.stroke();
      } else { context.closePath(); context.fill(); }
    }
    context.globalAlpha = 1;
  });
  context.setLineDash([]);
  board.players.forEach((player) => {
    context.beginPath(); context.arc(player.x, player.y + 3 * playerScale, 27 * playerScale, 0, Math.PI * 2); context.fillStyle = "#05223f70"; context.fill();
    context.beginPath(); context.arc(player.x, player.y, 24 * playerScale, 0, Math.PI * 2); context.fillStyle = player.color || (player.side === "home" ? "#1485dd" : "#e64b50");
    context.shadowColor = player.moving ? "#ffe36e" : "transparent"; context.shadowBlur = player.moving ? 22 : 0; context.fill(); context.shadowBlur = 0;
    context.strokeStyle = player.moving ? "#ffe36e" : "#f3f8ff"; context.lineWidth = player.moving ? 5 : 3; context.stroke();
    context.fillStyle = "#fff"; context.font = "bold 17px Arial, sans-serif"; context.textAlign = "center"; context.textBaseline = "middle";
    if (portrait) {
      context.save(); context.translate(player.x, player.y + 1); context.rotate(Math.PI / 2); context.fillText(String(player.number), 0, 0); context.restore();
    } else context.fillText(String(player.number), player.x, player.y + 1);
  });
  const ball = board.ball || { x: 500, y: 320 };
  context.beginPath(); context.arc(ball.x, ball.y + 2, 11, 0, Math.PI * 2); context.fillStyle = "#05223f70"; context.fill();
  context.beginPath(); context.arc(ball.x, ball.y, 10, 0, Math.PI * 2); context.fillStyle = "#f7f8f3"; context.fill();
  context.strokeStyle = "#142234"; context.lineWidth = 2; context.stroke();
  context.fillStyle = "#142234"; context.beginPath(); context.moveTo(ball.x, ball.y - 5); context.lineTo(ball.x + 4, ball.y - 2); context.lineTo(ball.x + 3, ball.y + 3); context.lineTo(ball.x - 3, ball.y + 3); context.lineTo(ball.x - 4, ball.y - 2); context.closePath(); context.fill();
  context.restore();
};


const drawSetPiecePdfFrame = (canvas, board, { playerScale = 0.9 } = {}) => {
  if (!canvas) return;
  const context = canvas.getContext("2d");
  if (!context) return;
  const width = canvas.width;
  const height = canvas.height;
  const fieldLeft = 30;
  const fieldRight = width - 30;
  const fieldTop = 120;
  const fieldBottom = height - 30;
  const sourceLeft = 41;
  const sourceWidth = 668;
  const sourceTop = 35;
  const sourceBottom = 690;
  const scaleX = (fieldRight - fieldLeft) / sourceWidth;
  const scaleY = (fieldBottom - fieldTop) / (sourceBottom - sourceTop);
  const mapPoint = (point) => ({
    x: fieldLeft + (1.171875 * Number(point?.y || 0) - sourceLeft) * scaleX,
    y: fieldTop + (1000 - Number(point?.x || 0) - sourceTop) * scaleY,
  });

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#24734e";
  context.fillRect(18, 22, width - 36, height - 44);
  for (let y = 30; y < height - 24; y += 96) {
    context.fillStyle = Math.floor((y - 30) / 96) % 2 === 0 ? "rgba(58,139,89,0.42)" : "rgba(17,91,60,0.18)";
    context.fillRect(20, y, width - 40, 48);
  }

  context.strokeStyle = "#e8f4e9";
  context.lineWidth = 5;
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(fieldLeft, fieldTop);
  context.lineTo(fieldRight, fieldTop);
  context.lineTo(fieldRight, fieldBottom);
  context.moveTo(fieldLeft, fieldTop);
  context.lineTo(fieldLeft, fieldBottom);
  context.stroke();
  const cornerRadius = Math.min(38, (fieldRight - fieldLeft) * 0.025);
  context.lineWidth = 4;
  context.beginPath();
  context.arc(fieldLeft, fieldTop, cornerRadius, 0, Math.PI / 2);
  context.moveTo(fieldRight, fieldTop + cornerRadius);
  context.arc(fieldRight, fieldTop, cornerRadius, Math.PI / 2, Math.PI);
  context.stroke();

  const penaltyLeft = fieldLeft + (259 - sourceLeft) * scaleX;
  const penaltyRight = fieldLeft + (491 - sourceLeft) * scaleX;
  const boxBottom = fieldTop + (209 - sourceTop) * scaleY;
  const sixLeft = fieldLeft + (316 - sourceLeft) * scaleX;
  const sixRight = fieldLeft + (434 - sourceLeft) * scaleX;
  const sixBottom = fieldTop + (103 - sourceTop) * scaleY;
  context.lineWidth = 5;
  context.strokeRect(penaltyLeft, fieldTop, penaltyRight - penaltyLeft, boxBottom - fieldTop);
  context.strokeRect(sixLeft, fieldTop, sixRight - sixLeft, sixBottom - fieldTop);
  const spot = { x: width / 2, y: fieldTop + (145 - sourceTop) * scaleY };
  context.fillStyle = "#e8f4e9";
  context.beginPath();
  context.arc(spot.x, spot.y, 5, 0, Math.PI * 2);
  context.fill();
  const dRadius = 126;
  const lineOffset = boxBottom - spot.y;
  const dHalfChord = Math.sqrt(Math.max(0, dRadius * dRadius - lineOffset * lineOffset));
  context.beginPath();
  context.arc(spot.x, spot.y, dRadius, Math.atan2(lineOffset, -dHalfChord), Math.atan2(lineOffset, dHalfChord), true);
  context.stroke();

  // Netted goal sits beyond the goal line, with a small runoff margin like the reference PDF.
  const goalWidth = Math.min(230, (fieldRight - fieldLeft) * 0.14);
  const goalLeft = width / 2 - goalWidth / 2;
  const goalTop = 0;
  const goalBottom = fieldTop + 1;
  context.fillStyle = "rgba(232,244,233,0.16)";
  context.fillRect(goalLeft, goalTop, goalWidth, goalBottom - goalTop);
  context.strokeStyle = "#e8f4e9";
  context.lineWidth = 5;
  context.strokeRect(goalLeft, goalTop, goalWidth, goalBottom - goalTop);
  context.strokeStyle = "rgba(232,244,233,0.62)";
  context.lineWidth = 1.5;
  context.beginPath();
  for (let x = goalLeft + 20; x < goalLeft + goalWidth; x += 20) {
    context.moveTo(x, goalTop);
    context.lineTo(x, goalBottom);
  }
  for (let y = goalTop + 18; y < goalBottom; y += 18) {
    context.moveTo(goalLeft, y);
    context.lineTo(goalLeft + goalWidth, y);
  }
  context.stroke();

  const opacityFor = (shape) => {
    const opacity = Number(shape.opacity);
    return Math.max(0, Math.min(1, Number.isFinite(opacity) ? opacity : shape.type === "text" ? 0.85 : 0.35)) * (shape._playbackOpacity ?? 1);
  };
  const sx = scaleX * 1.171875;
  const sy = scaleY;
  // Text boxes should stay visually consistent with the editor. The pitch is
  // deliberately stretched to fill a landscape PDF page, so applying sx/sy
  // independently to text-box dimensions makes them look unnaturally wide.
  const textScale = Math.min(sx, sy);
  const tx = fieldLeft - sourceLeft * scaleX;
  const ty = fieldTop + (965 * scaleY);
  context.save();
  context.setTransform(0, -sy, sx, 0, tx, ty);
  (board.shapes || []).forEach((shape) => {
    if (shape.type === "text") return;
    context.save();
    context.globalAlpha = opacityFor(shape);
    context.fillStyle = shape.color || "#62dcff";
    context.strokeStyle = shape.borderColor || shape.color || "#62dcff";
    context.lineWidth = Number(shape.strokeWidth) || 3;
    const x = Number(shape.x) || 0;
    const y = Number(shape.y) || 0;
    const w = Number(shape.width) || 0;
    const h = Number(shape.height) || 0;
    if (shape.type === "line") {
      context.beginPath(); context.moveTo(x, y); context.lineTo(x + w, y + h); context.stroke();
    } else if (shape.type === "circle") {
      context.beginPath(); context.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2); context.fill(); context.stroke();
    } else if (shape.type === "triangle") {
      context.beginPath(); context.moveTo(x + w / 2, y); context.lineTo(x + w, y + h); context.lineTo(x, y + h); context.closePath(); context.fill(); context.stroke();
    } else {
      context.fillRect(x, y, w, h); context.strokeRect(x, y, w, h);
    }
    context.restore();
  });
  context.restore();
  (board.shapes || []).filter((shape) => shape.type === "text").forEach((shape) => {
    const center = mapPoint({ x: (Number(shape.x) || 0) + (Number(shape.width) || 0) / 2, y: (Number(shape.y) || 0) + (Number(shape.height) || 0) / 2 });
    const boxWidth = Math.max(24, (Number(shape.width) || 0) * textScale);
    const boxHeight = Math.max(24, (Number(shape.height) || 0) * textScale);
    const left = center.x - boxWidth / 2;
    const top = center.y - boxHeight / 2;
    context.save();
    context.globalAlpha = opacityFor(shape);
    context.fillStyle = shape.backgroundColor || "#0c3769";
    context.strokeStyle = shape.borderColor || shape.color || "#62dcff";
    context.lineWidth = Math.max(1.5, (Number(shape.strokeWidth) || 3) * 1.4);
    context.fillRect(left, top, boxWidth, boxHeight);
    context.strokeRect(left, top, boxWidth, boxHeight);

    let fontSize = Math.max(10, (Number(shape.fontSize) || 22) * textScale);
    const padding = Math.max(6, fontSize * 0.35);
    const availableHeight = Math.max(16, boxHeight - padding * 2);
    let lines = [];
    for (let pass = 0; pass < 3; pass += 1) {
      context.font = `${shape.italic ? "italic " : ""}${shape.bold === false ? "normal" : "bold"} ${fontSize}px Arial, sans-serif`;
      lines = wrapTacticalCanvasText(context, shape.text || "", Math.max(20, boxWidth - padding * 2));
      const neededHeight = lines.length * fontSize * 1.2;
      if (neededHeight <= availableHeight) break;
      fontSize = Math.max(9, fontSize * (availableHeight / neededHeight));
    }
    const lineHeight = fontSize * 1.2;
    context.font = `${shape.italic ? "italic " : ""}${shape.bold === false ? "normal" : "bold"} ${fontSize}px Arial, sans-serif`;
    context.globalAlpha = (shape._playbackOpacity ?? 1);
    context.fillStyle = shape.textColor || "#ffffff";
    context.textAlign = ["left", "center", "right"].includes(shape.textAlign) ? shape.textAlign : "left";
    context.textBaseline = "middle";
    lines.forEach((line, index) => {
      const y = center.y + (index - (lines.length - 1) / 2) * lineHeight;
      const x = context.textAlign === "center" ? center.x : context.textAlign === "right" ? left + boxWidth - padding : left + padding;
      context.fillText(line, x, y, Math.max(20, boxWidth - padding * 2));
      if (shape.underline) {
        const measuredWidth = Math.min(boxWidth - padding * 2, context.measureText(line).width);
        const lineLeft = context.textAlign === "center" ? x - measuredWidth / 2 : context.textAlign === "right" ? x - measuredWidth : x;
        context.beginPath();
        context.moveTo(lineLeft, y + fontSize * 0.38);
        context.lineTo(lineLeft + measuredWidth, y + fontSize * 0.38);
        context.strokeStyle = shape.textColor || "#ffffff";
        context.lineWidth = Math.max(1, fontSize / 16);
        context.stroke();
      }
    });
    context.restore();
  });
  (board.arrows || []).forEach((arrow) => {
    const from = mapPoint(arrow.from || { x: 0, y: 0 });
    const to = mapPoint(arrow.to || arrow.from || { x: 0, y: 0 });
    const angle = Math.atan2(to.y - from.y, to.x - from.x);
    const color = arrow.color || "#ffe36e";
    const headSize = (Number(arrow.headSize) || 20) * 1.5;
    context.save();
    context.globalAlpha = arrow.opacity ?? 1;
    context.strokeStyle = color; context.fillStyle = color;
    context.lineWidth = (Number(arrow.strokeWidth) || 6) * 1.5;
    context.lineCap = "round";
    context.setLineDash(arrow.lineStyle === "dashed" ? [16, 11] : arrow.lineStyle === "dotted" ? [2, 9] : []);
    context.beginPath(); context.moveTo(from.x, from.y); context.lineTo(to.x, to.y); context.stroke();
    if (arrow.headStyle !== "none") {
      context.beginPath(); context.moveTo(to.x, to.y);
      context.lineTo(to.x - headSize * Math.cos(angle - Math.PI / 6), to.y - headSize * Math.sin(angle - Math.PI / 6));
      context.lineTo(to.x - headSize * Math.cos(angle + Math.PI / 6), to.y - headSize * Math.sin(angle + Math.PI / 6));
      if (arrow.headStyle === "open") context.stroke(); else { context.closePath(); context.fill(); }
    }
    context.restore();
  });

  (board.players || []).forEach((player) => {
    const point = mapPoint(player);
    const radius = 21 * playerScale;
    context.beginPath(); context.arc(point.x, point.y + 3, radius + 3, 0, Math.PI * 2); context.fillStyle = "rgba(5,34,63,0.42)"; context.fill();
    context.beginPath(); context.arc(point.x, point.y, radius, 0, Math.PI * 2); context.fillStyle = player.color || (player.side === "home" ? "#1485dd" : "#e64b50"); context.fill();
    context.strokeStyle = player.moving ? "#ffe36e" : "#f3f8ff"; context.lineWidth = player.moving ? 4 : 3; context.stroke();
    context.fillStyle = "#fff"; context.font = "bold 17px Arial, sans-serif"; context.textAlign = "center"; context.textBaseline = "middle";
    context.fillText(String(player.number), point.x, point.y + 1);
  });
  const ball = mapPoint(board.ball || { x: 500, y: 320 });
  // Keep the ball's visible centre on the exact transformed board coordinate;
  // a displaced shadow made it appear offset from where it was placed.
  context.beginPath(); context.arc(ball.x, ball.y, 10, 0, Math.PI * 2); context.fillStyle = "rgba(5,34,63,0.24)"; context.fill();
  context.beginPath(); context.arc(ball.x, ball.y, 9, 0, Math.PI * 2); context.fillStyle = "#f7f8f3"; context.fill();
  context.strokeStyle = "#142234"; context.lineWidth = 2; context.stroke();
  context.fillStyle = "#142234"; context.beginPath(); context.moveTo(ball.x, ball.y - 4); context.lineTo(ball.x + 3, ball.y - 2); context.lineTo(ball.x + 2, ball.y + 2); context.lineTo(ball.x - 2, ball.y + 2); context.lineTo(ball.x - 3, ball.y - 2); context.closePath(); context.fill();
};

function TacticalPad({ storageKey, homeTeamName, oppositionName }) {
  const clipStorageKey = `${storageKey}:movement-clips`;
  const pitchBoardsStorageKey = `${storageKey}:pitch-boards`;
  const savedTacticsStorageKey = `${storageKey}:saved-tactics`;
  const clipStorageKeyForMode = (viewMode) => `${clipStorageKey}:${viewMode}`;
  const stillSlidesStorageKeyForMode = (viewMode) => `${storageKey}:still-slides:${viewMode}`;
  const [board, setBoard] = useState(() => {
    try {
      const byMode = JSON.parse(localStorage.getItem(pitchBoardsStorageKey) || "{}");
      const legacy = JSON.parse(localStorage.getItem(storageKey) || "null");
      const savedMode = legacy?.pitchMode || "full";
      const saved = byMode?.[savedMode] || legacy;
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
  const [copiedPlayers, setCopiedPlayers] = useState(null);
  const [selectedShapeId, setSelectedShapeId] = useState(null);
  const [editingTextId, setEditingTextId] = useState(null);
  const [copiedGraphic, setCopiedGraphic] = useState(null);
  const [graphicClipboardStatus, setGraphicClipboardStatus] = useState("");
  const [selectedArrowId, setSelectedArrowId] = useState(null);
  const [enteringGraphicIds, setEnteringGraphicIds] = useState([]);
  const [newShapeOpacity, setNewShapeOpacity] = useState(0.35);
  const [newShapeColour, setNewShapeColour] = useState("#62dcff");
  const [newShapeBorderColour, setNewShapeBorderColour] = useState("#62dcff");
  const [newShapeStrokeWidth, setNewShapeStrokeWidth] = useState(4);
  const [newArrowLineStyle, setNewArrowLineStyle] = useState("solid");
  const [newArrowHeadStyle, setNewArrowHeadStyle] = useState("filled");
  const [newArrowHeadSize, setNewArrowHeadSize] = useState(20);
  const [newTextColor, setNewTextColor] = useState("#ffffff");
  const [newTextBoxColor, setNewTextBoxColor] = useState("#0c3769");
  const [dockPanel, setDockPanel] = useState("shapes");
  const [pitchMode, setPitchMode] = useState(() => board.pitchMode || "full");
  const [movementClips, setMovementClips] = useState(() => {
    try {
      const savedClips = JSON.parse(localStorage.getItem(clipStorageKeyForMode(board.pitchMode || "full")) || localStorage.getItem(clipStorageKey) || "[]");
      return Array.isArray(savedClips) ? savedClips : [];
    } catch { return []; }
  });
  const [stillSlides, setStillSlides] = useState(() => {
    try {
      const slides = JSON.parse(localStorage.getItem(stillSlidesStorageKeyForMode(board.pitchMode || "full")) || "[]");
      return Array.isArray(slides) ? slides : [];
    } catch { return []; }
  });
  const movementClipsRef = useRef(movementClips);
  const activeClipStorageKey = clipStorageKeyForMode(pitchMode);
  const activeStillSlidesStorageKey = stillSlidesStorageKeyForMode(pitchMode);
  const [savedTactics, setSavedTactics] = useState(() => {
    try { const saved = JSON.parse(localStorage.getItem(savedTacticsStorageKey) || "[]"); return Array.isArray(saved) ? saved : []; } catch { return []; }
  });
  const [selectedSavedTacticId, setSelectedSavedTacticId] = useState("");
  const [activeClipStart, setActiveClipStart] = useState(null);
  const [editingClipId, setEditingClipId] = useState(null);
  const [videoRecording, setVideoRecording] = useState(false);
  const [finishingVideo, setFinishingVideo] = useState(false);
  const [videoStatus, setVideoStatus] = useState("");
  const [playbackBoard, setPlaybackBoard] = useState(null);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [recordedVideo, setRecordedVideo] = useState(null);
  const pdfBannerStorageKey = `${storageKey}:pdf-banner`;
  const [pdfBanner, setPdfBanner] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(pdfBannerStorageKey) || "{}");
      return { title: "", fill: "#0f2d4e", text: "#ffffff", border: "#62dcff", pageBackground: "#eaf2f8", style: "rounded", ...saved };
    } catch { return { title: "", fill: "#0f2d4e", text: "#ffffff", border: "#62dcff", pageBackground: "#eaf2f8", style: "rounded" }; }
  });
  const pdfCoverStorageKey = `${storageKey}:pdf-cover`;
  const [pdfCover, setPdfCover] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(pdfCoverStorageKey) || "{}");
      return { title: "SET PLAYS", background: "#0f2d4e", text: "#ffffff", accent: "#62dcff", ...saved };
    } catch { return { title: "SET PLAYS", background: "#0f2d4e", text: "#ffffff", accent: "#62dcff" }; }
  });
  const pitchRef = useRef(null);
  const textEditorRef = useRef(null);
  const inlineTextRef = useRef(null);
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
  const loadedClipStorageKeyRef = useRef(activeClipStorageKey);
  const activeClipStorageKeyRef = useRef(activeClipStorageKey);

  const persistMovementClips = (nextOrUpdater) => {
    const next = typeof nextOrUpdater === "function" ? nextOrUpdater(movementClipsRef.current) : nextOrUpdater;
    movementClipsRef.current = next;
    setMovementClips(next);
    try { localStorage.setItem(activeClipStorageKeyRef.current, JSON.stringify(next)); } catch {}
    writeTeamAnalysisUpload(activeClipStorageKeyRef.current, next).catch(() => {});
  };

  const markGraphicEntering = (id) => setEnteringGraphicIds((current) => [...current, id]);
  const finishGraphicEntering = (id) => setEnteringGraphicIds((current) => current.filter((item) => item !== id));

  useEffect(() => { boardRef.current = board; }, [board]);

  useEffect(() => {
    if (pitchMode !== "box") return;
    setBoard((current) => {
      if ((current.setPieceBlueTwoFixVersion || 0) >= SET_PIECE_BLUE_TWO_FIX_VERSION) return current;
      const players = [...(current.players || [])];
      const blueTwoIndex = players.findIndex((player) => player.side === "home" && Number(player.number) === 2);
      if (blueTwoIndex >= 0) players[blueTwoIndex] = { ...players[blueTwoIndex], x: 860, y: 160 };
      else players.push({ id: `home-2-recovered-${Date.now()}`, side: "home", number: 2, x: 860, y: 160 });
      return { ...current, players, setPieceBlueTwoFixVersion: SET_PIECE_BLUE_TWO_FIX_VERSION };
    });
  }, [pitchMode, board]);

  useEffect(() => {
    const saveTimer = window.setTimeout(() => {
      try {
        localStorage.setItem(storageKey, JSON.stringify(board));
        const byMode = JSON.parse(localStorage.getItem(pitchBoardsStorageKey) || "{}");
        localStorage.setItem(pitchBoardsStorageKey, JSON.stringify({ ...byMode, [pitchMode]: board }));
      } catch {}
    }, 250);
    return () => window.clearTimeout(saveTimer);
  }, [board, storageKey, pitchMode, pitchBoardsStorageKey]);
  useEffect(() => {
    activeClipStorageKeyRef.current = activeClipStorageKey;
    try { localStorage.setItem(activeClipStorageKey, JSON.stringify(movementClips)); } catch {}
  }, [movementClips, activeClipStorageKey]);
  useEffect(() => {
    try { localStorage.setItem(activeStillSlidesStorageKey, JSON.stringify(stillSlides)); } catch {}
  }, [stillSlides, activeStillSlidesStorageKey]);
  useEffect(() => {
    let cancelled = false;
    if (loadedClipStorageKeyRef.current !== activeClipStorageKey) {
      loadedClipStorageKeyRef.current = activeClipStorageKey;
      let clipsForThisUser = [];
      try {
        const stored = JSON.parse(localStorage.getItem(activeClipStorageKey) || (pitchMode === "full" ? localStorage.getItem(clipStorageKey) : "[]") || "[]");
        if (Array.isArray(stored)) clipsForThisUser = stored;
      } catch {}
      movementClipsRef.current = clipsForThisUser;
      setMovementClips(clipsForThisUser);
    }
    readTeamAnalysisUpload(activeClipStorageKey).then((savedClips) => {
      if (cancelled || movementClipsRef.current.length || !Array.isArray(savedClips) || !savedClips.length) return;
      movementClipsRef.current = savedClips;
      setMovementClips(savedClips);
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [activeClipStorageKey, pitchMode, clipStorageKey]);
  useEffect(() => {
    try { localStorage.setItem(savedTacticsStorageKey, JSON.stringify(savedTactics)); } catch {}
  }, [savedTactics, savedTacticsStorageKey]);
  useEffect(() => {
    try { localStorage.setItem(pdfBannerStorageKey, JSON.stringify(pdfBanner)); } catch {}
  }, [pdfBanner, pdfBannerStorageKey]);
  useEffect(() => {
    try { localStorage.setItem(pdfCoverStorageKey, JSON.stringify(pdfCover)); } catch {}
  }, [pdfCover, pdfCoverStorageKey]);
  useEffect(() => () => {
    if (videoFrameRef.current) window.cancelAnimationFrame(videoFrameRef.current);
    if (clipPreviewFrameRef.current) window.cancelAnimationFrame(clipPreviewFrameRef.current);
    if (videoFinishTimerRef.current) window.clearTimeout(videoFinishTimerRef.current);
    const recorder = videoRecorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);
  useEffect(() => () => { if (recordedVideo?.url) URL.revokeObjectURL(recordedVideo.url); }, [recordedVideo]);

  const writePitchModeDraft = (viewMode, snapshot) => {
    try {
      const byMode = JSON.parse(localStorage.getItem(pitchBoardsStorageKey) || "{}");
      localStorage.setItem(pitchBoardsStorageKey, JSON.stringify({ ...byMode, [viewMode]: snapshot }));
      localStorage.setItem(storageKey, JSON.stringify(snapshot));
    } catch {}
  };
  const changePitchMode = (nextMode) => {
    if (!nextMode || nextMode === pitchMode || activeClipStart || finishingVideo) return;
    const currentBoard = { ...boardRef.current, pitchMode };
    writePitchModeDraft(pitchMode, currentBoard);
    let byMode = {};
    try { byMode = JSON.parse(localStorage.getItem(pitchBoardsStorageKey) || "{}"); } catch {}
    const nextBoard = byMode?.[nextMode]?.players?.length
      ? { ...freshTacticalBoard(homeTeamName, oppositionName), ...byMode[nextMode], pitchMode: nextMode }
      : { ...freshTacticalBoard(homeTeamName, oppositionName), pitchMode: nextMode };
    setPitchMode(nextMode);
    setBoard(nextBoard);
    boardRef.current = nextBoard;
    const nextClipKey = clipStorageKeyForMode(nextMode);
    let nextClips = [];
    try {
      const stored = JSON.parse(localStorage.getItem(nextClipKey) || "[]");
      if (Array.isArray(stored)) nextClips = stored;
    } catch {}
    movementClipsRef.current = nextClips;
    loadedClipStorageKeyRef.current = nextClipKey;
    activeClipStorageKeyRef.current = nextClipKey;
    setMovementClips(nextClips);
    let nextStillSlides = [];
    try {
      const storedSlides = JSON.parse(localStorage.getItem(stillSlidesStorageKeyForMode(nextMode)) || "[]");
      if (Array.isArray(storedSlides)) nextStillSlides = storedSlides;
    } catch {}
    setStillSlides(nextStillSlides);
    setSelectedTokens([]);
    setSelectedShapeId(null);
    setSelectedArrowId(null);
    setEditingTextId(null);
  };
  const saveCurrentTactic = (quiet = false) => {
    const name = (board.name || "").trim() || "Untitled tactic";
    const existing = savedTactics.find((item) => String(item.name || "").toLowerCase() === name.toLowerCase() && item.pitchMode === pitchMode);
    const tactic = {
      id: existing?.id || `tactic-${Date.now()}`,
      name,
      pitchMode,
      board: { ...board, name, pitchMode },
      movementClips: JSON.parse(JSON.stringify(movementClips)),
      stillSlides: JSON.parse(JSON.stringify(stillSlides)),
      savedAt: new Date().toISOString(),
    };
    setSavedTactics((current) => existing
      ? current.map((item) => item.id === existing.id ? tactic : item)
      : [...current, tactic]);
    setSelectedSavedTacticId(tactic.id);
    if (!quiet) setVideoStatus(`Saved “${name}” with ${movementClips.length} movement clip${movementClips.length === 1 ? "" : "s"} and ${stillSlides.length} still slide${stillSlides.length === 1 ? "" : "s"}.`);
    return tactic;
  };
  const loadSavedTactic = () => {
    const tactic = savedTactics.find((item) => item.id === selectedSavedTacticId);
    if (!tactic?.board) return;
    writePitchModeDraft(pitchMode, { ...boardRef.current, pitchMode });
    const nextMode = tactic.pitchMode || tactic.board.pitchMode || "full";
    const nextBoard = { ...freshTacticalBoard(homeTeamName, oppositionName), ...tactic.board, pitchMode: nextMode };
    const nextClips = Array.isArray(tactic.movementClips) ? JSON.parse(JSON.stringify(tactic.movementClips)) : [];
    const nextStillSlides = Array.isArray(tactic.stillSlides) ? JSON.parse(JSON.stringify(tactic.stillSlides)) : [];
    setPitchMode(nextMode);
    setBoard(nextBoard);
    boardRef.current = nextBoard;
    setMovementClips(nextClips);
    movementClipsRef.current = nextClips;
    setStillSlides(nextStillSlides);
    const nextClipKey = clipStorageKeyForMode(nextMode);
    loadedClipStorageKeyRef.current = nextClipKey;
    activeClipStorageKeyRef.current = nextClipKey;
    try { localStorage.setItem(nextClipKey, JSON.stringify(nextClips)); } catch {}
    try { localStorage.setItem(stillSlidesStorageKeyForMode(nextMode), JSON.stringify(nextStillSlides)); } catch {}
    setSelectedTokens([]);
    setSelectedShapeId(null);
    setSelectedArrowId(null);
    setEditingTextId(null);
    setActiveClipStart(null);
    setEditingClipId(null);
    setVideoStatus(`Loaded “${tactic.name}”.`);
  };
  const startNewTactic = () => {
    saveCurrentTactic(true);
    const blankBoard = { ...freshTacticalBoard(homeTeamName, oppositionName), pitchMode };
    setBoard(blankBoard);
    boardRef.current = blankBoard;
    setMovementClips([]);
    movementClipsRef.current = [];
    setStillSlides([]);
    setActiveClipStart(null);
    setEditingClipId(null);
    setSelectedTokens([]);
    setSelectedShapeId(null);
    setSelectedArrowId(null);
    setEditingTextId(null);
    setVideoStatus("Started a new tactic. Your previous tactic and clips are saved.");
  };
  const deleteSavedTactic = () => {
    const tactic = savedTactics.find((item) => item.id === selectedSavedTacticId);
    if (!tactic) return;
    setSavedTactics((current) => current.filter((item) => item.id !== tactic.id));
    setSelectedSavedTacticId("");
    setVideoStatus(`Deleted saved tactic “${tactic.name}”. The open board is unchanged.`);
  };

  const pointFromEvent = (event) => {
    const rect = pitchRef.current?.getBoundingClientRect();
    if (!rect) return { x: 500, y: 320 };
    const activePitchMode = videoPlaying ? (playbackBoard?.pitchMode || pitchMode) : pitchMode;
    const crop = activePitchMode === "box" ? BOX_PITCH_VIEW : { x: 0, y: 0, width: 750, height: activePitchMode === "half" ? HALF_PITCH_VIEW_HEIGHT : 1000 };
    const viewX = activePitchMode === "full" ? ((event.clientX - rect.left) / rect.width) * 1000 : crop.x + ((event.clientX - rect.left) / rect.width) * crop.width;
    const viewY = activePitchMode === "full" ? ((event.clientY - rect.top) / rect.height) * 640 : crop.y + ((event.clientY - rect.top) / rect.height) * crop.height;
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
      setSelectedShapeId(null);
      setSelectedArrowId(null);
      setSelectedTokens((current) => current.includes(player.id) ? current.filter((id) => id !== player.id) : [...current, player.id]);
      return;
    }
    setSelectedShapeId(null);
    setSelectedArrowId(null);
    setEditingTextId(null);
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
    setSelectedTokens([]);
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
    setSelectedTokens([]);
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
    setSelectedTokens([]);
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
        const newArrow = { id: `arrow-${Date.now()}`, from: arrowStart, to: end, color: newShapeColour, opacity: newShapeOpacity, strokeWidth: newShapeStrokeWidth, lineStyle: newArrowLineStyle, headStyle: newArrowHeadStyle, headSize: newArrowHeadSize };
        setBoard((current) => ({ ...current, arrows: [...current.arrows, newArrow] }));
        markGraphicEntering(newArrow.id);
        setSelectedArrowId(newArrow.id);
        setSelectedShapeId(null);
      }
      if (distance > 12 && ["rectangle", "square", "circle", "triangle", "line"].includes(mode)) {
        const box = shapeForDrag(arrowStart, end, mode);
        const newShape = { id: `shape-${Date.now()}`, type: mode, ...box, color: mode === "line" ? newShapeBorderColour : newShapeColour, borderColor: newShapeBorderColour, opacity: newShapeOpacity, strokeWidth: newShapeStrokeWidth };
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
        const newTextBox = { id: `text-${Date.now()}`, type: "text", x, y, width, height, text: "Text", fontSize: 22, color: newShapeColour, borderColor: newShapeBorderColour, textColor: newTextColor, backgroundColor: newTextBoxColor, opacity: 0.85 };
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
  const removeSelectedPlayer = useCallback(() => {
    if (!selectedTokens.length) return;
    setBoard((current) => ({ ...current, players: current.players.filter((player) => !selectedTokens.includes(player.id)) }));
    setSelectedTokens([]);
  }, [selectedTokens]);
  const selectedPlayers = board.players.filter((player) => selectedTokens.includes(player.id));
  const updateSelectedPlayers = (changes) => {
    if (!selectedTokens.length) return;
    setBoard((current) => ({ ...current, players: current.players.map((player) => selectedTokens.includes(player.id) ? { ...player, ...changes } : player) }));
  };
  const copySelectedPlayers = useCallback(() => {
    const chosenPlayers = boardRef.current.players.filter((player) => selectedTokens.includes(player.id));
    if (!chosenPlayers.length) return;
    setCopiedPlayers(JSON.parse(JSON.stringify(chosenPlayers)));
    setCopiedGraphic(null);
    setGraphicClipboardStatus(`${chosenPlayers.length === 1 ? "Disc" : `${chosenPlayers.length} discs`} copied. Paste to duplicate ${chosenPlayers.length === 1 ? "it" : "them"}.`);
  }, [selectedTokens]);
  const pasteCopiedPlayers = useCallback(() => {
    if (!copiedPlayers?.length) return;
    const selectedXs = copiedPlayers.map((player) => player.x);
    const selectedYs = copiedPlayers.map((player) => player.y);
    const dx = Math.max(...selectedXs) + 24 <= 976 ? 24 : -24;
    const dy = Math.max(...selectedYs) + 24 <= 616 ? 24 : -24;
    const duplicates = copiedPlayers.map((player, index) => ({
      ...player,
      id: `${player.side}-copy-${Date.now()}-${index}-${Math.random().toString(36).slice(2, 6)}`,
      x: Math.max(24, Math.min(976, player.x + dx)),
      y: Math.max(24, Math.min(616, player.y + dy)),
    }));
    setBoard((current) => ({ ...current, players: [...current.players, ...duplicates] }));
    setSelectedTokens(duplicates.map((player) => player.id));
    setSelectedShapeId(null);
    setSelectedArrowId(null);
    setMode("move");
    setGraphicClipboardStatus(`${duplicates.length === 1 ? "Disc" : `${duplicates.length} discs`} pasted. You can paste again to make another copy.`);
  }, [copiedPlayers]);
  const selectedShape = (board.shapes || []).find((shape) => shape.id === selectedShapeId);
  const selectedArrow = (board.arrows || []).find((arrow) => arrow.id === selectedArrowId);
  const selectedGraphic = selectedShape || selectedArrow;
  const copySelectedGraphic = useCallback(() => {
    if (!selectedGraphic) return;
    const kind = selectedShape ? "shape" : "arrow";
    setCopiedGraphic({ kind, item: JSON.parse(JSON.stringify(selectedGraphic)) });
    setCopiedPlayers(null);
    setGraphicClipboardStatus(`${kind === "arrow" ? "Arrow" : "Shape"} copied. Paste to make a duplicate.`);
  }, [selectedGraphic, selectedShape]);
  const pasteCopiedGraphic = useCallback(() => {
    if (!copiedGraphic) return;
    const newId = `${copiedGraphic.kind}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    if (copiedGraphic.kind === "shape") {
      const source = copiedGraphic.item;
      const width = Math.max(1, Number(source.width) || 220);
      const height = Math.max(1, Number(source.height) || 48);
      const pasted = {
        ...source,
        id: newId,
        x: Math.max(0, Math.min(1000 - width, Number(source.x) + 24)),
        y: Math.max(0, Math.min(640 - height, Number(source.y) + 24)),
      };
      setBoard((current) => ({ ...current, shapes: [...(current.shapes || []), pasted] }));
      setSelectedShapeId(pasted.id);
      setSelectedArrowId(null);
    } else {
      const source = copiedGraphic.item;
      const minX = Math.min(source.from.x, source.to.x);
      const maxX = Math.max(source.from.x, source.to.x);
      const minY = Math.min(source.from.y, source.to.y);
      const maxY = Math.max(source.from.y, source.to.y);
      let dx = Math.min(24, 1000 - maxX);
      let dy = Math.min(24, 640 - maxY);
      if (dx === 0) dx = -Math.min(24, minX);
      if (dy === 0) dy = -Math.min(24, minY);
      const pasted = {
        ...source,
        id: newId,
        from: { x: source.from.x + dx, y: source.from.y + dy },
        to: { x: source.to.x + dx, y: source.to.y + dy },
      };
      setBoard((current) => ({ ...current, arrows: [...(current.arrows || []), pasted] }));
      setSelectedArrowId(pasted.id);
      setSelectedShapeId(null);
    }
    setEnteringGraphicIds((current) => [...current, newId]);
    setEditingTextId(null);
    setMode("move");
    setGraphicClipboardStatus("Duplicate pasted. You can paste again to make another copy.");
  }, [copiedGraphic]);
  useEffect(() => {
    const handleGraphicShortcuts = (event) => {
      if (finishingVideo) return;
      const key = event.key.toLowerCase();
      const targetTag = event.target?.tagName?.toLowerCase();
      const isTextEntry = ["input", "textarea"].includes(targetTag) || event.target?.isContentEditable;
      const isTextField = isTextEntry || event.target?.getAttribute?.("data-tactical-text-input") === "true";
      if (["delete", "backspace"].includes(key) && !isTextField) {
        if (selectedTokens.length) {
          event.preventDefault();
          removeSelectedPlayer();
        } else if (selectedShape) {
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
      if (isTextEntry) return;
      if (key === "c" && selectedTokens.length) {
        event.preventDefault();
        copySelectedPlayers();
      } else if (key === "c" && selectedGraphic) {
        event.preventDefault();
        copySelectedGraphic();
      } else if (key === "v" && copiedPlayers?.length) {
        event.preventDefault();
        pasteCopiedPlayers();
      } else if (key === "v" && copiedGraphic) {
        event.preventDefault();
        pasteCopiedGraphic();
      }
    };
    document.addEventListener("keydown", handleGraphicShortcuts, true);
    return () => document.removeEventListener("keydown", handleGraphicShortcuts, true);
  }, [finishingVideo, selectedShape, selectedArrow, selectedGraphic, selectedTokens, copiedPlayers, copiedGraphic, copySelectedPlayers, pasteCopiedPlayers, copySelectedGraphic, pasteCopiedGraphic, removeSelectedPlayer]);
  const updateSelectedShape = (changes) => {
    if (!selectedShapeId) return;
    setBoard((current) => ({ ...current, shapes: (current.shapes || []).map((shape) => shape.id === selectedShapeId ? { ...shape, ...changes } : shape) }));
  };
  const startInlineTextEdit = (shape, replacementText = null) => {
    setSelectedShapeId(shape.id);
    setSelectedArrowId(null);
    setDockPanel("text");
    if (replacementText !== null) {
      setBoard((current) => ({ ...current, shapes: (current.shapes || []).map((item) => item.id === shape.id ? { ...item, text: replacementText } : item) }));
    }
    setEditingTextId(shape.id);
    const nextText = replacementText ?? shape.text ?? "";
    window.requestAnimationFrame(() => {
      inlineTextRef.current?.focus();
      inlineTextRef.current?.setSelectionRange(nextText.length, nextText.length);
    });
  };
  const insertBulletIntoSelectedText = () => {
    if (selectedShape?.type !== "text") return;
    const currentText = selectedShape.text || "";
    const textarea = textEditorRef.current;
    const start = textarea?.selectionStart ?? currentText.length;
    const end = textarea?.selectionEnd ?? currentText.length;
    const before = currentText.slice(0, start);
    const after = currentText.slice(end);
    const prefix = before && !before.endsWith("\n") ? "\n" : "";
    const inserted = `${prefix}• `;
    updateSelectedShape({ text: `${before}${inserted}${after}` });
    const caret = before.length + inserted.length;
    window.requestAnimationFrame(() => {
      textEditorRef.current?.focus();
      textEditorRef.current?.setSelectionRange(caret, caret);
    });
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
      return JSON.stringify(destination) !== JSON.stringify(origin);
    })).size;
    const fromArrows = from.arrows || [];
    const toArrows = to.arrows || [];
    const movedArrows = new Set([...fromArrows, ...toArrows].filter((arrow, index, all) => all.findIndex((item) => item.id === arrow.id) === index).filter((arrow) => {
      const origin = fromArrows.find((item) => item.id === arrow.id);
      const destination = toArrows.find((item) => item.id === arrow.id);
      if (!origin || !destination) return true;
      return JSON.stringify(destination) !== JSON.stringify(origin);
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
    const existingClip = editingClipId ? movementClipsRef.current.find((item) => item.id === editingClipId) : null;
    const clip = { id: editingClipId || `movement-${Date.now()}`, from, to, ballPath, movedCount, durationMs: existingClip?.durationMs || tacticalClipDuration({ ballPath }) };
    if (editingClipId) persistMovementClips((current) => current.map((item) => item.id === editingClipId ? clip : item));
    else persistMovementClips((current) => [...current, clip]);
    const savedClipNumber = editingClipId ? movementClipsRef.current.findIndex((item) => item.id === editingClipId) + 1 : movementClipsRef.current.length;
    activeClipStartRef.current = null;
    activeClipBallPathRef.current = [];
    setActiveClipStart(null);
    setEditingClipId(null);
    setVideoStatus(editingClipId ? `Movement clip ${savedClipNumber} updated.` : `Movement clip ${movementClipsRef.current.length} saved. Add another clip or play all clips.`);
  };
  const cancelMovementClip = () => {
    if (editingClipId) {
      const originalClip = movementClipsRef.current.find((clip) => clip.id === editingClipId);
      if (originalClip) setBoard(originalClip.to);
    }
    activeClipStartRef.current = null;
    activeClipBallPathRef.current = [];
    setActiveClipStart(null);
    setEditingClipId(null);
    setVideoStatus(editingClipId ? "Clip edits cancelled." : "Movement clip cancelled.");
  };
  const editMovementClip = (clip, clipNumber) => {
    if (finishingVideo || activeClipStartRef.current) return;
    const from = JSON.parse(JSON.stringify(clip.from));
    const to = JSON.parse(JSON.stringify(clip.to));
    setBoard(to);
    setPitchMode(to.pitchMode || pitchMode);
    activeClipStartRef.current = from;
    activeClipBallPathRef.current = JSON.parse(JSON.stringify(clip.ballPath || []));
    setActiveClipStart(from);
    setEditingClipId(clip.id);
    setDockPanel("animation");
    setSelectedTokens([]);
    setSelectedShapeId(null);
    setSelectedArrowId(null);
    setVideoStatus(`Editing clip ${clipNumber}. Change the final positions or drawings, then save this clip.`);
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
    const exportPitchMode = pitchMode;
    canvas.width = exportPitchMode === "full" ? 1280 : exportPitchMode === "box" ? Math.round(1280 * BOX_PITCH_VIEW.width / BOX_PITCH_VIEW.height) : 960;
    canvas.height = exportPitchMode === "full" ? 819 : exportPitchMode === "half" ? 832 : 1280;
    drawTacticalVideoFrame(canvas, { ...clips[0].from, pitchMode: exportPitchMode }, exportPitchMode);
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
        const orientedFinalFrame = { ...finalFrame, pitchMode: exportPitchMode };
        drawTacticalVideoFrame(canvas, orientedFinalFrame, exportPitchMode);
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
        pitchMode: exportPitchMode,
        name: boardRef.current.name,
        homeName: boardRef.current.homeName,
        awayName: boardRef.current.awayName,
        arrows,
        shapes,
        players,
        ball,
      };
      drawTacticalVideoFrame(canvas, frameBoard, exportPitchMode);
      setPlaybackBoard(frameBoard);
      videoFrameRef.current = window.requestAnimationFrame(drawFrame);
    };
    videoFrameRef.current = window.requestAnimationFrame(drawFrame);
  };
  const addStillSlide = () => {
    if (finishingVideo || activeClipStart) return;
    const snapshot = JSON.parse(JSON.stringify({ ...boardRef.current, pitchMode }));
    const slideTitle = `Slide ${stillSlides.length + 1}`;
    const slide = { id: `still-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, title: slideTitle, bannerTitle: slideTitle, board: snapshot };
    setStillSlides((current) => [...current, slide]);
    setVideoStatus(`Added still slide ${stillSlides.length + 1}. Capture another setup whenever you’re ready.`);
  };
  const renameStillSlide = (slideId, title) => {
    setStillSlides((current) => current.map((slide) => slide.id === slideId ? { ...slide, title, bannerTitle: title } : slide));
  };
  const loadStillSlide = (slide) => {
    if (!slide?.board || finishingVideo || activeClipStart) return;
    const restored = { ...freshTacticalBoard(homeTeamName, oppositionName), ...slide.board, pitchMode };
    setBoard(restored);
    boardRef.current = restored;
    setSelectedTokens([]);
    setSelectedShapeId(null);
    setSelectedArrowId(null);
    setVideoStatus(`Loaded “${slide.title || "still slide"}” onto the pitch.`);
  };
  const replaceStillSlide = (slideId) => {
    const snapshot = JSON.parse(JSON.stringify({ ...boardRef.current, pitchMode }));
    setStillSlides((current) => current.map((slide) => slide.id === slideId ? { ...slide, board: snapshot } : slide));
    setVideoStatus("Still slide updated from the current pitch.");
  };
  const downloadClipStillsPdf = async () => {
    if (finishingVideo || activeClipStart) return;
    const exportPitchMode = pitchMode;
    const isSetPiecePdf = exportPitchMode === "box";
    const canvas = videoCanvasRef.current;
    if (!canvas) {
      setVideoStatus("Could not prepare the pitch images for the PDF.");
      return;
    }
    canvas.width = isSetPiecePdf ? 1814 : exportPitchMode === "full" ? 1280 : 960;
    canvas.height = isSetPiecePdf ? 1080 : exportPitchMode === "full" ? 819 : 832;
    const frames = stillSlides.length
      ? stillSlides.map((slide, index) => ({ label: slide.title || `Slide ${index + 1}`, bannerTitle: slide.bannerTitle || slide.title || `Slide ${index + 1}`, board: { ...slide.board, pitchMode: exportPitchMode } }))
      : [{ label: "Current tactic", board: { ...boardRef.current, pitchMode: exportPitchMode } }];
    movementClips.forEach((clip, index) => {
      frames.push({ label: `Clip ${index + 1} · Start`, board: { ...clip.from, pitchMode: exportPitchMode } });
      frames.push({ label: `Clip ${index + 1} · Finish`, board: { ...clip.to, pitchMode: exportPitchMode } });
    });
    const orientation = exportPitchMode === "half" ? "portrait" : "landscape";
    const pdf = new jsPDF(orientation, "mm", "a4");
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const pageSideMargin = isSetPiecePdf ? 6 : 12;
    const imageMaxWidth = pageWidth - pageSideMargin * 2;
    const pitchTop = isSetPiecePdf ? 30 : 39;
    const imageMaxHeight = pageHeight - pitchTop - (isSetPiecePdf ? 6 : 8);
    const toRgb = (colour, fallback) => {
      const value = /^#[\da-f]{6}$/i.test(colour || "") ? colour : fallback;
      return [1, 3, 5].map((index) => parseInt(value.slice(index, index + 2), 16));
    };
    const loadBadgeData = async (team) => {
      const path = clubBadgePathFor(team);
      if (!path) return null;
      try {
        const url = supabase.storage.from("Club Badges").getPublicUrl(path).data?.publicUrl;
        if (!url) return null;
        const response = await fetch(url);
        if (!response.ok) return null;
        const blob = await response.blob();
        return await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : null);
          reader.onerror = () => resolve(null);
          reader.readAsDataURL(blob);
        });
      } catch { return null; }
    };
    const homeTeam = boardRef.current.homeName || homeTeamName || "Home Team";
    const awayTeam = boardRef.current.awayName || oppositionName || "Opposition";
    if (isSetPiecePdf) {
      const [homeBadge, awayBadge] = await Promise.all([loadBadgeData(homeTeam), loadBadgeData(awayTeam)]);
      pdf.setFillColor(...toRgb(pdfCover.background, "#0f2d4e"));
      pdf.rect(0, 0, pageWidth, pageHeight, "F");
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(27);
      pdf.setTextColor(...toRgb(pdfCover.text, "#ffffff"));
      pdf.text((pdfCover.title.trim() || "SET PLAYS").toUpperCase(), pageWidth / 2, 32, { align: "center", maxWidth: pageWidth - 24 });
      pdf.setDrawColor(...toRgb(pdfCover.accent, "#62dcff"));
      pdf.setLineWidth(1.1);
      pdf.line(pageWidth * 0.32, 40, pageWidth * 0.68, 40);
      if (homeBadge) pdf.addImage(homeBadge, "PNG", pageWidth / 2 - 27, 50, 54, 54);
      pdf.setFontSize(22);
      pdf.setTextColor(...toRgb(pdfCover.text, "#ffffff"));
      pdf.text(homeTeam, pageWidth / 2, 116, { align: "center", maxWidth: pageWidth - 30 });
      pdf.setDrawColor(...toRgb(pdfCover.accent, "#62dcff"));
      pdf.setLineWidth(0.6);
      pdf.line(pageWidth * 0.36, 128, pageWidth * 0.64, 128);
      pdf.setFontSize(18);
      pdf.setTextColor(...toRgb(pdfCover.accent, "#62dcff"));
      pdf.text("VS", pageWidth / 2, 145, { align: "center" });
      pdf.setFontSize(18);
      pdf.setTextColor(...toRgb(pdfCover.text, "#ffffff"));
      const opponentWidth = pdf.getTextWidth(awayTeam);
      const opponentTextX = pageWidth / 2 - (opponentWidth + (awayBadge ? 20 : 0)) / 2;
      pdf.text(awayTeam, opponentTextX, 166, { align: "left", maxWidth: pageWidth - 40 });
      if (awayBadge) pdf.addImage(awayBadge, "PNG", opponentTextX + opponentWidth + 5, 155, 14, 14);
    }
    frames.forEach((frame, index) => {
      if (isSetPiecePdf || index > 0) pdf.addPage("a4", orientation);
      pdf.setFillColor(...toRgb(pdfBanner.pageBackground, "#eaf2f8"));
      pdf.rect(0, 0, pageWidth, pageHeight, "F");
      if (isSetPiecePdf) drawSetPiecePdfFrame(canvas, frame.board, { playerScale: 0.9 });
      else drawTacticalVideoFrame(canvas, frame.board, exportPitchMode, { playerScale: 0.9 });
      const bannerX = pageSideMargin;
      const bannerY = isSetPiecePdf ? 8 : 20;
      const bannerWidth = imageMaxWidth;
      const bannerHeight = isSetPiecePdf ? 22 : 16;
      if (!isSetPiecePdf) {
        pdf.setFont("helvetica", "normal");
        pdf.setFontSize(8);
        pdf.setTextColor(35, 55, 75);
        pdf.text(`${boardRef.current.homeName || homeTeamName} vs ${boardRef.current.awayName || oppositionName}`, pageWidth / 2, 9, { align: "center", maxWidth: bannerWidth });
        pdf.setFontSize(9);
        pdf.text(frame.label, pageWidth / 2, 15, { align: "center", maxWidth: bannerWidth });
      }
      pdf.setFillColor(...toRgb(pdfBanner.fill, "#0f2d4e"));
      pdf.setDrawColor(...toRgb(pdfBanner.border, "#62dcff"));
      pdf.setLineWidth(0.7);
      if (pdfBanner.style === "square") pdf.rect(bannerX, bannerY, bannerWidth, bannerHeight, "FD");
      else pdf.roundedRect(bannerX, bannerY, bannerWidth, bannerHeight, 2, 2, "FD");
      pdf.setFont("helvetica", "bold");
      const tacticTitle = (frame.bannerTitle || "").trim() || pdfBanner.title.trim() || boardRef.current.name || "Tactic";
      const titleBaseSize = 16;
      pdf.setFontSize(Math.max(11, Math.min(titleBaseSize, titleBaseSize - Math.max(0, tacticTitle.length - 28) * 0.18)));
      pdf.setTextColor(...toRgb(pdfBanner.text, "#ffffff"));
      pdf.text(tacticTitle, pageWidth / 2, bannerY + bannerHeight * 0.67, { align: "center", maxWidth: bannerWidth - 12 });
      const image = canvas.toDataURL("image/png");
      const scale = Math.min(imageMaxWidth / canvas.width, imageMaxHeight / canvas.height);
      const imageWidth = canvas.width * scale;
      const imageHeight = canvas.height * scale;
      pdf.addImage(image, "PNG", (pageWidth - imageWidth) / 2, pitchTop + Math.max(0, (imageMaxHeight - imageHeight) / 2), imageWidth, imageHeight);
    });
    const filename = (boardRef.current.name || "tactical-clips").trim().replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "tactical-clips";
    pdf.save(`${filename}-stills.pdf`);
    setVideoStatus(`PDF ready with ${stillSlides.length} still slide${stillSlides.length === 1 ? "" : "s"} and ${movementClips.length} clip${movementClips.length === 1 ? "" : "s"} as still images.`);
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
        <svg ref={pitchRef} className="sr-tactical-pitch" preserveAspectRatio="xMidYMid meet" style={{ width: pitchViewMode === "full" ? undefined : "auto", height: pitchViewMode === "full" ? undefined : pitchViewMode === "half" ? "min(56vh, 580px)" : "min(50vh, 520px)", minHeight: pitchViewMode === "full" ? undefined : 0, maxWidth: pitchViewMode === "full" ? undefined : "100%", maxHeight: pitchViewMode === "full" ? undefined : pitchViewMode === "half" ? "min(56vh, 580px)" : "min(50vh, 520px)", aspectRatio: pitchViewMode === "full" ? "25 / 16" : pitchViewMode === "half" ? `750 / ${HALF_PITCH_VIEW_HEIGHT}` : `${BOX_PITCH_VIEW.width} / ${BOX_PITCH_VIEW.height}` }} viewBox={pitchViewMode === "full" ? "0 0 1000 640" : pitchViewMode === "half" ? `0 0 750 ${HALF_PITCH_VIEW_HEIGHT}` : `${BOX_PITCH_VIEW.x} ${BOX_PITCH_VIEW.y} ${BOX_PITCH_VIEW.width} ${BOX_PITCH_VIEW.height}`} role="img" aria-label={`${pitchViewMode === "full" ? "Full" : pitchViewMode === "half" ? "Half pitch plus a little beyond halfway" : "18-yard box set-piece"} tactical football pitch. Drag the discs to position players.`} onPointerDown={handlePitchPointerDown} onPointerMove={handlePitchPointerMove} onPointerUp={handlePitchPointerUp} onPointerCancel={handlePitchPointerUp}>
          <defs><pattern id="sr-tactical-grass-stripes" width="80" height="80" patternUnits="userSpaceOnUse"><rect width="80" height="80" fill="#24734e" /><rect width="80" height="40" fill="#3a8b59" opacity="0.38" /></pattern>{(displayedBoard.arrows || []).map((arrow) => { const headSize = Number(arrow.headSize) || 20; const headStyle = arrow.headStyle || "filled"; return <marker key={`head-${arrow.id}`} id={`sr-tactical-arrowhead-${arrow.id}`} markerWidth={headSize} markerHeight={headSize} refX={headSize} refY={headSize / 2} orient="auto" markerUnits="userSpaceOnUse"><path d={`M0,0 L${headSize},${headSize / 2} L0,${headSize}`} fill={headStyle === "open" ? "none" : arrow.color || "#ffe36e"} stroke={arrow.color || "#ffe36e"} strokeWidth={Math.max(1, headSize / 8)} strokeLinejoin="round" fillOpacity={arrow.opacity ?? 1} strokeOpacity={arrow.opacity ?? 1} /></marker>; })}<marker id="sr-tactical-arrowhead-preview" markerWidth={newArrowHeadSize} markerHeight={newArrowHeadSize} refX={newArrowHeadSize} refY={newArrowHeadSize / 2} orient="auto" markerUnits="userSpaceOnUse"><path d={`M0,0 L${newArrowHeadSize},${newArrowHeadSize / 2} L0,${newArrowHeadSize}`} fill={newArrowHeadStyle === "open" ? "none" : newShapeColour} stroke={newShapeColour} strokeWidth={Math.max(1, newArrowHeadSize / 8)} strokeLinejoin="round" /></marker></defs>
          {pitchViewMode === "full" ? <>
            <rect x="12" y="12" width="976" height="616" rx="7" className="sr-tactical-grass" />
            <rect x="12" y="12" width="976" height="616" rx="7" fill="url(#sr-tactical-grass-stripes)" pointerEvents="none" />
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
            <rect x="12" y="12" width="616" height="976" rx="7" fill="url(#sr-tactical-grass-stripes)" pointerEvents="none" />
              {pitchViewMode === "half" ? <>
              <rect x="35" y="35" width="570" height="930" className="sr-tactical-line" />
              <line x1="35" y1="470" x2="605" y2="470" className="sr-tactical-line" />
              <circle cx="320" cy="470" r="83" className="sr-tactical-line" />
              <circle cx="320" cy="470" r="4" className="sr-tactical-mark" />
              <rect x="155" y="35" width="330" height="120" className="sr-tactical-line" />
              <rect x="235" y="35" width="170" height="50" className="sr-tactical-line" />
              <path d="M238 155 A82 82 0 0 0 402 155" className="sr-tactical-line" />
              <circle cx="320" cy="145" r="4" className="sr-tactical-mark" />
            </> : <>
              <rect x="35" y="35" width="570" height="930" className="sr-tactical-line" />
              <path d="M35 55 A20 20 0 0 0 55 35 M585 35 A20 20 0 0 0 605 55" className="sr-tactical-line" />
              <rect x="125" y="35" width="390" height="160" className="sr-tactical-line" />
              <rect x="220" y="35" width="200" height="60" className="sr-tactical-line" />
              <path d="M240 195 A82 82 0 0 0 400 195" className="sr-tactical-line" />
              <circle cx="320" cy="145" r="4" className="sr-tactical-mark" />
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
            const border = shape.borderColor || fill;
            const opacity = shape.opacity ?? 0.35;
            const strokeWidth = Number(shape.strokeWidth) || 3;
            const textAlign = ["left", "center", "right"].includes(shape.textAlign) ? shape.textAlign : "left";
            const textStyle = { fontSize: `${shape.fontSize || 22}px`, color: shape.textColor || "#fff", backgroundColor: shape.backgroundColor || "#0c3769", fontWeight: shape.bold === false ? 400 : 700, fontStyle: shape.italic ? "italic" : "normal", textDecoration: shape.underline ? "underline" : "none", textAlign };
            let shapeBody;
            if (shape.type === "text") {
              shapeBody = <><rect x={shape.x} y={shape.y} width={shape.width} height={shape.height} rx="5" fill={shape.backgroundColor || "#0c3769"} fillOpacity={shape.opacity ?? 0.85} stroke={selected ? "#ffe36e" : border} strokeOpacity={1} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} />{editingTextId === shape.id && !videoPlaying ? <foreignObject x={shape.x + 5} y={shape.y + 4} width={Math.max(20, shape.width - 10)} height={Math.max(24, shape.height - 8)}><textarea xmlns="http://www.w3.org/1999/xhtml" ref={inlineTextRef} className="sr-tactical-inline-text" style={textStyle} autoFocus aria-label="Edit text on pitch" value={shape.text || ""} onPointerDown={(event) => event.stopPropagation()} onChange={(event) => setBoard((current) => ({ ...current, shapes: (current.shapes || []).map((item) => item.id === shape.id ? { ...item, text: event.target.value } : item) }))} onBlur={() => setEditingTextId(null)} onKeyDown={(event) => { if (event.key === "Escape") { event.preventDefault(); setEditingTextId(null); event.currentTarget.blur(); } }} /></foreignObject> : <foreignObject x={shape.x + 5} y={shape.y + 4} width={Math.max(20, shape.width - 10)} height={Math.max(24, shape.height - 8)}><div xmlns="http://www.w3.org/1999/xhtml" className="sr-tactical-text-render" style={{ ...textStyle, backgroundColor: "transparent", justifyContent: textAlign === "center" ? "center" : textAlign === "right" ? "flex-end" : "flex-start" }} tabIndex={0} role="textbox" aria-multiline="true" aria-label="Tactical text box. Click and type to edit." data-tactical-text-input="true" onClick={(event) => { event.stopPropagation(); setSelectedShapeId(shape.id); }} onKeyDown={(event) => { if (event.metaKey || event.ctrlKey || event.altKey) return; if (event.key === "Enter") { event.preventDefault(); startInlineTextEdit(shape); } else if (event.key === "Backspace") { event.preventDefault(); startInlineTextEdit(shape, (shape.text || "").slice(0, -1)); } else if (event.key.length === 1) { event.preventDefault(); startInlineTextEdit(shape, event.key); } }} onDoubleClick={(event) => { event.stopPropagation(); startInlineTextEdit(shape); }}>{shape.text || "Text"}</div></foreignObject>}</>;
            } else if (shape.type === "line") shapeBody = <line x1={shape.x} y1={shape.y} x2={shape.x + shape.width} y2={shape.y + shape.height} stroke={selected ? "#ffe36e" : border} strokeOpacity={opacity} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} pointerEvents="stroke" />;
            else if (shape.type === "circle") shapeBody = <ellipse cx={shape.x + shape.width / 2} cy={shape.y + shape.height / 2} rx={shape.width / 2} ry={shape.height / 2} fill={fill} fillOpacity={opacity} stroke={selected ? "#ffe36e" : border} strokeOpacity={selected ? 1 : opacity} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} />;
            else if (shape.type === "triangle") shapeBody = <polygon points={`${shape.x + shape.width / 2},${shape.y} ${shape.x + shape.width},${shape.y + shape.height} ${shape.x},${shape.y + shape.height}`} fill={fill} fillOpacity={opacity} stroke={selected ? "#ffe36e" : border} strokeOpacity={selected ? 1 : opacity} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} />;
            else shapeBody = <rect x={shape.x} y={shape.y} width={shape.width} height={shape.height} rx="2" fill={fill} fillOpacity={opacity} stroke={selected ? "#ffe36e" : border} strokeOpacity={selected ? 1 : opacity} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} />;
            const handles = selected && !videoPlaying ? [["nw", shape.x, shape.y], ["ne", shape.x + shape.width, shape.y], ["sw", shape.x, shape.y + shape.height], ["se", shape.x + shape.width, shape.y + shape.height]] : [];
            const keepTextReadable = shape.type === "text" && pitchViewMode !== "full";
            const counterRotation = keepTextReadable ? `rotate(90 ${shape.x + shape.width / 2} ${shape.y + shape.height / 2})` : undefined;
            return <g key={`${shape.id}-${pitchViewMode}`} {...graphicEvents}><g transform={counterRotation}>{shapeBody}{handles.map(([corner, x, y]) => <circle key={corner} className="sr-tactical-resize-handle" style={{ cursor: corner === "nw" || corner === "se" ? "nwse-resize" : "nesw-resize" }} cx={x} cy={y} r="7" onPointerDown={(event) => beginShapeResize(event, shape, corner)} />)}</g></g>;
          })}
          {arrowPreview && arrowStart && mode === "arrow" && <line x1={arrowStart.x} y1={arrowStart.y} x2={arrowPreview.x} y2={arrowPreview.y} className="sr-tactical-arrow sr-tactical-arrow-preview" stroke={newShapeColour} strokeDasharray={newArrowLineStyle === "dashed" ? "14 9" : newArrowLineStyle === "dotted" ? "2 8" : undefined} markerEnd={newArrowHeadStyle === "none" ? undefined : "url(#sr-tactical-arrowhead-preview)"} />}
          {arrowPreview && arrowStart && ["rectangle", "square", "circle", "triangle", "line", "text"].includes(mode) && (() => {
            const shape = mode === "text" ? textBoxForDrag(arrowStart, arrowPreview) : shapeForDrag(arrowStart, arrowPreview, mode);
            if (mode === "text") return <g transform={pitchViewMode === "full" ? undefined : `rotate(90 ${shape.x + shape.width / 2} ${shape.y + shape.height / 2})`}><rect {...shape} rx="5" fill={newTextBoxColor} fillOpacity="0.85" stroke={newShapeBorderColour} strokeWidth={2} className="sr-tactical-shape-preview" /></g>;
            if (mode === "line") return <line x1={arrowStart.x} y1={arrowStart.y} x2={arrowPreview.x} y2={arrowPreview.y} stroke={newShapeBorderColour} strokeOpacity={newShapeOpacity} strokeWidth={3} className="sr-tactical-shape-preview" />;
            if (mode === "circle") return <ellipse cx={shape.x + shape.width / 2} cy={shape.y + shape.height / 2} rx={shape.width / 2} ry={shape.height / 2} fill={newShapeColour} fillOpacity={newShapeOpacity} stroke={newShapeBorderColour} strokeOpacity={newShapeOpacity} strokeWidth={3} className="sr-tactical-shape-preview" />;
            if (mode === "triangle") return <polygon points={`${shape.x + shape.width / 2},${shape.y} ${shape.x + shape.width},${shape.y + shape.height} ${shape.x},${shape.y + shape.height}`} fill={newShapeColour} fillOpacity={newShapeOpacity} stroke={newShapeBorderColour} strokeOpacity={newShapeOpacity} strokeWidth={3} className="sr-tactical-shape-preview" />;
            return <rect {...shape} fill={newShapeColour} fillOpacity={newShapeOpacity} stroke={newShapeBorderColour} strokeOpacity={newShapeOpacity} strokeWidth={3} className="sr-tactical-shape-preview" />;
          })()}
          {(displayedBoard.arrows || []).map((arrow) => {
            const selected = !videoPlaying && selectedArrowId === arrow.id;
            const strokeWidth = Number(arrow.strokeWidth) || 6;
            const dashPattern = arrow.lineStyle === "dashed" ? "14 9" : arrow.lineStyle === "dotted" ? "2 8" : undefined;
            return <g key={arrow.id}>
              <line x1={arrow.from.x} y1={arrow.from.y} x2={arrow.to.x} y2={arrow.to.y} stroke={arrow.color || "#ffe36e"} strokeOpacity={arrow.opacity ?? 1} strokeWidth={selected ? strokeWidth + 2 : strokeWidth} strokeDasharray={dashPattern} strokeLinecap="round" className={`sr-tactical-arrow ${selected ? "selected" : ""} ${draggingArrowId === arrow.id ? "dragging" : ""} ${enteringGraphicIds.includes(arrow.id) ? "entering" : ""}`} markerEnd={arrow.headStyle === "none" ? undefined : `url(#sr-tactical-arrowhead-${arrow.id})`} onAnimationEnd={() => finishGraphicEntering(arrow.id)} onPointerDown={(event) => beginArrowDrag(event, arrow)} />
              {selected && [["from", arrow.from], ["to", arrow.to]].map(([endpoint, point]) => <circle key={endpoint} className="sr-tactical-arrow-resize-handle" cx={point.x} cy={point.y} r="7" onPointerDown={(event) => beginArrowEndResize(event, arrow, endpoint)} />)}
            </g>;
          })}
          <g className={`sr-tactical-ball ${draggingBall ? "dragging" : ""} ${videoPlaying ? "video-playing" : ""}`} transform={`translate(${displayedBoard.ball?.x ?? 500} ${displayedBoard.ball?.y ?? 320})`} onPointerDown={beginBallDrag}>
            <circle r={pitchViewMode === "full" ? 13 : 11} className="sr-tactical-ball-shadow" /><circle r={pitchViewMode === "full" ? 10 : 8} className="sr-tactical-ball-shell" />
            <path d={pitchViewMode === "full" ? "M0 -5 L4 -2 L3 3 L-3 3 L-4 -2 Z" : "M0 -4 L3 -2 L2 2 L-2 2 L-3 -2 Z"} className="sr-tactical-ball-patch" />
          </g>
          {displayedBoard.players.map((player) => <g key={player.id} className={`sr-tactical-player ${player.side} ${!videoPlaying && selectedTokens.includes(player.id) ? "selected" : ""} ${draggingIds.includes(player.id) ? "dragging" : ""} ${videoPlaying ? "video-playing" : ""}`} transform={`translate(${player.x} ${player.y})`} onPointerDown={(event) => beginPlayerDrag(event, player)}>
            <circle r={pitchViewMode === "full" ? 27 : 22} className="sr-tactical-disc-shadow" />
            <circle r={pitchViewMode === "full" ? 24 : 19} className="sr-tactical-disc" style={player.color ? { fill: player.color } : undefined} />
            <text y="1" style={{ fontSize: pitchViewMode === "full" ? undefined : "14px" }} transform={pitchViewMode === "full" ? undefined : "rotate(90)"}>{player.number}</text>
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
            <button type="button" className={activeClipStart ? "sr-tactical-recording" : "sr-tactical-video"} onClick={beginOrSaveMovementClip} disabled={finishingVideo}>{activeClipStart ? editingClipId ? "Save clip edits" : "Save movement clip" : "Start movement clip"}</button>
            {activeClipStart && <button type="button" onClick={cancelMovementClip} disabled={finishingVideo}>Cancel clip</button>}
            <button type="button" className={videoRecording ? "sr-tactical-recording" : "sr-tactical-video"} onClick={playAllMovementClips} disabled={!movementClips.length || finishingVideo || Boolean(activeClipStart)}>{finishingVideo ? "Playing clips…" : `Play all clips${movementClips.length ? ` (${movementClips.length})` : ""}`}</button>
            <button type="button" onClick={downloadClipStillsPdf} disabled={finishingVideo || Boolean(activeClipStart)}>Download stills PDF</button>
            <small>Start a clip, move the discs or ball, then save it. Repeat to build a sequence.</small>
          </div>
          {pitchMode === "box" && <div className="sr-tactical-pdf-banner-controls" aria-label="PDF title page settings">
            <strong>PDF title page</strong>
            <label><span>Title</span><input aria-label="PDF title page title" type="text" value={pdfCover.title} onChange={(event) => setPdfCover((current) => ({ ...current, title: event.target.value }))} /></label>
            <label><span>Background</span><input aria-label="PDF title page background colour" type="color" value={pdfCover.background} onChange={(event) => setPdfCover((current) => ({ ...current, background: event.target.value }))} /></label>
            <label><span>Text</span><input aria-label="PDF title page text colour" type="color" value={pdfCover.text} onChange={(event) => setPdfCover((current) => ({ ...current, text: event.target.value }))} /></label>
            <label><span>Accent</span><input aria-label="PDF title page accent colour" type="color" value={pdfCover.accent} onChange={(event) => setPdfCover((current) => ({ ...current, accent: event.target.value }))} /></label>
          </div>}
          <div className="sr-tactical-pdf-banner-controls" aria-label="PDF banner settings">
            <strong>PDF tactic banner</strong>
            <label><span>Banner title</span><input aria-label="PDF banner title" type="text" value={pdfBanner.title} placeholder={board.name || "Use tactic name"} onChange={(event) => setPdfBanner((current) => ({ ...current, title: event.target.value }))} /></label>
            <label><span>Background</span><input aria-label="PDF banner background colour" type="color" value={pdfBanner.fill} onChange={(event) => setPdfBanner((current) => ({ ...current, fill: event.target.value }))} /></label>
            <label><span>Text</span><input aria-label="PDF banner text colour" type="color" value={pdfBanner.text} onChange={(event) => setPdfBanner((current) => ({ ...current, text: event.target.value }))} /></label>
            <label><span>Border</span><input aria-label="PDF banner border colour" type="color" value={pdfBanner.border} onChange={(event) => setPdfBanner((current) => ({ ...current, border: event.target.value }))} /></label>
            <label><span>Page background</span><input aria-label="PDF page background colour" type="color" value={pdfBanner.pageBackground || "#eaf2f8"} onChange={(event) => setPdfBanner((current) => ({ ...current, pageBackground: event.target.value }))} /></label>
            <label><span>Style</span><select aria-label="PDF banner style" value={pdfBanner.style} onChange={(event) => setPdfBanner((current) => ({ ...current, style: event.target.value }))}><option value="rounded">Rounded</option><option value="square">Square</option></select></label>
          </div>
          <div className="sr-tactical-still-slides">
            <div className="sr-tactical-still-slides-heading"><strong>Still slides</strong><button type="button" onClick={addStillSlide} disabled={finishingVideo || Boolean(activeClipStart)}>Add current setup as slide</button><small>Capture the pitch as it is now. You can add another slide without recording a movement.</small></div>
            {stillSlides.length > 0 && <ol>{stillSlides.map((slide, index) => <li key={slide.id}>
              <span className="sr-tactical-still-slide-number">{index + 1}</span>
              <label><span>Slide banner title</span><input aria-label={`Slide ${index + 1} banner title`} value={slide.bannerTitle || slide.title || `Slide ${index + 1}`} onChange={(event) => renameStillSlide(slide.id, event.target.value)} /></label>
              <button type="button" onClick={() => loadStillSlide(slide)} disabled={finishingVideo || Boolean(activeClipStart)}>Load</button>
              <button type="button" onClick={() => replaceStillSlide(slide.id)} disabled={finishingVideo || Boolean(activeClipStart)}>Update</button>
              <button type="button" onClick={() => setStillSlides((current) => current.filter((item) => item.id !== slide.id))} disabled={finishingVideo || Boolean(activeClipStart)}>Remove</button>
            </li>)}</ol>}
          </div>
          {movementClips.length > 0 && <div className="sr-tactical-clip-list"><strong>Movement clips — preview, edit, and adjust timing</strong><ol>{movementClips.map((clip, index) => <li key={clip.id}><button type="button" className="sr-tactical-clip-preview-button" onClick={() => previewMovementClip(clip, index + 1)} disabled={finishingVideo || Boolean(activeClipStart)}>Clip {index + 1}: {clip.movedCount} piece{clip.movedCount === 1 ? "" : "s"} moved <span>Preview</span></button><button type="button" onClick={() => editMovementClip(clip, index + 1)} disabled={finishingVideo || Boolean(activeClipStart)}>Edit</button><button type="button" onClick={() => persistMovementClips((current) => current.filter((item) => item.id !== clip.id))} disabled={finishingVideo || Boolean(activeClipStart)}>Remove</button><label className="sr-tactical-clip-duration"><span>Duration</span><input aria-label={`Clip ${index + 1} duration`} type="range" min="2000" max="12000" step="500" value={tacticalClipDuration(clip)} disabled={finishingVideo || Boolean(activeClipStart && editingClipId !== clip.id)} onChange={(event) => persistMovementClips((current) => current.map((item) => item.id === clip.id ? { ...item, durationMs: Number(event.target.value) } : item))} /><output>{(tacticalClipDuration(clip) / 1000).toFixed(1)}s</output></label></li>)}</ol><div className="sr-tactical-clip-list-actions"><button type="button" onClick={() => persistMovementClips((current) => current.slice(0, -1))} disabled={finishingVideo || Boolean(activeClipStart)}>Undo last clip</button><button type="button" onClick={() => persistMovementClips([])} disabled={finishingVideo || Boolean(activeClipStart)}>Clear clips</button></div></div>}
        </div>}
        {dockPanel === "shapes" && <div className="sr-tactical-dock-panel" role="tabpanel">
          <div className="sr-tactical-tools sr-tactical-shape-palette" role="toolbar" aria-label="Shape and drawing tools">
            <button type="button" className={`sr-tactical-icon-tool ${mode === "move" && !multiSelectMode ? "selected" : ""}`} onClick={() => { setMode("move"); setMultiSelectMode(false); }}><TacticalToolIcon type="move" /><span>Select</span></button>
            <button type="button" className={`sr-tactical-icon-tool ${multiSelectMode ? "selected" : ""}`} onClick={() => { setMode("move"); setMultiSelectMode((value) => !value); }}><TacticalToolIcon type="multiple" /><span>Multi</span></button>
            <button type="button" className={`sr-tactical-icon-tool ${mode === "arrow" ? "selected" : ""}`} onClick={() => { setMode("arrow"); setSelectedShapeId(null); setSelectedArrowId(null); }}><TacticalToolIcon type="arrow" /><span>Arrow</span></button>
            {[["line", "Line"], ["rectangle", "Rectangle"], ["square", "Square"], ["circle", "Circle"], ["triangle", "Triangle"]].map(([tool, label]) => <button key={tool} type="button" className={`sr-tactical-icon-tool ${mode === tool ? "selected" : ""}`} onClick={() => { setMode(tool); setSelectedShapeId(null); setSelectedArrowId(null); }}><TacticalToolIcon type={tool} /><span>{label}</span></button>)}
            <label className="sr-tactical-color-control"><span>Colour</span><input type="color" value={selectedGraphic?.color || newShapeColour} onChange={(event) => { setNewShapeColour(event.target.value); updateSelectedGraphic({ color: event.target.value }); }} /></label>
            {mode !== "arrow" && <label className="sr-tactical-color-control"><span>Border</span><input aria-label="Shape border colour" type="color" value={selectedShape?.borderColor || selectedShape?.color || newShapeBorderColour} onChange={(event) => { setNewShapeBorderColour(event.target.value); if (selectedShape) updateSelectedShape({ borderColor: event.target.value }); }} /></label>}
            <label className="sr-tactical-opacity-control"><span>Transparency</span><input aria-label="Shape transparency" type="range" min="0" max="95" step="5" value={100 - Math.round((selectedGraphic?.opacity ?? newShapeOpacity) * 100)} onChange={(event) => { const opacity = 1 - Number(event.target.value) / 100; if (selectedGraphic) updateSelectedGraphic({ opacity }); else setNewShapeOpacity(opacity); }} /><output>{100 - Math.round((selectedGraphic?.opacity ?? newShapeOpacity) * 100)}%</output></label>
            <label className="sr-tactical-stroke-width-control"><span>Line width</span><input aria-label="Line width" type="range" min="1" max="16" step="1" value={selectedGraphic?.strokeWidth || newShapeStrokeWidth} onChange={(event) => { const strokeWidth = Number(event.target.value); setNewShapeStrokeWidth(strokeWidth); updateSelectedGraphic({ strokeWidth }); }} /><output>{selectedGraphic?.strokeWidth || newShapeStrokeWidth}px</output></label>
            {(selectedArrow || mode === "arrow") && <><label className="sr-tactical-line-style-control arrow-style-line"><span>Arrow line</span><select aria-label="Arrow line style" value={selectedArrow?.lineStyle || newArrowLineStyle} onChange={(event) => { setNewArrowLineStyle(event.target.value); if (selectedArrow) updateSelectedGraphic({ lineStyle: event.target.value }); }}><option value="solid">Solid</option><option value="dashed">Dashed</option><option value="dotted">Dotted</option></select></label><label className="sr-tactical-line-style-control arrow-style-head"><span>Arrow head</span><select aria-label="Arrow head style" value={selectedArrow?.headStyle || newArrowHeadStyle} onChange={(event) => { setNewArrowHeadStyle(event.target.value); if (selectedArrow) updateSelectedGraphic({ headStyle: event.target.value }); }}><option value="filled">Filled</option><option value="open">Open</option><option value="none">None</option></select></label><label className="sr-tactical-line-style-control arrow-style-size"><span>Head size</span><input aria-label="Arrowhead size" type="range" min="8" max="36" step="2" value={selectedArrow?.headSize || newArrowHeadSize} onChange={(event) => { const headSize = Number(event.target.value); setNewArrowHeadSize(headSize); if (selectedArrow) updateSelectedGraphic({ headSize }); }} /><output>{selectedArrow?.headSize || newArrowHeadSize}px</output></label></>}
            <button type="button" onClick={() => { if (selectedShape) setBoard((current) => ({ ...current, shapes: (current.shapes || []).filter((shape) => shape.id !== selectedShapeId) })); else if (selectedArrowId) setBoard((current) => ({ ...current, arrows: (current.arrows || []).filter((arrow) => arrow.id !== selectedArrowId) })); setSelectedShapeId(null); setSelectedArrowId(null); }} disabled={!selectedGraphic}>Delete selected</button>
            <button type="button" onClick={() => { setBoard((current) => ({ ...current, shapes: [], arrows: [] })); setSelectedShapeId(null); setSelectedArrowId(null); }} disabled={!(board.shapes || []).length && !board.arrows.length}>Clear shapes</button>
          </div>
          <div className="sr-tactical-copy-actions"><button type="button" onClick={copySelectedGraphic} disabled={!selectedGraphic}>Copy selected</button><button type="button" onClick={pasteCopiedGraphic} disabled={!copiedGraphic}>Paste copy</button><small>{graphicClipboardStatus || "Select any shape, line, text box, or arrow. Copy/Paste also works with Cmd/Ctrl+C and Cmd/Ctrl+V."}</small></div>
          <small>{selectedArrow ? "Drag either endpoint to resize. Adjust its line and head above. Use Cmd/Ctrl+C to copy, Cmd/Ctrl+V to paste, or Delete to remove it." : selectedShape ? "Drag a corner handle to resize. Change its style above. Use Cmd/Ctrl+C to copy, Cmd/Ctrl+V to paste, or Delete to remove it." : "Choose a tool, then draw on the pitch. Select any shape, line, or arrow to edit, copy, paste, or delete it."}</small>
        </div>}
        {dockPanel === "text" && <div className="sr-tactical-dock-panel sr-tactical-text-panel" role="tabpanel">
          <button type="button" className={`sr-tactical-icon-tool ${mode === "text" ? "selected" : ""}`} onClick={() => { setMode("text"); setSelectedShapeId(null); setSelectedArrowId(null); setEditingTextId(null); }}><TacticalToolIcon type="text" /><span>Draw text box</span></button>
          {selectedShape?.type === "text" && <>
            <label className="sr-tactical-text-editor-control"><span>Edit selected text</span><textarea ref={textEditorRef} rows="4" value={selectedShape.text || ""} onChange={(event) => updateSelectedShape({ text: event.target.value })} placeholder="Type text. Press Enter for a new line." /></label>
            <div className="sr-tactical-text-format-toolbar" role="toolbar" aria-label="Text formatting">
              <button type="button" aria-label="Bold" aria-pressed={selectedShape.bold !== false} className={selectedShape.bold !== false ? "active" : ""} onClick={() => updateSelectedShape({ bold: selectedShape.bold === false })}><strong>B</strong></button>
              <button type="button" aria-label="Italic" aria-pressed={Boolean(selectedShape.italic)} className={selectedShape.italic ? "active" : ""} onClick={() => updateSelectedShape({ italic: !selectedShape.italic })}><em>I</em></button>
              <button type="button" aria-label="Underline" aria-pressed={Boolean(selectedShape.underline)} className={selectedShape.underline ? "active" : ""} onClick={() => updateSelectedShape({ underline: !selectedShape.underline })}><u>U</u></button>
              <span aria-hidden="true" />
              {["left", "center", "right"].map((alignment) => <button key={alignment} type="button" aria-label={`Align ${alignment}`} aria-pressed={(selectedShape.textAlign || "left") === alignment} className={(selectedShape.textAlign || "left") === alignment ? "active" : ""} onClick={() => updateSelectedShape({ textAlign: alignment })}>{alignment === "left" ? "⇤" : alignment === "center" ? "↔" : "⇥"}</button>)}
              <button type="button" onClick={insertBulletIntoSelectedText}>Add bullet point</button>
            </div>
            <label className="sr-tactical-font-size-control"><span>Text size</span><input aria-label="Text size" type="range" min="12" max="42" step="1" value={selectedShape.fontSize || 22} onChange={(event) => updateSelectedShape({ fontSize: Number(event.target.value) })} /><output>{selectedShape.fontSize || 22}px</output></label>
          </>}
          <div className="sr-tactical-text-colours"><label><span>Text colour</span><input aria-label="Text colour" type="color" value={selectedShape?.type === "text" ? selectedShape.textColor || "#ffffff" : newTextColor} onChange={(event) => { setNewTextColor(event.target.value); if (selectedShape?.type === "text") updateSelectedShape({ textColor: event.target.value }); }} /></label><label><span>Box colour</span><input aria-label="Text box colour" type="color" value={selectedShape?.type === "text" ? selectedShape.backgroundColor || "#0c3769" : newTextBoxColor} onChange={(event) => { setNewTextBoxColor(event.target.value); if (selectedShape?.type === "text") updateSelectedShape({ backgroundColor: event.target.value }); }} /></label><label><span>Border colour</span><input aria-label="Text box border colour" type="color" value={selectedShape?.type === "text" ? selectedShape.borderColor || selectedShape.color || newShapeBorderColour : newShapeBorderColour} onChange={(event) => { setNewShapeBorderColour(event.target.value); if (selectedShape?.type === "text") updateSelectedShape({ borderColor: event.target.value }); }} /></label></div>
          <small>{selectedShape?.type === "text" ? "Click the text box and type, or double-click to edit existing text. Enter adds a line. Use the toolbar for bold, italic, underline, alignment, and bullets." : "Choose Draw text box, then drag on the pitch to set its size. New boxes switch back to Select automatically."}</small>
        </div>}
        {dockPanel === "tools" && <div className="sr-tactical-dock-panel" role="tabpanel">
          <div className="sr-tactical-tools" role="toolbar" aria-label="Player and pitch tools">
            <label className="sr-tactical-pitch-mode"><span>Pitch view</span><select aria-label="Pitch view" value={pitchMode} onChange={(event) => changePitchMode(event.target.value)} disabled={Boolean(activeClipStart) || finishingVideo}><option value="full">Full pitch · landscape</option><option value="half">Half pitch · portrait</option><option value="box">18-yard box · set-pieces</option></select></label>
            <button type="button" onClick={() => addPlayer("home")}>+ Home disc</button>
            <button type="button" onClick={() => addPlayer("away")}>+ Opponent disc</button>
            <button type="button" onClick={removeSelectedPlayer} disabled={!selectedTokens.length}>Remove selected player</button>
            <button type="button" onClick={() => { setBoard((current) => ({ ...current, arrows: [] })); setSelectedArrowId(null); }} disabled={!board.arrows.length}>Clear arrows</button>
            <button type="button" onClick={() => { setBoard({ ...freshTacticalBoard(homeTeamName, oppositionName), pitchMode }); setSelectedTokens([]); setSelectedShapeId(null); setSelectedArrowId(null); activeClipStartRef.current = null; activeClipBallPathRef.current = []; setActiveClipStart(null); setEditingClipId(null); }}>Reset pitch</button>
          </div>
          <div className="sr-tactical-disc-edit-controls" aria-label="Selected disc settings">
            <label><span>Disc colour</span><input aria-label="Selected disc colour" type="color" value={selectedPlayers[0]?.color || (selectedPlayers[0]?.side === "away" ? "#e64b50" : "#1485dd")} disabled={!selectedPlayers.length} onChange={(event) => updateSelectedPlayers({ color: event.target.value })} /></label>
            <label><span>Disc number</span><input aria-label="Selected disc number" type="number" min="1" max="99" step="1" value={selectedPlayers.length === 1 ? selectedPlayers[0].number : ""} disabled={selectedPlayers.length !== 1} onChange={(event) => { const nextNumber = Number(event.target.value); if (nextNumber >= 1 && nextNumber <= 99) updateSelectedPlayers({ number: nextNumber }); }} /></label>
            <small>{selectedPlayers.length > 1 ? "Select one disc to change its number. " : ""}Select discs and use Cmd/Ctrl+C, Cmd/Ctrl+V, or Delete/Backspace to copy, paste, and remove them.</small>
          </div>
          <div className="sr-tactical-tools sr-tactical-saved-tactics" role="toolbar" aria-label="Save and open tactics">
            <button type="button" className="sr-tactical-video" onClick={() => saveCurrentTactic()} disabled={finishingVideo || Boolean(activeClipStart)}>Save tactic and clips</button>
            <label className="sr-tactical-saved-tactic-select"><span>Saved tactics</span><select aria-label="Saved tactics" value={selectedSavedTacticId} onChange={(event) => setSelectedSavedTacticId(event.target.value)}><option value="">Choose a saved tactic</option>{savedTactics.map((tactic) => <option key={tactic.id} value={tactic.id}>{tactic.name} · {tactic.pitchMode === "full" ? "Full pitch" : tactic.pitchMode === "half" ? "Half pitch" : "18-yard box"} · {tactic.movementClips?.length || 0} clips · {tactic.stillSlides?.length || 0} slides</option>)}</select></label>
            <button type="button" onClick={loadSavedTactic} disabled={!selectedSavedTacticId || finishingVideo || Boolean(activeClipStart)}>Open saved tactic</button>
            <button type="button" onClick={deleteSavedTactic} disabled={!selectedSavedTacticId || finishingVideo || Boolean(activeClipStart)}>Delete saved tactic</button>
            <button type="button" onClick={startNewTactic} disabled={finishingVideo || Boolean(activeClipStart)}>Save and start new tactic</button>
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
