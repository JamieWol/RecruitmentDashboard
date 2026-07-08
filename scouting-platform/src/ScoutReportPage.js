import React, { useState, useRef, useEffect } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
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
  const exportRef = useRef(null);
  const [clipsLink, setClipsLink] = useState("");
  const [filteredPlayers, setFilteredPlayers] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [photoDataUrl, setPhotoDataUrl] = useState("");
  const [photoFile, setPhotoFile] = useState(null);
  const [uploadingPhoto, setUploadingPhoto] = useState(false);
  const [photoStatus, setPhotoStatus] = useState("");
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

      const images = Array.from(exportRef.current.querySelectorAll("img"));
      await Promise.all(
        images.map(
          (img) =>
            new Promise((resolve) => {
              if (img.complete) return resolve();
              img.onload = resolve;
              img.onerror = resolve;
            })
        )
      );

      const canvas = await html2canvas(exportRef.current, {
        scale: 2,
        useCORS: true,
        allowTaint: false,
        backgroundColor: "#ffffff",
      });

      const imgData = canvas.toDataURL("image/png");

      const pdf = new jsPDF("p", "mm", "a4");
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, "PNG", 0, 10, pdfWidth, pdfHeight);
      pdf.save(`${selectedPlayer["Player Name"]}_Report.pdf`);
    };

    const slugifyLegacy = (text) =>
      String(text || "")
        .toLowerCase()
        .trim()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");

    const slugify = (text) => {
      const charMap = {
        "ł": "l", "Ł": "l",
        "đ": "d", "Đ": "d",
        "ð": "d", "Ð": "d",
        "þ": "th", "Þ": "th",
        "æ": "ae", "Æ": "ae",
        "œ": "oe", "Œ": "oe",
        "ø": "o", "Ø": "o",
        "ı": "i", "İ": "i",
        "ß": "ss",
      };

      return String(text || "")
        .split("")
        .map((ch) => charMap[ch] || ch)
        .join("")
        .normalize("NFD")
        .replace(/[̀-ͯ]/g, "")
        .toLowerCase()
        .trim()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
    };

    const getDisplayName = (fullName) => {
      const cleaned = String(fullName || "").replace(/\s+/g, " ").trim();
      if (!cleaned) return "";
      const parts = cleaned.split(" ").filter(Boolean);
      if (parts.length <= 2) return cleaned;
      return `${parts[0]} ${parts[parts.length - 1]}`;
    };

    const buildPhotoCandidates = (player) => {
      const fullName =
        player?.["Full Player Name"] ||
        player?.["Player Name"] ||
        player?.["Display Name"] ||
        player?.Player ||
        "";

      const displayName = getDisplayName(fullName || player?.["Player Name"] || "");
      const rawCandidates = uniquePreserveOrder([
        fullName,
        displayName,
        player?.["Player Name"],
        player?.["Display Name"],
      ].filter(Boolean));

      const bases = uniquePreserveOrder([
        ...rawCandidates.map(slugify),
        ...rawCandidates.map(slugifyLegacy),
      ].filter(Boolean));

      const variants = uniquePreserveOrder(
        bases.flatMap((base) => [base, `_${base}`, `__${base}`])
      );

      return variants.map(
        (filename) =>
          `https://syjsmvvsvvprxibqoizw.supabase.co/storage/v1/object/public/player-photos/player-photos/${filename}.png`
      );
    };

    const getPlayerPhoto = (player) => {
      if (player?._photoUrl) return player._photoUrl;
      if (player?.Photo) return player.Photo;

      const candidates = buildPhotoCandidates(player);
      return candidates[0] || "/placeholder-player.png";
    };

  useEffect(() => {
    if (!selectedPlayer) {
      setPhotoDataUrl("");
      return;
    }

    let cancelled = false;
    const candidates = buildPhotoCandidates(selectedPlayer);

    const loadPhoto = async () => {
      for (const url of candidates) {
        try {
          const response = await fetch(url, { mode: "cors" });
          if (!response.ok) continue;

          const blob = await response.blob();
          const dataUrl = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(String(reader.result || ""));
            reader.onerror = reject;
            reader.readAsDataURL(blob);
          });

          if (!cancelled) setPhotoDataUrl(dataUrl);
          return;
        } catch {
          // try next candidate
        }
      }

      if (!cancelled) setPhotoDataUrl("");
    };

    loadPhoto();

    return () => {
      cancelled = true;
    };
  }, [selectedPlayer]);

  const DATE_HINTS = [
    "date", "dob", "birth", "expiry", "expires", "joined", "created",
    "updated", "time", "report", "contract"
  ];

  const isDateHeader = (key) => {
    const text = String(key || "").toLowerCase();
    return DATE_HINTS.some((hint) => text.includes(hint));
  };

  const excelSerialToISO = (serial) => {
    try {
      const parsed = XLSX?.SSF?.parse_date_code?.(serial);
      if (!parsed) return null;
      const yyyy = String(parsed.y).padStart(4, "0");
      const mm = String(parsed.m).padStart(2, "0");
      const dd = String(parsed.d).padStart(2, "0");
      return `${yyyy}-${mm}-${dd}`;
    } catch {
      return null;
    }
  };

  const isDateLikeValue = (value) => {
    if (value instanceof Date && !Number.isNaN(value.getTime())) return true;
    if (typeof value !== "string") return false;
    const s = value.trim();
    if (!s) return false;
    return (
      /^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$/.test(s) ||
      /^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:[ T].*)?$/.test(s) ||
      /^\w{3,9}\s+\d{1,2},\s+\d{4}$/.test(s)
    );
  };

  const normalizeUploadedValue = (key, value) => {
    if (value === null || value === undefined || value === "") return "";

    if (value instanceof Date && !Number.isNaN(value.getTime())) {
      return value.toISOString().slice(0, 10);
    }

    if (typeof value === "number") {
      if (isDateHeader(key)) {
        const iso = excelSerialToISO(value);
        if (iso) return iso;
      }
      return value;
    }

    const raw = String(value).trim();
    if (!raw) return "";

    if (isDateHeader(key) || isDateLikeValue(raw)) {
      const parsed = Date.parse(raw);
      if (!Number.isNaN(parsed)) {
        return new Date(parsed).toISOString().slice(0, 10);
      }
    }

    const numeric = toNumber(raw);
    return numeric !== null ? numeric : raw;
  };

  const looksLikeDateColumn = (rows, key) => {
    if (isDateHeader(key)) return true;
    const sample = (rows || [])
      .map((row) => row?.[key])
      .filter((v) => v !== null && v !== undefined && String(v).trim() !== "");
    if (sample.length === 0) return false;
    const sampleSize = Math.min(sample.length, 20);
    const dateCount = sample.slice(0, sampleSize).filter((v) => isDateLikeValue(v)).length;
    return dateCount >= Math.max(1, Math.ceil(sampleSize * 0.6));
  };

  const getCompetitionValue = (row) =>
    String(
      row?.["Current Team Country (Tier)"] ||
      row?.["Competition Name"] ||
      row?.["Competition"] ||
      ""
    ).trim();

  const normalizeCompetitionValue = (value) => String(value || "").trim().toLowerCase();


  const getPlayerBackground = (player) => {
    if (!player) return "";

    const name = player["Player Name"] || player["Full Player Name"] || "The player";
    const team = player["Team"] || player["Display Team"] || "their club";
    const position = player["Primary Position"] || player["Display Position"] || "player";
    const competition = getCompetitionValue(player);
    const age = player["Age"] ? `${player["Age"]}-year-old ` : "";
    const nationality = player["Nationality"] ? `${player["Nationality"]} ` : "";
    const minutes = player["Minutes Played"] ? `${player["Minutes Played"]} minutes` : "limited minutes data";

    return `${name} is a ${age}${nationality}${position.toLowerCase()} at ${team}${competition ? `, competing in ${competition}` : ""}. This draft background paragraph is generated from the uploaded data and is styled to show what a Transfermarkt-based summary could look like later. The player profile currently indicates ${minutes}, which helps frame his recent usage and role within the squad.`;
  };

  const handlePhotoUpload = async () => {
    if (!selectedPlayer || !photoFile) return;

    const playerName =
      selectedPlayer["Full Player Name"] ||
      selectedPlayer["Player Name"] ||
      selectedPlayer["Display Name"] ||
      selectedPlayer.Player ||
      "";

    if (!playerName) {
      setPhotoStatus("No player name found.");
      return;
    }

    const cleanName = getDisplayName(playerName);
    const filename = `${slugify(cleanName)}.png`;
    const path = `player-photos/${filename}`;

    const supabaseUrl = process.env.REACT_APP_SUPABASE_URL || "https://syjsmvvsvvprxibqoizw.supabase.co";
    const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY || "";

    if (!supabaseAnonKey) {
      setPhotoStatus("Missing REACT_APP_SUPABASE_ANON_KEY.");
      return;
    }

    try {
      setUploadingPhoto(true);
      setPhotoStatus("");

      const uploadRes = await fetch(`${supabaseUrl}/storage/v1/object/player-photos/${path}`, {
        method: "POST",
        headers: {
          "Content-Type": "image/png",
          "Authorization": `Bearer ${supabaseAnonKey}`,
          "apikey": supabaseAnonKey,
          "x-upsert": "true",
        },
        body: photoFile,
      });

      if (!uploadRes.ok) {
        const errorText = await uploadRes.text();
        throw new Error(errorText || "Upload failed");
      }

      const publicUrl = `${supabaseUrl}/storage/v1/object/public/player-photos/${path}`;

      setPlayers((prev) => prev.map((p) => (
        (p["Player Label"] === selectedPlayer["Player Label"] ||
         p["Full Player Name"] === selectedPlayer["Full Player Name"]) ? { ...p, _photoUrl: publicUrl, Photo: publicUrl } : p
      )));
      setFilteredPlayers((prev) => prev.map((p) => (
        (p["Player Label"] === selectedPlayer["Player Label"] ||
         p["Full Player Name"] === selectedPlayer["Full Player Name"]) ? { ...p, _photoUrl: publicUrl, Photo: publicUrl } : p
      )));
      setSelectedPlayer((prev) => ({ ...prev, _photoUrl: publicUrl, Photo: publicUrl }));
      setPhotoDataUrl(publicUrl);
      setPhotoStatus("Photo uploaded and saved to Supabase.");
      setPhotoFile(null);
    } catch (err) {
      setPhotoStatus(`Upload failed: ${err?.message || "Unknown error"}`);
    } finally {
      setUploadingPhoto(false);
    }
  };

  const NAME_CANDIDATES = ["Player Name", "Player", "Name", "player", "name", "Footballer"];
  const TEAM_CANDIDATES = ["Team", "Club", "Squad", "team", "club", "Current Team"];
  const LEAGUE_CANDIDATES = ["League", "Competition", "competition", "league", "Competition Name"];
  const POSITION_CANDIDATES = ["Position", "Primary Position", "Role", "position"];
  const MINUTES_CANDIDATES = [
    "Minutes played", "Minutes", "mins", "Min", "minutes played", "Minutes Played", "Minutes (Last 2 years)",
  ];
  const AGE_CANDIDATES = ["Age", "age"];
  const CONTRACT_DAYS_CANDIDATES = [
    "Contract Expiry (days left)", "Contract expiry (days left)", "Contract days left", "Days left on contract",
  ];
  const CONTRACT_DATE_CANDIDATES = ["Contract expires", "Contract Expiry", "Contract end", "Expiry"];
  const PHOTO_CANDIDATES = ["Photo", "_photoUrl", "playerPhoto", "Image", "Photo URL"];

  const META_EXACT = new Set([
    "Player", "Name", "Team", "Club", "Squad", "League", "Competition",
    "Competition Name", "Position", "Primary Position", "Role", "Age",
    "Minutes played", "Minutes", "mins", "Min", "Minutes Played",
    "Contract expires", "Passport country", "Foot", "Height", "Weight",
    "Valuation", "Contract Expiry (days left)", "Woman player no", "Player no",
    "Match no", "Team no", "Season", "Appearances", "90s Played", "Starting Appearances",
    "Photo", "_photoUrl", "playerPhoto", "Image", "Photo URL",
    "Date", "DOB", "Birth Date", "Date of Birth", "Contract Date", "Report Date", "Joined Date",
    "__player_name__", "__team__", "__league__", "__position__", "__age__",
    "__minutes__","__contract_days__", "__contract_date__", "__row_id__",
    "Display Name", "Display Team", "Display League", "Display Position",
    "Transfermarkt Link", "Primary Archetype", "Secondary Archetype", "Player Label",
    "Overall Score"
  ]);

  const META_KEYWORDS = [
    "id", "name", "team", "club", "squad", "player", "match", "season",
    "league", "competition", "birth", "height", "weight", "passport",
    "country", "foot", "shirt", "age", "position", "role", "minute", "photo", "date", "dob", "birth", "expiry", "expires", "created", "updated", "time",
  ];

  const toNumber = (value) => {
    if (value === null || value === undefined) return null;
    const raw = String(value).trim();
    if (!raw) return null;
    const cleaned = raw.replace(/,/g, "");
    const num = Number(cleaned);
    return Number.isFinite(num) ? num : null;
  };

  const uniquePreserveOrder = (items) => {
    const seen = new Set();
    const out = [];
    items.forEach((item) => {
      if (item !== null && item !== undefined && String(item).trim() !== "" && !seen.has(item)) {
        seen.add(item);
        out.push(item);
      }
    });
    return out;
  };

  const findFirstExisting = (row, candidates) =>
    candidates.find((c) => row && Object.prototype.hasOwnProperty.call(row, c) && row[c] !== null && row[c] !== "");

  const isLowerBetterMetric = (metricName) => {
    const text = String(metricName).toLowerCase();
    return ["turnover", "turnovers", "dispossess", "miscontrol", "ball lost", "lost possession", "foul", "error"].some((k) => text.includes(k));
  };

  const normalizeRows = (rows) => {
    const normalized = (rows || []).map((row, idx) => {
      const normalizedRow = {};
      Object.keys(row || {}).forEach((key) => {
        normalizedRow[key] = normalizeUploadedValue(key, row[key]);
      });

      const out = { ...normalizedRow };

      const nameCol = findFirstExisting(normalizedRow, NAME_CANDIDATES) || "Player Name";
      const teamCol = findFirstExisting(normalizedRow, TEAM_CANDIDATES) || "Team";
      const leagueCol = findFirstExisting(normalizedRow, LEAGUE_CANDIDATES) || "Competition Name";
      const positionCol = findFirstExisting(normalizedRow, POSITION_CANDIDATES) || "Primary Position";
      const minutesCol = findFirstExisting(normalizedRow, MINUTES_CANDIDATES) || "Minutes Played";
      const ageCol = findFirstExisting(normalizedRow, AGE_CANDIDATES) || "Age";
      const contractDaysCol = findFirstExisting(normalizedRow, CONTRACT_DAYS_CANDIDATES) || "Contract Expiry (days left)";
      const contractDateCol = findFirstExisting(normalizedRow, CONTRACT_DATE_CANDIDATES) || "Contract expires";
      const photoCol = findFirstExisting(normalizedRow, PHOTO_CANDIDATES) || "Photo";

      const rawPlayerName = String(normalizedRow[nameCol] ?? normalizedRow["Player Name"] ?? normalizedRow["Name"] ?? normalizedRow["Player"] ?? `Player ${idx + 1}`).trim() || `Player ${idx + 1}`;
      const playerName = getDisplayName(rawPlayerName);

      out["Full Player Name"] = rawPlayerName;
      out["Player Name"] = playerName;
      out["Display Name"] = playerName;

      out["Team"] = String(normalizedRow[teamCol] ?? normalizedRow["Team"] ?? normalizedRow["Club"] ?? normalizedRow["Squad"] ?? "").trim();
      out["Display Team"] = out["Team"];

      out["Competition Name"] = String(normalizedRow[leagueCol] ?? normalizedRow["Competition Name"] ?? normalizedRow["League"] ?? normalizedRow["Competition"] ?? "").trim();
      out["Display League"] = out["Competition Name"];

      out["Primary Position"] = String(normalizedRow[positionCol] ?? normalizedRow["Primary Position"] ?? normalizedRow["Position"] ?? normalizedRow["Role"] ?? "").trim();
      out["Display Position"] = out["Primary Position"];

      const ageValue = toNumber(normalizedRow[ageCol] ?? normalizedRow["Age"]);
      if (ageValue !== null) out["Age"] = ageValue;

      const minutesValue = toNumber(normalizedRow[minutesCol] ?? normalizedRow["Minutes Played"]);
      if (minutesValue !== null) out["Minutes Played"] = minutesValue;

      const contractDaysValue = toNumber(normalizedRow[contractDaysCol] ?? normalizedRow["Contract Expiry (days left)"]);
      if (contractDaysValue !== null) out["Contract Expiry (days left)"] = contractDaysValue;

      const contractDateValue = normalizedRow[contractDateCol] ?? normalizedRow["Contract expires"];
      if (contractDateValue) out["Contract expires"] = contractDateValue;

      const photoValue = normalizedRow[photoCol] ?? normalizedRow["Photo"] ?? normalizedRow["_photoUrl"];
      if (photoValue) out["_photoUrl"] = photoValue;

      out["Player Label"] = playerName;
      out["__player_name__"] = rawPlayerName;
      out["__team__"] = out["Team"];
      out["__league__"] = out["Competition Name"];
      out["__position__"] = out["Primary Position"];
      out["__age__"] = ageValue;
      out["__minutes__"] = minutesValue;
      out["__contract_days__"] = contractDaysValue;
      out["__contract_date__"] = contractDateValue;
      out["__row_id__"] = idx;

      return out;
    });

    const counts = normalized.reduce((acc, row) => {
      const key = row["Player Name"] || "";
      if (!key) return acc;
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});

    const labelCounts = {};
    return normalized.map((row, idx) => {
      const name = String(row["Player Name"] || "").trim();
      let label = name;
      if (counts[name] > 1) {
        const suffix = row["Team"] || row["Competition Name"] || `Row ${idx + 1}`;
        label = `${name} (${suffix})`;
      }
      if (labelCounts[label]) {
        label = `${label} #${idx + 1}`;
      }
      labelCounts[label] = true;
      return { ...row, "Player Label": label };
    });
  };

  const inferMetricColumns = (rows) => {
    const metrics = [];
    if (!rows || rows.length === 0) return metrics;

    const firstRow = rows[0];
    Object.keys(firstRow).forEach((col) => {
      if (META_EXACT.has(col)) return;
      const lower = String(col).toLowerCase();
      if (META_KEYWORDS.some((k) => lower.includes(k))) return;
      if (looksLikeDateColumn(rows, col)) return;

      const numericValues = rows
        .map((r) => toNumber(r[col]))
        .filter((v) => v !== null && v !== undefined);

      if (numericValues.length < Math.max(3, Math.floor(rows.length * 0.7))) return;
      if (numericValues.length === 0) return;

      const distinctCount = new Set(numericValues).size;
      if (distinctCount >= 5) metrics.push(col);
    });

    return uniquePreserveOrder(metrics);
  };

  const sortRowsByScore = (rows, metricNames) => {
    if (!rows || rows.length === 0 || !metricNames || metricNames.length === 0) return rows || [];
    return [...rows].sort((a, b) => {
      const aScore = Number(averageRating(a, rows, metricNames)) || 0;
      const bScore = Number(averageRating(b, rows, metricNames)) || 0;
      return bScore - aScore;
    });
  };


  const processRows = (rows) => {
    const parsed = (rows || []).map((p) => {
      const obj = {};
      Object.keys(p || {}).forEach((k) => {
        obj[k] = normalizeUploadedValue(k, p[k]);
      });
      return obj;
    });

    const normalized = normalizeRows(parsed);
    const detected = inferMetricColumns(normalized);

    const ranked = sortRowsByScore(normalized, detected);

    setPlayers(normalized);
    setFilteredPlayers(ranked);
    setMetrics(detected);
    setScatterMetrics({ x: detected[0] || "", y: detected[1] || "" });

    const comps = uniquePreserveOrder(
      ranked
        .map((p) => getCompetitionValue(p))
        .filter(Boolean)
    );
    setCompetitions(["All", ...comps]);

    setSelectedPlayer(ranked[0] || null);
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileName = file.name.toLowerCase();

    if (fileName.endsWith(".xlsx") || fileName.endsWith(".xls")) {
      const buffer = await file.arrayBuffer();
      const workbook = XLSX.read(buffer, { type: "array", cellDates: true });
      const sheetName = workbook.SheetNames?.[0];
      const sheet = sheetName ? workbook.Sheets[sheetName] : null;

      if (!sheet) return;

      const rawRows = XLSX.utils.sheet_to_json(sheet, { defval: "", raw: true });
      processRows(rawRows);
      return;
    }

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        processRows(res.data);
      },
    });
  };

  const applyFilters = () => {
    const f = players.filter((p) =>
      (filters.minMinutes === 0 || (p["Minutes Played"] ?? 0) >= filters.minMinutes) &&
      (filters.maxMinutes === 99999 || (p["Minutes Played"] ?? 0) <= filters.maxMinutes) &&
      (filters.competition === "All" ? true : normalizeCompetitionValue(getCompetitionValue(p)) === normalizeCompetitionValue(filters.competition)) &&
      (filters.positions.length === 0 ? true : filters.positions.includes(String(p["Primary Position"] ?? "")))
    );

    const ranked = sortRowsByScore(f, metrics);
    setFilteredPlayers(ranked);
    if (selectedPlayer && !ranked.includes(selectedPlayer)) {
      setSelectedPlayer(ranked[0] || null);
    } else if (!selectedPlayer && ranked.length > 0) {
      setSelectedPlayer(ranked[0]);
    }
  };

  const percentilesForPlayer = (player, rows = filteredPlayers, metricList = metrics) =>
    metricList.map((m) => {
      const values = rows
        .map((p) => toNumber(p[m]))
        .filter((v) => v !== null && v !== undefined);

      if (values.length === 0 || player[m] === undefined || player[m] === null || player[m] === "") {
        return {
          metric: m,
          value: 0,
          raw: player[m] ?? "",
          leagueAvg: 0,
        };
      }

      let pct = (values.filter((v) => v <= Number(player[m])).length / values.length) * 100;
      if (isLowerBetterMetric(m)) pct = 100 - pct;

      const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
      let avgPct = (values.filter((v) => v <= avgValue).length / values.length) * 100;
      if (isLowerBetterMetric(m)) avgPct = 100 - avgPct;

      return {
        metric: m,
        value: Math.round(pct),
        raw: Number(player[m]).toFixed(2),
        leagueAvg: Math.round(avgPct),
      };
    });

  const averageRating = (player, rows = filteredPlayers, metricList = metrics) => {
    if (!metricList || metricList.length === 0) return 0;
    const pcts = percentilesForPlayer(player, rows, metricList).map((p) => p.value);
    if (!pcts.length) return 0;
    return (pcts.reduce((a, b) => a + b, 0) / pcts.length).toFixed(1);
  };

  const overallRank = (player, rows = filteredPlayers, metricList = metrics) => {
    const ranked = [...rows]
      .map((p) => ({ player: p, rating: Number(averageRating(p, rows, metricList)) }))
      .sort((a, b) => b.rating - a.rating);

    return ranked.findIndex((r) => r.player === player) + 1;
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

  const categoryScore = (player, categoryMetrics, rows = filteredPlayers) => {
    const existingMetrics = categoryMetrics.filter((m) => metrics.includes(m));
    if (existingMetrics.length === 0) return 0;

    const scores = existingMetrics.map((m) => {
      const values = rows.map((p) => toNumber(p[m])).filter((v) => v !== null && v !== undefined);
      if (values.length === 0 || player[m] === undefined || player[m] === null || player[m] === "") return 0;
      let pct = (values.filter((v) => v <= Number(player[m])).length / values.length) * 100;
      if (isLowerBetterMetric(m)) pct = 100 - pct;
      return pct;
    });

    return Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
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

  const uniquePositions = Array.from(new Set(filteredPlayers.map((p) => p["Primary Position"]).filter(Boolean)));

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
      <input type="file" accept=".csv,.xlsx,.xls" onChange={handleUpload} style={{ marginTop: 10 }} />

      {/* Filters + Right Column */}
      <div style={{ marginTop: 20, display:"grid", gridTemplateColumns:"260px 1fr", gap:24, maxWidth:1400, margin:"auto" }}>
        {/* Filter Panel */}
        <div style={{ background:"rgba(255,255,255,0.95)", padding:16, borderRadius:12, boxShadow:"0 4px 12px rgba(0,0,0,0.08)", height:"fit-content" }}>
          <h3 style={{ marginTop:0, marginBottom:12, color:"#1f77b4" }}>Filters</h3>
          <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Min Minutes</label><input type="number" onChange={e => setFilters({...filters, minMinutes:Number(e.target.value)})} /></div>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Max Minutes</label><input type="number" onChange={e => setFilters({...filters, maxMinutes:Number(e.target.value)})} /></div>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Competition</label>
              <select onChange={e=>setFilters({...filters, competition:e.target.value})} style={{ width: "100%", maxWidth: "100%", boxSizing: "border-box" }}>

                {competitions.map(c=><option key={c}>{c}</option>)}
              </select>
            </div>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Position</label>
              <select multiple style={{height:80, width: "100%", maxWidth: "100%", boxSizing: "border-box" }} onChange={e=>setFilters({...filters, positions:Array.from(e.target.selectedOptions, o=>o.value)})}>
                {uniquePositions.map(pos=><option key={pos}>{pos}</option>)}
              </select>
            </div>
            <hr style={{ opacity:0.2 }} />
            <div><label style={{ fontSize:12, fontWeight:600 }}>Scatter X</label>
              <select value={scatterMetrics.x} onChange={e=>setScatterMetrics({...scatterMetrics, x:e.target.value})} style={{ width: "100%", maxWidth: "100%", boxSizing: "border-box" }}>

                {metrics.map(m=><option key={m}>{m}</option>)}
              </select>
            </div>
            <div><label style={{ fontSize:12, fontWeight:600 }}>Scatter Y</label>
              <select value={scatterMetrics.y} onChange={e=>setScatterMetrics({...scatterMetrics, y:e.target.value})} style={{ width: "100%", maxWidth: "100%", boxSizing: "border-box" }}>

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
               src={photoDataUrl || getPlayerPhoto(selectedPlayer)}
               alt={selectedPlayer["Player Name"]}
               crossOrigin="anonymous"
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
              <div>Competition: <strong>{getCompetitionValue(selectedPlayer)}</strong></div>
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

                                <div
                                  style={{
                                    marginTop: 18,
                                    background: "rgba(255,255,255,0.95)",
                                    border: "1px solid #ddd",
                                    borderRadius: 8,
                                    padding: 16,
                                  }}
                                >
                                  <h4 style={{ color: "#1f77b4", marginTop: 0, marginBottom: 10 }}>
                                    Upload Player Photo
                                  </h4>

                                  <input
                                    type="file"
                                    accept="image/png"
                                    onChange={(e) => setPhotoFile(e.target.files?.[0] || null)}
                                    style={{ display: "block", marginBottom: 10 }}
                                  />

                                  <button
                                    type="button"
                                    onClick={handlePhotoUpload}
                                    disabled={!photoFile || uploadingPhoto}
                                    style={{
                                      background: "#1f77b4",
                                      color: "#fff",
                                      border: "none",
                                      padding: "8px 12px",
                                      borderRadius: 6,
                                      cursor: uploadingPhoto ? "not-allowed" : "pointer",
                                      fontWeight: 600,
                                    }}
                                  >
                                    {uploadingPhoto ? "Uploading..." : "Upload to Supabase"}
                                  </button>

                                  {photoStatus && (
                                    <div style={{ marginTop: 8, fontSize: 13, color: "#555" }}>
                                      {photoStatus}
                                    </div>
                                  )}
                                </div>

                                <div
                                  style={{
                                    marginTop: 18,
                                    background: "rgba(255,255,255,0.95)",
                                    border: "1px solid #ddd",
                                    borderRadius: 8,
                                    padding: 16,
                                  }}
                                >
                                  <h4 style={{ color: "#1f77b4", marginTop: 0, marginBottom: 10 }}>
                                    Player Background
                                  </h4>
                                  <p style={{ color: "#000", lineHeight: 1.7, margin: 0 }}>
                                    {getPlayerBackground(selectedPlayer)}
                                  </p>
                                </div>

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












