import { ClubName } from "./ClubBadge";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "./supabaseClient";
import { useAuth } from "./AuthContext";
import { removeBackground } from "@imgly/background-removal";
import ReportImportModal from "./ReportImportModal";
import { splitImportedFixtures } from "./reportImport";
import { choosePlayerRecord, normalizePlayerName, playerNameMatchScore } from "./playerProfileMatching";
const clubFromReport = (...values) => {
  const club = values.find((value) => {
    if (typeof value !== "string") return false;
    const normalized = value.trim().toLowerCase();
    return normalized && !["club not added", "—", "-", "n/a", "unknown"].includes(normalized);
  });
  return typeof club === "string" ? club.trim() : "";
};
const empty = {
  type: "Long Report",
  foot: "",
  playedPosition: "",
  performance: "",
  potential: "",
  conclusion: "",
  reasons: "",
  inPossession: "",
  outPossession: "",
  physical: "",
  behaviour: "",
  strengths: "",
  weaknesses: "",
};
const readProfileCheckerFilters = () => {
  const defaults = { foot: "", age: "", position: "", performance: [], potential: [] };
  try {
    const saved = JSON.parse(sessionStorage.getItem("profileCheckerFilters") || "{}");
    return {
      ...defaults,
      ...saved,
      performance: Array.isArray(saved.performance) ? saved.performance.map(String) : saved.performance ? [String(saved.performance)] : [],
      potential: Array.isArray(saved.potential) ? saved.potential : saved.potential ? [saved.potential] : [],
    };
  } catch {
    return defaults;
  }
};
const photoBase =
  "https://syjsmvvsvvprxibqoizw.supabase.co/storage/v1/object/public/player-photos/player-photos/";
const photoSlug = (name, lower = false) =>
  String(name || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((x) =>
      x
        .normalize("NFD")
        .replace(/[̀-ͯ]/g, "")
        .replace(/[^a-zA-Z0-9]+/g, "_")
        [lower ? "toLowerCase" : "toString"](),
    )
    .join("_");
const playerPhoto = (name, lower = true) =>
  `${photoBase}${photoSlug(name, lower)}.png`;
const playerPhotoFor = (record) => {
  const direct = ["Photo", "photo", "_photoUrl", "photoUrl", "playerPhoto", "Image", "image", "Photo URL"]
    .map((key) => record?.[key])
    .find((value) => typeof value === "string" && value.trim());
  return direct || playerPhoto(record?.Name || record?.name || record?.player);
};
const photoCandidates = (name) => {
  const raw = String(name || "").trim();
  const parts = raw.split(/\s+/).filter(Boolean);
  const display = parts.length > 2 ? `${parts[0]} ${parts.at(-1)}` : raw;
  const twoNameVariants = parts.length > 2 ? parts.slice(0, -1).map((_, index) => parts.slice(index, index + 2).join(" ")) : [];
  const nameVariants = [raw, ...twoNameVariants, display];
  const rawBases = nameVariants.map((value) => value.replace(/[^\p{L}\p{N}]+/gu, "_").replace(/^_+|_+$/g, ""));
  const normalizedBases = rawBases.map((value) => value.normalize("NFD").replace(/[̀-ͯ]/g, ""));
  // The original photo downloader made filenames by replacing each non-ASCII
  // character with an underscore (for example, Rúben Dias -> r_ben_dias).
  const legacyBases = nameVariants.map((value) => value.trim().replace(/[^a-z0-9]/gi, "_").toLowerCase().replace(/^_+|_+$/g, ""));
  const bases = [...new Set([...rawBases, ...normalizedBases, ...legacyBases].filter(Boolean))];
  return [...new Set(bases.flatMap((base) => [base, base.toLowerCase(), base.toUpperCase(), `_${base}`, `_${base.toLowerCase()}`, `__${base}`]).map((base) => `${photoBase}${base}.png`))];
};
const retryPhoto = (e, name) => {
  const image = e.currentTarget;
  const candidates = photoCandidates(name);
  const next = Number(image.dataset.photoFallback || 0) + 1;
  if (candidates[next - 1]) { image.dataset.photoFallback = String(next); image.src = candidates[next - 1]; }
  else image.style.display = "none";
};
const focusPlayerCutout = (blob) => new Promise((resolve) => {
  const url = URL.createObjectURL(blob);
  const image = new Image();
  image.onload = () => {
    const source = document.createElement("canvas");
    source.width = image.naturalWidth; source.height = image.naturalHeight;
    const sourceContext = source.getContext("2d"); sourceContext.drawImage(image, 0, 0);
    const pixels = sourceContext.getImageData(0, 0, source.width, source.height).data;
    let left = source.width, top = source.height, right = 0, bottom = 0;
    for (let y = 0; y < source.height; y += 2) for (let x = 0; x < source.width; x += 2) {
      if (pixels[(y * source.width + x) * 4 + 3] > 18) { left = Math.min(left, x); top = Math.min(top, y); right = Math.max(right, x); bottom = Math.max(bottom, y); }
    }
    URL.revokeObjectURL(url);
    if (right <= left || bottom <= top) { resolve(blob); return; }
    const paddingX = Math.round((right - left) * 0.08);
    const cropX = Math.max(0, left - paddingX);
    const cropWidth = Math.min(source.width - cropX, right - left + paddingX * 2);
    const subjectHeight = bottom - top;
    const cropHeight = Math.min(source.height - top, Math.max(Math.round(subjectHeight * 0.78), Math.round(cropWidth * 1.15)));
    const crop = document.createElement("canvas"); crop.width = cropWidth; crop.height = cropHeight;
    crop.getContext("2d").drawImage(source, cropX, top, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
    crop.toBlob((result) => resolve(result || blob), "image/png");
  };
  image.onerror = () => { URL.revokeObjectURL(url); resolve(blob); };
  image.src = url;
});
const profileFields = ["foot", "playedPosition", "performance", "potential", "conclusion", "reasons", "inPossession", "outPossession", "physical", "behaviour", "strengths", "weaknesses"];
const stopWords = new Set(["a", "an", "and", "are", "as", "at", "for", "from", "good", "has", "have", "in", "is", "looking", "of", "player", "that", "the", "to", "with"]);
const profileTerms = (value) => String(value || "").toLowerCase().replace(/[^a-z0-9%+.#-]+/g, " ").split(/\s+/).filter((term) => term.length > 1 && !stopWords.has(term));
const reportSearchText = (item) => {
  const report = item.report || {};
  return [item.player, item.club, item.position, item.foot, item.age, item.performance, item.potential, ...Object.values(item), ...profileFields.map((field) => report[field])].filter((value) => typeof value === "string" || typeof value === "number").join(" ").toLowerCase();
};
const profileEvidence = (item, terms) => {
  const report = item.report || {};
  return profileFields.map((field) => String(report[field] || "").trim()).find((value) => terms.some((term) => value.toLowerCase().includes(term))) || "Report matches the requested profile criteria.";
};
const isFixtureName = (value) => {
  const name = String(value || "").trim();
  return Boolean(name) && !/^(multiple|fixture not added|date not added)$/i.test(name);
};
const originalPublishedFixtures = {
  "tobias bech kristensen": [
    { name: "AGF v MIDTJYLLAND", date: "2026-09-02" },
    { name: "AGF v BENFICA", date: "2026-08-27" },
    { name: "HORSENS v AGF", date: "2026-09-20" },
  ],
  "ignacio maestro puch": [
    { name: "PUEBLA v ATLANTE", date: "2026-09-19" },
    { name: "PUEBLA v TOLUCA", date: "2026-09-16" },
    { name: "TIGRES UANL v PUEBLA", date: "2026-09-27" },
  ],
};
const originalFixturesFor = (item) => originalPublishedFixtures[
  String(item?.player || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ")
] || [];
const storedFixtures = (item) => {
  const candidates = [item?.games, item?.report?.__fixtures];
  for (const candidate of candidates) {
    if (Array.isArray(candidate) && candidate.some((fixture) => isFixtureName(typeof fixture === "string" ? fixture : fixture?.name))) return candidate;
  }
  return [];
};
const fixtureRecords = (item) => {
  const saved = storedFixtures(item);
  const recovered = originalFixturesFor(item);
  const fallbackName = isFixtureName(item?.game) ? item.game : isFixtureName(item?.fixture_summary) ? item.fixture_summary : "";
  const records = saved.length ? saved : (recovered.length ? recovered : (fallbackName ? [{ name: fallbackName, date: item.fixtureDates?.[0] || item.date || "" }] : []));
  return records.flatMap((fixture, index) => {
    const name = typeof fixture === "string" ? fixture : fixture?.name || "";
    if (!isFixtureName(name)) return [];
    const date = typeof fixture === "string" ? item?.fixtureDates?.[index] || item?.date || "" : fixture?.date || item?.fixtureDates?.[index] || item?.date || "";
    const split = splitImportedFixtures(name, date);
    return split.length ? split : (name ? [{ name, date }] : []);
  });
};
const fixtureLabel = (item) => {
  const fixtures = fixtureRecords(item);
  if (fixtures.length > 1) return "Multiple";
  if (fixtures.length === 1) {
    if (typeof fixtures[0] === "string") {
      return item?.fixtureDates?.[0] ? `${item.fixtureDates[0]} · ${fixtures[0]}` : fixtures[0];
    }
    const name = fixtures[0]?.name || "Fixture not added";
    return fixtures[0]?.date ? `${fixtures[0].date} · ${name}` : name;
  }
  return item?.fixtureDates?.[0] && item?.game
    ? `${item.fixtureDates[0]} · ${item.game}`
    : item?.game || item?.fixture_summary || "Fixture not added";
};
const cardFixtureLabel = (item) => {
  const fixtures = fixtureRecords(item);
  if (fixtures.length > 1) {
    const reportDate = item?.completedAt || item?.completed_at || item?.date || fixtures[0]?.date;
    return reportDate ? `${reportDate} · Multiple` : "Multiple";
  }
  return fixtureLabel(item);
};
const assignmentBelongsToScout = (assignment, user, accountProfile) => {
  if (!assignment?.scoutId) return true;
  const ids = [user?.id, accountProfile?.id].filter(Boolean).map((value) => String(value));
  const names = [accountProfile?.full_name, user?.email].filter(Boolean).map((value) => String(value).trim().toLowerCase());
  return ids.includes(String(assignment.scoutId)) || names.includes(String(assignment.scout || "").trim().toLowerCase());
};
const playerDobKeys = ["DOB", "Date of Birth", "date_of_birth", "dateOfBirth", "date of birth", "birth_date", "birthDate"];
const ageFromValue = (value) => {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const text = String(value).trim();
  if (!/^\d{1,3}$/.test(text)) return null;
  const age = Number(text);
  return age >= 0 && age <= 120 ? age : null;
};
const ageFromDob = (value) => {
  if (!value) return null;
  let year, month, day;
  const text = String(value).trim();
  const iso = text.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  const dayFirst = text.match(/^(\d{1,2})[/. -](\d{1,2})[/. -](\d{4})$/);
  if (iso) [, year, month, day] = iso;
  else if (dayFirst) [, day, month, year] = dayFirst;
  else {
    const parsed = new Date(text);
    if (Number.isNaN(parsed.getTime())) return null;
    year = parsed.getFullYear(); month = parsed.getMonth() + 1; day = parsed.getDate();
  }
  year = Number(year); month = Number(month); day = Number(day);
  const dob = new Date(year, month - 1, day);
  if (dob.getFullYear() !== year || dob.getMonth() !== month - 1 || dob.getDate() !== day) return null;
  const today = new Date();
  let age = today.getFullYear() - year;
  if (today.getMonth() + 1 < month || (today.getMonth() + 1 === month && today.getDate() < day)) age -= 1;
  return age >= 0 ? age : null;
};
const matchesAgeFilter = (age, filter) => {
  if (!filter) return true;
  if (age === null || age === undefined || !Number.isFinite(age)) return false;
  if (filter === "under21") return age < 21;
  if (filter === "21to24") return age >= 21 && age <= 24;
  if (filter === "25to29") return age >= 25 && age <= 29;
  if (filter === "30plus") return age >= 30;
  return false;
};
const reportSummaryPhrases = (values, limit = 3) => {
  const seen = new Set();
  return values
    .flatMap((value) => Array.isArray(value) ? value : String(value || "").split(/[\n•]+/))
    .map((value) => String(value || "").replace(/^\s*(?:[-*]\s*|\d+[.)]\s*)/, "").trim())
    .filter((value) => {
      if (!value) return false;
      const key = value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(0, limit);
};
const playerSummaryText = (player, profile, reports = []) => {
  const source = player || profile || {};
  const reportRecords = reports.map((item) => item?.report || item || {});
  if (reportRecords.length) {
    const name = source.Name || source.name || profile?.player || "This player";
    const strengths = reportSummaryPhrases(reportRecords.map((report) => report.strengths));
    const weaknesses = reportSummaryPhrases(reportRecords.map((report) => report.weaknesses));
    const conclusions = reportSummaryPhrases(reportRecords.map((report) => report.conclusion), 2);
    const performanceGrades = reportSummaryPhrases(reportRecords.map((report) => report.performance), 5)
      .filter((grade) => /^[1-5](?:\.0)?$/.test(grade))
      .map(Number);
    const potentialGrades = reportSummaryPhrases(reportRecords.map((report) => report.potential), 6)
      .filter((grade) => /^[A-F]$/i.test(grade))
      .map((grade) => grade.toUpperCase());
    const sentences = [`Across ${reportRecords.length} published scouting report${reportRecords.length === 1 ? "" : "s"}, scouts have assessed ${name}.`];
    if (strengths.length) sentences.push(`Reported strengths include ${strengths.join(", ")}.`);
    if (weaknesses.length) sentences.push(`Areas identified for improvement include ${weaknesses.join(", ")}.`);
    if (conclusions.length) sentences.push(`Scout conclusions: ${conclusions.join(" ")}`);
    if (performanceGrades.length) {
      const low = Math.min(...performanceGrades);
      const high = Math.max(...performanceGrades);
      sentences.push(`Performance grades range from ${low} to ${high} out of 5.`);
    }
    if (potentialGrades.length) sentences.push(`Potential grades recorded include ${reportSummaryPhrases(potentialGrades, 6).join(", ")}.`);
    if (strengths.length || weaknesses.length || conclusions.length || performanceGrades.length || potentialGrades.length) return sentences.join(" ");
  }
  const existingSummary = ["Player Summary", "player_summary", "playerSummary", "AI Summary", "ai_summary", "Generated Summary", "background_summary", "Summary", "summary"]
    .map((key) => source[key])
    .find((value) => typeof value === "string" && value.trim());
  if (existingSummary) return existingSummary.trim();
  const name = source.Name || source.name || profile?.player || "This player";
  const age = ageFromDob(playerDobKeys.map((key) => source[key]).find((value) => value)) ??
    ageFromValue(source.Age ?? source.age);
  const nationality = source.Nationality || source.nationality;
  const position = source["Playing Position"] || source["Primary Position"] || source.Position || source.position || source.primary_position;
  const club = source.Club || source.club || source.Team || source.team || profile?.club;
  const league = source.League || source.league || source.Competition || source.competition;
  const minutesValue = source["Minutes Played"] ?? source["Minutes played"] ?? source.minutes_played ?? source.Minutes ?? source.minutes;
  const minutes = Number.isFinite(Number(minutesValue)) && String(minutesValue ?? "").trim() !== ""
    ? `${Math.round(Number(minutesValue)).toLocaleString()} minutes`
    : "";
  const identity = [age !== null ? `${age}-year-old` : "", position].filter(Boolean).join(" ");
  let summary = `${name}${identity ? ` is a ${identity}` : nationality ? ` is from ${nationality}` : ""}`;
  if (identity && nationality) summary += ` from ${nationality}`;
  if (club) summary += ` at ${club}`;
  summary += ".";
  if (league) summary += ` The player competes in ${league}.`;
  if (minutes) summary += ` The available data records ${minutes} played.`;
  return summary;
};
export default function ScoutingReportsPageFinal() {
  const nav = useNavigate();
  const { appState, updateAppState, user, profile: accountProfile } = useAuth();
  const reportViewStorageKey = `scoutingReportsView:${user?.id || user?.email || "guest"}`;
  const reportImportStorageKey = `scoutingReportImport:${user?.id || user?.email || "guest"}`;
  const [items, setItems] = useState(() => JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"));
  const [tab, setTab] = useState(() => localStorage.getItem(`${reportViewStorageKey}:tab`) || sessionStorage.getItem("scoutingReportsTab") || "My Assignments");
  const [query, setQuery] = useState("");
  const [playerSearch, setPlayerSearch] = useState("");
  const [playerMatches, setPlayerMatches] = useState([]);
  const [profile, setProfile] = useState(() => {
    const n = JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"), name = localStorage.getItem("scoutingProfilePlayer");
    return name ? n.find((x) => x.player === name) || { player: name } : null;
  });
  const [active, setActive] = useState(() => {
    try {
      const saved = JSON.parse(sessionStorage.getItem("scoutingActiveReport") || "null");
      return saved?.active ? { ...saved.active, report: saved.report || saved.active.report || null } : null;
    } catch { return null; }
  });
  const [sharedReports, setSharedReports] = useState([]);
  const [sharedAssignments, setSharedAssignments] = useState([]);
  const [playerData, setPlayerData] = useState(null);
  const [playerDataLoading, setPlayerDataLoading] = useState(false);
  const [playerDataError, setPlayerDataError] = useState("");
  const [ageDate, setAgeDate] = useState(() => new Date());
  const [summaryDraft, setSummaryDraft] = useState("");
  const [summaryEditing, setSummaryEditing] = useState(false);
  const [photoUploading, setPhotoUploading] = useState(false);
  const [photoStatus, setPhotoStatus] = useState("");
  const [profileQuery, setProfileQuery] = useState(() => sessionStorage.getItem("profileCheckerQuery") || "");
  const [profileResults, setProfileResults] = useState([]);
  const [profilePlayerDirectory, setProfilePlayerDirectory] = useState({});
  const [dashboardView, setDashboardView] = useState(() => localStorage.getItem(`${reportViewStorageKey}:view`) || sessionStorage.getItem("scoutingDashboardView") || "reports");
  const [reportImportOpen, setReportImportOpen] = useState(() => sessionStorage.getItem(`${reportImportStorageKey}:open`) === "true");
  const [importNotice, setImportNotice] = useState("");
  const [actionsMenuOpen, setActionsMenuOpen] = useState(false);
  const actionsMenuRef = useRef(null);
  const [profileFilters, setProfileFilters] = useState(readProfileCheckerFilters);
  const profileFiltersActive = Object.entries(profileFilters).some(([key, value]) => ["performance", "potential"].includes(key) ? value.length > 0 : Boolean(value));
  const summaryKey = String(playerData?.id || playerData?.player_id || profile?.playerId || normalizePlayerName(profile?.player));
  const savedPlayerSummary = appState?.playerSummaries?.[summaryKey];
  const summaryReports = [...new Map([
    ...items.filter((item) => item.player === profile?.player && String(item.status || "").toLowerCase() === "published"),
    ...sharedReports.filter((item) => item.player === profile?.player && String(item.status || "Published").toLowerCase() === "published"),
  ].map((item) => [String(item.assignment_id || item.id || `${item.player}-${item.completed_at || item.date || ""}`), item])).values()];
  const generatedPlayerSummary = playerSummaryText(playerData, profile, summaryReports);
  const shownPlayerSummary = typeof savedPlayerSummary === "string" ? savedPlayerSummary : generatedPlayerSummary;
  useEffect(() => {
    setSummaryDraft(shownPlayerSummary);
    setSummaryEditing(false);
  }, [summaryKey, playerData, savedPlayerSummary, shownPlayerSummary]);
  useEffect(() => {
    localStorage.setItem(`${reportViewStorageKey}:view`, dashboardView);
    sessionStorage.setItem("scoutingDashboardView", dashboardView);
  }, [dashboardView, reportViewStorageKey]);
  useEffect(() => {
    localStorage.setItem(`${reportViewStorageKey}:tab`, tab);
    sessionStorage.setItem("scoutingReportsTab", tab);
  }, [tab, reportViewStorageKey]);
  useEffect(() => {
    sessionStorage.setItem(`${reportImportStorageKey}:open`, String(reportImportOpen));
  }, [reportImportOpen, reportImportStorageKey]);
  useEffect(() => {
    if (!actionsMenuOpen) return undefined;
    const closeOnOutsideClick = (event) => {
      if (!actionsMenuRef.current?.contains(event.target)) setActionsMenuOpen(false);
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setActionsMenuOpen(false);
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [actionsMenuOpen]);
  useEffect(() => { sessionStorage.setItem("profileCheckerFilters", JSON.stringify(profileFilters)); }, [profileFilters]);
  useEffect(() => { sessionStorage.setItem("profileCheckerQuery", profileQuery); }, [profileQuery]);
  useEffect(() => {
    const now = new Date();
    const nextMidnight = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1, 0, 0, 1);
    const timer = setTimeout(() => setAgeDate(new Date()), nextMidnight.getTime() - now.getTime());
    return () => clearTimeout(timer);
  }, [ageDate]);
  useEffect(() => {
    const selectedPlayer = profile || active;
    if (!selectedPlayer?.player) {
      setPlayerData(null);
      setPlayerDataLoading(false);
      setPlayerDataError("");
      return;
    }
    let cancelled = false;
    const loadMatchedPlayer = async () => {
      setPlayerData(null);
      setPlayerDataLoading(true);
      setPlayerDataError("");
      const playerId = selectedPlayer.playerId || selectedPlayer.player_id || selectedPlayer["Player Id"];
      const expectedClub = selectedPlayer.player_club || selectedPlayer.playerClub || selectedPlayer.club || selectedPlayer.Club || selectedPlayer.team || selectedPlayer.Team;
      let directRow = null;
      if (playerId !== undefined && playerId !== null && String(playerId).trim()) {
        const { data } = await supabase.from("players").select("*").eq("id", playerId).limit(2);
        const match = (data || []).find((row) => playerNameMatchScore(selectedPlayer.player, row.Name || row.name || row.player_name || row.player) > 0);
        if (match) directRow = match;
      }
      if (directRow) {
        if (!cancelled) {
          setPlayerData(directRow);
          setPlayerDataLoading(false);
        }
        return;
      }

      let lastQueryError = null;
      const queryMatchingPlayers = async (pattern, limit) => {
        let successfulQuery = false;
        for (const column of ["Name", "name", "player_name", "player"]) {
          const { data, error } = await supabase.from("players").select("*").ilike(column, pattern).limit(limit);
          if (error) {
            lastQueryError = error;
            continue;
          }
          successfulQuery = true;
          const matches = (data || []).filter((row) => playerNameMatchScore(selectedPlayer.player, row.Name || row.name || row.player_name || row.player) > 0);
          if (matches.length) return { rows: matches, error: null };
        }
        return { rows: [], error: successfulQuery ? null : lastQueryError };
      };

      let { rows, error } = await queryMatchingPlayers(selectedPlayer.player, 100);
      if (!rows.length) {
        const lastName = normalizePlayerName(selectedPlayer.player).split(" ").filter(Boolean).at(-1);
        if (lastName && lastName.length > 1) {
          const fallback = await queryMatchingPlayers(`%${lastName}%`, 100);
          rows = fallback.rows;
          error = fallback.error;
        }
      }
      const matchedRecord = choosePlayerRecord(rows, selectedPlayer.player, expectedClub);
      if (!cancelled) {
        setPlayerData(matchedRecord);
        setPlayerDataError(error?.message || (!matchedRecord && rows.length > 1
          ? "Several database records match this name, but none could be selected confidently. Check the player’s name and club in the player database."
          : ""));
        setPlayerDataLoading(false);
      }
    };
    loadMatchedPlayer().catch((error) => {
      if (!cancelled) {
        setPlayerData(null);
        setPlayerDataError(error?.message || "Could not load this player from the database.");
        setPlayerDataLoading(false);
      }
    });
    return () => { cancelled = true; };
  }, [profile, active]);
  const uploadProfilePhoto = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !profile?.player) return;
    const filename = photoSlug(profile.player, true);
    const path = `player-photos/${filename}.png`;
    try {
      setPhotoUploading(true); setPhotoStatus("Removing background…");
      const processedFile = await removeBackground(file);
      const focusedFile = await focusPlayerCutout(processedFile);
      setPhotoStatus("Uploading cutout…");
      const { error: uploadError } = await supabase.storage.from("player-photos").upload(path, focusedFile, { upsert: true, contentType: "image/png" });
      if (uploadError) throw uploadError;
      const { data: publicData } = supabase.storage.from("player-photos").getPublicUrl(path);
      const publicUrl = publicData.publicUrl;
      const id = playerData?.id || playerData?.player_id;
      const updateQuery = id ? supabase.from("players").update({ Photo: publicUrl }).eq("id", id) : supabase.from("players").update({ Photo: publicUrl }).ilike("Name", profile.player);
      const { error: updateError } = await updateQuery;
      if (updateError) console.warn("Photo uploaded, but player record was not updated", updateError);
      setPlayerData((current) => ({ ...(current || {}), Photo: publicUrl }));
      setPhotoStatus("Photo added");
    } catch (error) {
      const message = error?.message || "Please try again";
      setPhotoStatus(message.toLowerCase().includes("row-level security") ? "Upload blocked by Supabase permissions. Run the included player-photo policy SQL, then try again." : `Upload failed: ${message}`);
    } finally { setPhotoUploading(false); event.target.value = ""; }
  };
  const [report, setReport] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem("scoutingActiveReport") || "null")?.report || empty; } catch { return empty; }
  });
  const [publishError, setPublishError] = useState("");
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!active) return;
    // Restore the report and its permissions together after a tab switch or refresh.
    setEditing(active.status !== "Published" && assignmentBelongsToScout(active, user, accountProfile));
  }, [active, user, accountProfile]);
  useEffect(() => {
    if (active) sessionStorage.setItem("scoutingActiveReport", JSON.stringify({ active, report }));
  }, [active, report]);
  const publishedAssignmentIds = useMemo(
    () => new Set(sharedReports.map((x) => String(x.assignment_id || x.id))),
    [sharedReports],
  );
  useEffect(() => {
    // Keep the local bootstrap list visible until the cloud state has loaded.
    // Otherwise the first null appState render can briefly (or permanently,
    // after a quick navigation) replace a newly-created assignment with [].
    if (!appState) return;
    const localAssignments = Array.isArray(appState.assignments) ? appState.assignments : [];
    const combined = [...localAssignments, ...sharedAssignments]
      .filter((assignment) => String(assignment?.status || "").toLowerCase() !== "published")
      .filter((assignment) => !publishedAssignmentIds.has(String(assignment?.id)));
    setItems([...new Map(combined.map((assignment) => [String(assignment.id), assignment])).values()]);
  }, [appState, sharedAssignments, publishedAssignmentIds]);
  useEffect(() => {
    if (!user || !accountProfile?.club) return;
    supabase
      .from("club_assignments")
      .select("id, assigned_to, assignment, status")
      .eq("club", accountProfile.club)
      .then(({ data, error }) => {
        if (error) {
          console.error("Could not load shared assignments", error);
          return;
        }
        const loadedAssignments = (data || []).map((row) => ({
          ...(row.assignment || {}),
          id: row.assignment?.id || row.id,
          sharedAssignmentId: row.id,
          scoutId: row.assigned_to || row.assignment?.scoutId,
          status: row.status || row.assignment?.status || "Not Started",
        }));
        setSharedAssignments([...new Map(loadedAssignments.map((assignment) => [String(assignment.player || assignment.id).trim().toLowerCase(), assignment])).values()]);
      });
  }, [user, accountProfile?.club]);
  useEffect(() => {
    if (!user || !accountProfile?.club) return;
    supabase.from("club_reports").select("*").eq("club", accountProfile.club).eq("status", "Published").then(({ data, error }) => {
      if (error) console.error("Could not load club reports", error);
      const published = (data || []).map((item) => {
        if (storedFixtures(item).length) return item;
        const fixtures = originalFixturesFor(item);
        if (!fixtures.length) return item;
        const fixtureDates = fixtures.map((fixture) => fixture.date);
        const report = { ...(item.report || {}), __fixtures: fixtures, __fixtureDates: fixtureDates };
        const reportId = item.id || item.assignment_id;
        if (reportId !== undefined && reportId !== null) {
          supabase.from("club_reports").update({ report, fixture_summary: "Multiple" }).eq("id", reportId)
            .then(({ error: repairError }) => { if (repairError) console.error("Could not restore the original report fixtures", repairError); });
          supabase.from("club_assignments").select("assignment").eq("club", accountProfile.club).eq("id", reportId).limit(1)
            .then(({ data: assignmentRows, error: assignmentReadError }) => {
              const assignment = assignmentRows?.[0]?.assignment;
              if (assignmentReadError || !assignment) return;
              const restoredAssignment = {
                ...assignment,
                games: fixtures,
                fixtureDates,
                game: "Multiple",
                report: { ...(assignment.report || {}), __fixtures: fixtures, __fixtureDates: fixtureDates },
              };
              supabase.from("club_assignments").update({ assignment: restoredAssignment }).eq("club", accountProfile.club).eq("id", reportId)
                .then(({ error: assignmentRepairError }) => { if (assignmentRepairError) console.error("Could not restore assignment fixtures", assignmentRepairError); });
            });
        }
        return { ...item, report, games: fixtures, fixtureDates, game: "Multiple", fixture_summary: "Multiple" };
      });
      setSharedReports(published);
    });
  }, [user, accountProfile?.club]);
  const [shortlistPicker, setShortlistPicker] = useState(false);
  const [selectedShortlist, setSelectedShortlist] = useState("");
  const [selectedPosition, setSelectedPosition] = useState("CF-0");
  useEffect(() => {
    if (playerSearch.trim().length < 2) {
      setPlayerMatches([]);
      return;
    }
    supabase
      .from("players")
      .select("*")
      .ilike("Name", `%${playerSearch.trim()}%`)
      .limit(8)
      .then(({ data }) => setPlayerMatches([...new Map((data || []).map((p) => [String(p.Name || p.name || "").trim().toLowerCase(), p])).values()]))
      .catch(() => setPlayerMatches([]));
  }, [playerSearch]);
  useEffect(() => {
    if (active) setReport(active.report || { ...empty });
  }, [active]);
  const saveItems = (n) => {
    setItems(n);
    updateAppState({ ...appState, assignments: n });
  };
  const importOldReports = async (records) => {
    const createdAt = new Date().toISOString();
    const imported = records.map((record, index) => {
      const game = String(record.game || "").trim();
      const date = String(record.date || "").trim();
      const fixture = splitImportedFixtures(game, date);
      const fixtureDates = fixture.map((item) => item.date).filter(Boolean);
      return {
        id: Date.now() + index,
        playerId: record.playerId || record.playerMatch?.id || record.playerMatch?.player_id || null,
        player: String(record.player || "").trim(),
        club: String(record.club || "").trim(),
        position: String(record.position || record.report?.playedPosition || "").trim(),
        scout: accountProfile?.full_name || user?.email || "Imported report",
        scoutId: user?.id || "",
        game,
        games: fixture,
        fixtureDates,
        date,
        viewing: record.viewing || "Video",
        status: "Draft",
        importedAt: createdAt,
        importSource: record.sourceFile || "File upload",
        importSheet: record.sourceSheet || "",
        report: {
          ...empty,
          ...(record.report || {}),
          type: record.report?.type || "Long Report",
          playedPosition: record.report?.playedPosition || record.position || "",
          __fixtures: fixture,
          __fixtureDates: fixtureDates,
          __importSource: record.sourceFile || "File upload",
        },
      };
    });
    saveItems([...items, ...imported]);
    setImportNotice(`${imported.length} old report${imported.length === 1 ? "" : "s"} imported as drafts.`);
    setReportImportOpen(false);
  };
  const deleteAssignment = async (assignment) => {
    const assignmentId = String(assignment?.id);
    const next = items.filter((item) => String(item?.id) !== assignmentId);
    saveItems(next);
    setSharedAssignments((current) => current.filter((item) => String(item?.id) !== assignmentId));
    if (user && accountProfile?.club) {
      const { error } = await supabase
        .from("club_assignments")
        .delete()
        .eq("club", accountProfile.club)
        .eq("id", Number(assignment?.sharedAssignmentId || assignment?.id));
      if (error) console.error("Could not delete shared assignment", error);
    }
  };
  const shortlistPositions = [
    "GK",
    "LB",
    "LCB",
    "CB",
    "RCB",
    "RB",
    "DM",
    "LM",
    "LCM",
    "CM",
    "RCM",
    "RM",
    "LW",
    "AM",
    "RW",
    "CF",
    "ST",
  ];
  const addProfileToShortlist = () => {
    const lists = appState?.shortlists || [];
    const list = lists.find((x) => String(x.id) === String(selectedShortlist));
    if (!list) return;
    const id = String(playerData?.id || playerData?.player_id || profile.playerId || profile.player);
    const role = String(selectedPosition).split("-")[0];
    const usedSlots = (list.players || []).filter((x) => String(x.slot || "").split("-")[0] === role).map((x) => Number(String(x.slot).split("-")[1])).filter(Number.isFinite);
    let slotIndex = 0;
    while (usedSlots.includes(slotIndex)) slotIndex += 1;
    const slot = `${role}-${slotIndex}`;
    const playerClub = dataValue("club", "Club", "team", "Team");
    const player = {
      ...profile,
      id,
      player: playerData?.Name || playerData?.name || profile.player,
      club: playerClub === "—" ? profile.club || "" : playerClub,
      slot,
      tags: [],
    };
    const next = {
      ...list,
      players: [
        ...(list.players || []).filter(
          (x) => !(String(x.id) === id && x.slot === slot),
        ),
        player,
      ],
    };
    updateAppState({ ...appState, shortlists: lists.map((x) => (String(x.id) === String(list.id) ? next : x)) });
    setShortlistPicker(false);
  };
  const shown = useMemo(
    () => {
      const shared = sharedReports.map((x) => {
        const assignmentId = x.assignment_id || x.id;
        const assignment = sharedAssignments.find((candidate) => String(candidate.sharedAssignmentId || candidate.id) === String(assignmentId))
          || sharedAssignments.find((candidate) => normalizePlayerName(candidate.player) === normalizePlayerName(x.player));
        const fixtures = storedFixtures(x).length ? storedFixtures(x) : storedFixtures(assignment);
        const fixtureDates = [x.fixtureDates, assignment?.fixtureDates, x.report?.__fixtureDates, assignment?.report?.__fixtureDates].find((dates) => Array.isArray(dates) && dates.length) || [];
        const report = fixtures.length ? { ...(x.report || {}), __fixtures: fixtures, __fixtureDates: fixtureDates } : (x.report || {});
        return {
        ...assignment,
        ...x,
        id: assignmentId,
        report,
        games: fixtures,
        fixtureDates,
        game: isFixtureName(x.game) ? x.game : isFixtureName(assignment?.game) ? assignment.game : fixtures.length > 1 ? "Multiple" : fixtures[0]?.name || "",
        status: "Published",
        club: clubFromReport(
          x.player_club, x.playerClub, x.team, x.Team,
          x.report?.player_club, x.report?.playerClub, x.report?.club,
          x.report?.Club, x.report?.team, x.report?.Team,
        ) || "Club not added",
        position: x.position || x.report?.playedPosition || "",
        date: x.completed_at,
        scout: x.scout,
      };
      });
      const source = tab === "Published" ? shared : items;
      const currentIds = [user?.id, accountProfile?.id].filter(Boolean).map((value) => String(value));
      const currentNames = [accountProfile?.full_name, user?.email].filter(Boolean).map((value) => String(value).trim().toLowerCase());
      const belongsToCurrentScout = (assignment) => {
        if (tab !== "My Assignments") return true;
        if (!assignment.scoutId) return true;
        const assignedId = String(assignment.scoutId);
        return currentIds.includes(assignedId) || currentNames.includes(String(assignment.scout || "").trim().toLowerCase());
      };
      return [...new Map(source.map((x) => [String(x.id), x])).values()]
        .filter(belongsToCurrentScout)
        .filter((x) => {
          const club = clubFromReport(x.player_club, x.playerClub, x.club, x.Club, x.team, x.Team) ||
            profilePlayerDirectory[normalizePlayerName(x.player)]?.club || "";
          return `${x.player} ${club} ${x.scout}`
            .toLowerCase()
            .includes(query.toLowerCase());
        })
        .filter((x) =>
          tab === "Published"
            ? x.status === "Published"
            : x.status !== "Published",
        );
    },
    [items, query, tab, sharedReports, sharedAssignments, profilePlayerDirectory, user?.id, user?.email, accountProfile?.id, accountProfile?.full_name],
  );
  const profileCandidates = useMemo(() => {
    const publishedLocal = items.filter((item) => item.status === "Published");
    const publishedShared = sharedReports.map((item) => ({
      ...item,
      id: item.assignment_id || item.id,
      report: item.report,
      status: "Published",
      club: clubFromReport(
        item.player_club, item.playerClub, item.team, item.Team,
        item.report?.player_club, item.report?.playerClub, item.report?.club,
        item.report?.Club, item.report?.team, item.report?.Team,
      ) || "Club not added",
      position: item.position || item.report?.playedPosition || "",
    }));
    return [...new Map([...publishedLocal, ...publishedShared].map((item) => [String(item.id), item])).values()];
  }, [items, sharedReports]);
  useEffect(() => {
    // In shared reports, top-level `club` is the scout's organisation, not the player's club.
    // Use the normalized report candidates and local assignments, then enrich from the players table.
    const records = [...profileCandidates, ...items];
    const names = records.map((item) => String(item.player || item.Name || item.name || "").trim()).filter(Boolean);
    let cancelled = false;
    const directory = {};
    records.forEach((record) => {
      const key = normalizePlayerName(record.player || record.Name || record.name);
      if (!key) return;
      const club = clubFromReport(
        record.player_club, record.playerClub, record.club, record.Club,
        record.team, record.Team, record.Squad, record.squad,
        record.report?.player_club, record.report?.playerClub,
        record.report?.club, record.report?.Club, record.report?.team,
        record.report?.Team,
      );
      if (club) directory[key] = { ...(directory[key] || {}), club };
    });
    if (!names.length) {
      setProfilePlayerDirectory(directory);
      return () => { cancelled = true; };
    }
    const loadPlayerDetails = async () => {
      const uniqueNames = [...new Map(names.map((name) => [normalizePlayerName(name), name])).values()];
      for (let offset = 0; offset < uniqueNames.length; offset += 20) {
        const batch = uniqueNames.slice(offset, offset + 20);
        const responses = await Promise.all(batch.map((name) => supabase.from("players").select("*").ilike("Name", name).limit(5)));
        responses.forEach(({ data, error }) => {
          if (error) throw error;
          (data || []).forEach((player) => {
            const playerName = normalizePlayerName(player.Name || player.name);
            const ageKey = Object.keys(player).find((key) => key.toLowerCase() === "age");
            const dobKey = Object.keys(player).find((key) => playerDobKeys.some((name) => name.toLowerCase() === key.toLowerCase()));
            const club = clubFromReport(player.club, player.Club, player.team, player.Team, player.Squad, player.squad);
            if (playerName) directory[playerName] = {
              ...(directory[playerName] || {}),
              dob: dobKey ? player[dobKey] : directory[playerName]?.dob || null,
              age: ageKey ? player[ageKey] : directory[playerName]?.age || null,
              club: directory[playerName]?.club || club,
            };
          });
        });
      }
      if (!cancelled) setProfilePlayerDirectory(directory);
    };
    setProfilePlayerDirectory(directory);
    loadPlayerDetails().catch(() => { if (!cancelled) setProfilePlayerDirectory(directory); });
    return () => { cancelled = true; };
  }, [profileCandidates, items, sharedReports]);
  const runProfileCheck = () => {
    const terms = profileTerms(profileQuery);
    if (!terms.length && !profileFiltersActive) { setProfileResults([]); return; }
    const grouped = new Map();
    const matchesFilters = (item) => {
      const report = item.report || {};
      const foot = String(report.foot || item.foot || item.preferred_foot || item["Preferred Foot"] || "").toLowerCase();
      const directoryEntry = profilePlayerDirectory[normalizePlayerName(item.player)];
      const itemDobKey = Object.keys({ ...item, ...report }).find((key) => playerDobKeys.some((name) => name.toLowerCase() === key.toLowerCase()));
      const age = ageFromDob(directoryEntry?.dob) ?? ageFromValue(directoryEntry?.age) ??
        ageFromDob(itemDobKey ? report[itemDobKey] ?? item[itemDobKey] : null) ??
        ageFromValue(report.age ?? item.age ?? item.Age ?? item.player_age);
      const performance = String(report.performance || item.performance || "").match(/[1-5]/)?.[0] || "";
      const potential = String(report.potential || item.potential || "");
      const position = String(report.playedPosition || item.position || item.primary_position || "").toLowerCase();
      return (!profileFilters.foot || foot === profileFilters.foot.toLowerCase()) &&
        matchesAgeFilter(age, profileFilters.age) &&
        (!profileFilters.position || position === profileFilters.position.toLowerCase()) &&
        (!profileFilters.performance.length || profileFilters.performance.includes(performance)) &&
        (!profileFilters.potential.length || profileFilters.potential.includes(potential.trim().toUpperCase()));
    };
    profileCandidates.filter(matchesFilters).forEach((item) => {
      const text = reportSearchText(item);
      const matches = terms.filter((term) => text.includes(term) || text.split(/\s+/).some((word) => word.startsWith(term)));
      if (!matches.length && terms.length) return;
      const key = String(item.player || item.id);
      const current = grouped.get(key) || { ...item, reports: 0, score: 0, matches: [], evidence: "" };
      current.reports += 1;
      current.score = terms.length ? Math.min(100, Math.round((new Set([...current.matches, ...matches]).size / terms.length) * 100)) : 100;
      current.matches = [...new Set([...current.matches, ...matches])];
      current.evidence = current.evidence || profileEvidence(item, terms);
      grouped.set(key, current);
    });
    setProfileResults([...grouped.values()].sort((a, b) => b.score - a.score || b.reports - a.reports));
  };
  useEffect(() => {
    if (profileQuery.trim() || profileFiltersActive) runProfileCheck();
    // Keep results in sync when a filter, player data or the local date changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileFilters, profilePlayerDirectory, ageDate]);
  const exportProfileNames = () => {
    if (!profileResults.length) return;
    const csv = ["Player,Club,Position,Match"].concat(profileResults.map((item) => [item.player, item.club || "", item.position || "", `${item.score}%`].map((value) => `"${String(value).replace(/"/g, '""')}"`).join(","))).join("\n");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    link.download = "profile-checker-players.csv";
    link.click();
  };
  const openReport = (x) => {
    setActive(x);
    setReport(x.report || empty);
    setEditing(x.status !== "Published" && assignmentBelongsToScout(x, user, accountProfile));
    setPublishError("");
  };
  const searchPlayer = () => {
    const match =
      items.find(
        (x) => x.player?.toLowerCase() === query.trim().toLowerCase(),
      ) ||
      items.find((x) =>
        x.player?.toLowerCase().includes(query.trim().toLowerCase()),
      );
    if (match) {
      localStorage.setItem("scoutingProfileOrigin", "assignments");
      setActive(null);
      setProfile(match);
      setQuery("");
    }
  };
  const saveReport = () => {
    const required = [
      ["Preferred foot", report.foot],
      ["Position played", report.playedPosition || active.position],
      ["Conclusion", report.conclusion],
      ["Reasons why", report.reasons],
      ["Performance grade", report.performance],
      ["Potential grade", report.potential],
    ];
    if (report.type === "Long Report") {
      required.push(["Strengths", report.strengths], ["Weaknesses", report.weaknesses]);
    }
    const missing = required.filter(([, value]) => !String(value || "").trim()).map(([label]) => label);
    if (missing.length) {
      setPublishError(`Complete these sections before publishing: ${missing.join(", ")}. Match content can be left blank.`);
      return;
    }
    setPublishError("");
    const completedAt = new Date().toISOString().slice(0, 10);
    const preservedFixtures = fixtureRecords(active);
    const preservedFixtureDates = active.fixtureDates?.length ? active.fixtureDates : active.report?.__fixtureDates || [];
    const reportWithFixtures = {
      ...report,
      __fixtures: preservedFixtures,
      __fixtureDates: preservedFixtureDates,
    };
    const n = items.map((x) =>
      x.id === active.id ? { ...x, report: reportWithFixtures, status: "Published", completedAt } : x,
    );
    saveItems(n);
    const publishedImmediately = {
      id: Number(active.id),
      assignment_id: Number(active.id),
      player: active.player,
      player_club: active.club || "",
      report: reportWithFixtures,
      status: "Published",
      completed_at: completedAt,
      scout: active.scout || "",
      games: preservedFixtures,
      game: isFixtureName(active.game) ? active.game : preservedFixtures.length > 1 ? "Multiple" : preservedFixtures[0]?.name || "",
      fixtureDates: preservedFixtureDates,
      viewing: active.viewing || "",
      date: active.date || completedAt,
    };
    setSharedReports((current) => [
      ...current.filter((item) => String(item.assignment_id || item.id) !== String(active.id)),
      publishedImmediately,
    ]);
    setTab("Published");
    if (user && accountProfile?.club) {
      supabase.from("club_reports").upsert({
        id: Number(active.id), assignment_id: Number(active.id), player_id: active.playerId || null,
        player: active.player, club: accountProfile.club, player_club: active.club || "", author_id: user.id, report: reportWithFixtures, status: "Published",
        updated_at: new Date().toISOString(), completed_at: active.completedAt || active.date || new Date().toISOString().slice(0, 10), scout: active.scout || "", fixture_summary: preservedFixtures.length > 1 ? "Multiple" : (preservedFixtures[0] ? `${preservedFixtures[0].date || ""} · ${preservedFixtures[0].name || preservedFixtures[0]}` : isFixtureName(active.game) ? active.game : ""),
      }, { onConflict: "id" }).then(async ({ error }) => {
        if (error) {
          console.error("Could not publish shared report", error);
          return;
        }
        const { error: assignmentError } = await supabase
          .from("club_assignments")
          .update({ status: "Published", assignment: { ...active, report: reportWithFixtures, status: "Published", games: preservedFixtures, fixtureDates: preservedFixtureDates } })
          .eq("club", accountProfile.club)
          .eq("id", Number(active.id));
        if (assignmentError) console.error("Could not mark shared assignment published", assignmentError);
      });
    }
    setActive(null);
    sessionStorage.removeItem("scoutingActiveReport");
    setProfile(n.find((x) => x.id === active.id) || profile);
  };
  const field = (l, k, r = 5) => (
    <label className="sr-field">
      <span>{l}</span>
      <textarea
        rows={r}
        value={report[k]}
        readOnly={!editing}
        onChange={(e) => setReport({ ...report, [k]: e.target.value })}
      />
    </label>
  );
  const dataValue = (...keys) => {
    for (const source of [playerData, profile, active]) {
      if (!source) continue;
      const key = keys.find((k) => source[k] !== undefined && source[k] !== null && source[k] !== "");
      if (key) return source[key];
    }
    return "—";
  };
  const savePlayerSummary = () => {
    const playerSummaries = { ...(appState?.playerSummaries || {}), [summaryKey]: summaryDraft.trim() };
    updateAppState({
      ...appState,
      assignments: appState?.assignments || [],
      shortlists: appState?.shortlists || [],
      tags: appState?.tags || [],
      playerSummaries,
    });
    setSummaryDraft(summaryDraft.trim());
    setSummaryEditing(false);
  };
  const clubName = dataValue("club", "Club", "team", "Team");
  const clubBadge = dataValue("badgeUrl", "club_badge_url", "clubBadgeUrl", "badge_url");
  const profileHasDirectPhoto = Boolean(["Photo", "photo", "_photoUrl", "photoUrl", "playerPhoto", "Image", "image", "Photo URL"].some((key) => String((playerData || profile)?.[key] || "").trim()));
  const reportFoot = [
    ...items.filter((x) => x.player === profile?.player),
    ...sharedReports.filter((x) => x.player === profile?.player),
  ]
    .reverse()
    .find((x) => x.report?.foot)?.report?.foot;
  const grades = (
    <div className="sr-grade-row">
      <label className="sr-field">
        <span>Performance Grade (1–5)</span>
        <select
          value={report.performance}
          disabled={!editing}
          onChange={(e) =>
            setReport({ ...report, performance: e.target.value })
          }
        >
          <option value="">Select</option>
          {[5, 4, 3, 2, 1].map((x) => (
            <option key={x}>{x}</option>
          ))}
        </select>
      </label>
      <label className="sr-field">
        <span>Potential Grade (A–F)</span>
        <select
          value={report.potential}
          disabled={!editing}
          onChange={(e) => setReport({ ...report, potential: e.target.value })}
        >
          <option value="">Select</option>
          {["A", "B", "C", "D", "E", "F"].map((x) => (
            <option key={x}>{x}</option>
          ))}
        </select>
      </label>
    </div>
  );
  const reportPage = active && (
    <div className="sr-modal">
      <section
        className={`sr-form sr-report ${!editing ? "readonly" : ""}`}
      >
        <button className="sr-report-back" onClick={() => { setActive(null); setProfile(null); sessionStorage.removeItem("scoutingActiveReport"); }}>‹ Back to Assignments</button>
        <div className="sr-form-head sr-report-banner">
          <div className="sr-report-banner-main">
            <img
              className="sr-report-banner-photo"
              src={playerPhotoFor({ ...active, ...(playerData || {}) })}
              alt=""
              onError={(e) => retryPhoto(e, playerData?.Name || playerData?.name || active.player)}
            />
            <div>
              <div className="sr-kicker">PLAYER REPORT</div>
              <button
                className="sr-report-player-link"
                onClick={() => {
                  setActive(null);
                  setProfile(active);
                }}
              >
                <span>{active.player}</span>
              </button>
              <p>{dataValue("club", "Club", "team", "Team")}</p>
              <p className="sr-report-player-meta">
                Primary:{" "}
                {dataValue(
                  "Playing Position",
                  "Primary Position",
                  "Position",
                  "playing_position",
                  "primary_position",
                  "position",
                )}{" "}
                · Secondary:{" "}
                {dataValue(
                  "Secondary Position",
                  "secondary_position",
                  "Secondary position",
                )}{" "}
                · DOB: {dataValue("DOB", "Date of Birth", "date_of_birth")}
              </p>
            </div>
          </div>
          {editing && (
            <button className="sr-cyan sr-banner-publish" onClick={saveReport}>
              {active.status === "Published" ? "Republish Report" : "Publish Report"}
            </button>
          )}
          {!editing && active.status !== "Published" && <p className="sr-readonly-note">Read-only: this assignment belongs to another scout.</p>}
          {publishError && <p className="sr-publish-error" role="alert">{publishError}</p>}
        </div>
        <div className="sr-fixture-box">
          <strong>Assigned Fixture</strong>
          <div className="sr-fixture-list">
            {(fixtureRecords(active).length ? fixtureRecords(active) : [{ name: active.game || "Fixture not added", date: active.fixtureDates?.[0] || active.date || "Date not added" }]).map((fixture, i) => (
              <div className="sr-fixture-card" key={`${typeof fixture === "string" ? fixture : fixture.name}-${i}`}>
                <strong>{typeof fixture === "string" ? fixture : fixture.name || "Fixture not added"}</strong>
                <small>{typeof fixture === "string" ? active.fixtureDates?.[i] || "Date not added" : fixture.date || "Date not added"}</small>
              </div>
            ))}
          </div>
          <small>{active.viewing || "Viewing not added"}</small>
          {editing && <button
            className="sr-edit-games"
            onClick={() => {
              localStorage.setItem("editingAssignment", JSON.stringify(active));
              setActive(null);
              nav("/create-assignment");
            }}
          >Edit Games</button>}
        </div>
        <div className="sr-form-grid">
          <label className="sr-field">
            <span>Report type</span>
            <select
              value={report.type}
              disabled={!editing}
              onChange={(e) => setReport({ ...report, type: e.target.value })}
            >
              <option>Long Report</option>
              <option>Short Report</option>
            </select>
          </label>
          <label className="sr-field">
            <span>Footage</span>
            <select
              value={report.footage || "Full game"}
              disabled={!editing}
              onChange={(e) =>
                setReport({ ...report, footage: e.target.value })
              }
            >
              <option>Full game</option>
              <option>Edited footage</option>
            </select>
          </label>
          <label className="sr-field">
            <span>Viewing</span>
            <select
              value={report.viewing || active.viewing || "Video"}
              disabled={!editing}
              onChange={(e) =>
                setReport({ ...report, viewing: e.target.value })
              }
            >
              <option>Live</option>
              <option>Video</option>
            </select>
          </label>
          <label className="sr-field">
            <span>Preferred foot</span>
            <select
              value={report.foot}
              disabled={!editing}
              onChange={(e) => setReport({ ...report, foot: e.target.value })}
            >
              <option value="">Select</option>
              <option>Right</option>
              <option>Left</option>
              <option>Both</option>
            </select>
          </label>
          <label className="sr-field">
            <span>Position Played</span>
            <select
              value={report.playedPosition || active.position || ""}
              disabled={!editing}
              onChange={(e) =>
                setReport({ ...report, playedPosition: e.target.value })
              }
            >
              <option value="">Select position</option>
              {[
                "GK",
                "LB",
                "LCB",
                "CB",
                "RCB",
                "RB",
                "DM",
                "LM",
                "LCM",
                "CM",
                "RCM",
                "RM",
                "LW",
                "AM",
                "RW",
                "CF",
                "ST",
              ].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
        </div>
        {report.type === "Long Report" && (
          <div className="sr-report-columns">
            <div>
              {field("In Possession", "inPossession")}
              {field("Out of Possession", "outPossession")}
              {field("Physical", "physical")}
              {field("On-Pitch Behaviour", "behaviour")}
            </div>
            <div>
              {field("Strengths", "strengths")}
              {field("Weaknesses", "weaknesses")}
              {field("Conclusion", "conclusion", 4)}
            </div>
          </div>
        )}
        {report.type === "Short Report" && field("Conclusion", "conclusion", 4)}
        {grades}
        {field("Reasons Why", "reasons", 6)}
          {active?.status === "Published" && !editing && (!active.author_id || active.author_id === user?.id) && (
          <div className="sr-report-bottom-actions">
            <button className="sr-outline" onClick={() => setEditing(true)}>
              Edit Report
            </button>
          </div>
        )}
      </section>
    </div>
  );
  if (profile)
    return (
      <main className="sr-page sr-profile-page">
        <button
          className="sr-back"
          onClick={() => {
            if (
              localStorage.getItem("scoutingProfileOrigin") === "shortlists"
            ) {
              localStorage.removeItem("scoutingProfilePlayer");
              localStorage.removeItem("scoutingProfileOrigin");
              localStorage.removeItem("scoutingProfileReturnPath");
              nav("/shortlists");
            } else if (localStorage.getItem("scoutingProfileOrigin") === "squad-plan") {
              const returnPath = localStorage.getItem("scoutingProfileReturnPath") || "/squad-plan?new=1";
              localStorage.removeItem("scoutingProfilePlayer");
              localStorage.removeItem("scoutingProfileOrigin");
              localStorage.removeItem("scoutingProfileReturnPath");
              nav(returnPath);
            } else {
              localStorage.removeItem("scoutingProfilePlayer");
              localStorage.removeItem("scoutingProfileOrigin");
              localStorage.removeItem("scoutingProfileReturnPath");
              setProfile(null);
            }
          }}
        >
          ‹{" "}
          {localStorage.getItem("scoutingProfileOrigin") === "shortlists"
            ? "Back to previous page"
            : localStorage.getItem("scoutingProfileOrigin") === "squad-plan"
              ? "Back to Squad Analysis"
              : "Back to assignments"}
        </button>
        <section className="sr-profile-head">
          <div>
            <div className="sr-kicker">PLAYER PROFILE</div>
            <h1>{profile.player}</h1>
            <p>
              <ClubName club={clubName} /> ·{" "}
              {dataValue("Playing Position", "Primary Position", "Position", "playing_position", "primary_position", "position")}
            </p>
          </div>
          <button className="sr-cyan" onClick={() => nav("/create-assignment")}>
            Create Assignment
          </button>
        </section>
        <section className="sr-profile-summary">
          <div className="sr-avatar">
            <img
              src={playerPhotoFor({ ...profile, ...(playerData || {}) })}
              alt=""
              onLoad={(e) => {
                e.currentTarget.nextElementSibling.style.display = "none";
              }}
              onError={(e) => retryPhoto(e, playerData?.Name || playerData?.name || profile.player)}
            />
            <span>{profile.player.slice(0, 2).toUpperCase()}</span>
            {!profileHasDirectPhoto && <label className="sr-add-photo" title="Add player photo"><input type="file" accept="image/*" onChange={uploadProfilePhoto} disabled={photoUploading} />{photoUploading ? "…" : "+"}</label>}
          </div>
          <div>
            <h2>{profile.player}</h2>
            <p className="sr-player-club">
              <ClubName club={clubName} externalUrl={clubBadge !== "—" ? clubBadge : undefined} size={24} />
            </p>
            <span>
              {[...new Map([
                ...items.filter((x) => x.player === profile.player && x.status === "Published"),
                ...sharedReports.filter((x) => x.player === profile.player).map((x) => ({ ...x, id: x.assignment_id || x.id, report: x.report, status: "Published", club: clubFromReport(x.player_club, x.playerClub, x.team, x.Team, x.report?.player_club, x.report?.playerClub, x.report?.club, x.report?.Club, x.report?.team, x.report?.Team), date: x.completed_at, game: x.fixture_summary, scout: x.scout })),
              ].map((x) => [x.id, x])).values()].length} {" "}
              published reports
            </span>
          </div>
          {photoStatus && <small className="sr-photo-status">{photoStatus}</small>}
          <button
            className="sr-outline"
            onClick={() => {
              const lists = appState?.shortlists || [];
              setSelectedShortlist(lists[0]?.id || "");
              setShortlistPicker(true);
            }}
          >
            Add to Shortlist
          </button>
        </section>
        <section
          className="sr-player-details sr-player-summary-editor"
          style={{ width: "100%", boxSizing: "border-box", border: "1px solid #2f84c5", borderRadius: 12, padding: 20, marginBottom: 20 }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 10 }}>
            <h2 style={{ margin: 0 }}>Player Summary</h2>
            {!summaryEditing && <button className="sr-outline" type="button" onClick={() => setSummaryEditing(true)}>Edit Summary</button>}
          </div>
          {summaryEditing ? (
            <>
              <textarea
                aria-label="Player Summary"
                value={summaryDraft}
                onChange={(e) => setSummaryDraft(e.target.value)}
                rows={5}
                style={{ display: "block", width: "100%", boxSizing: "border-box", resize: "vertical", padding: 12, font: "inherit", lineHeight: 1.55, border: "1px solid #cbd5e1", borderRadius: 8 }}
              />
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 12 }}>
                {summaryReports.length > 0 && <button className="sr-outline" type="button" onClick={() => setSummaryDraft(generatedPlayerSummary)}>Use Report Summary</button>}
                <button className="sr-outline" type="button" onClick={() => { setSummaryDraft(shownPlayerSummary); setSummaryEditing(false); }}>Cancel</button>
                <button className="sr-cyan" type="button" onClick={savePlayerSummary}>Save Summary</button>
              </div>
            </>
          ) : (
            <p style={{ margin: 0, whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{shownPlayerSummary || "No summary yet. Select Edit Summary to add one."}</p>
          )}
        </section>
        <section className="sr-profile-layout">
          <section className="sr-player-details">
            <h2>Player Information</h2>
            <div className="sr-detail-list">
              {[
                ["Name", ["Name", "name"]],
                ["Club", ["club", "Club", "team", "Team"]],
                ["DOB", ["DOB", "Date of Birth", "date_of_birth", "dateOfBirth", "dob", "birth_date", "birthDate"]],
                ["Age", ["Age", "age"]],
                ["Nationality", ["Nationality", "nationality"]],
                [
                  "Dominant Foot",
                  ["Dominant Foot", "Preferred Foot", "preferred_foot"],
                ],
                ["Contract Expiry", ["Contract Expiry", "contract_expiry"]],
                ["Height", ["Height", "height"]],
                [
                  "Position",
                  [
                    "Playing Position",
                    "Primary Position",
                    "Position",
                    "playing_position",
                    "primary_position",
                    "position",
                  ],
                ],
                ["Agent", ["Agent", "agent"]],
              ].map(([label, keys]) => (
                <div className="sr-detail-row" key={label}>
                  <strong>{label}:</strong>
                  <span>
                    {label === "Name" ? (
                      profile.player || dataValue(...keys)
                    ) : label === "Club" ? (
                      <ClubName club={dataValue(...keys)} size={20} />
                    ) : label === "Age" ? (
                      ageFromDob(dataValue(...playerDobKeys)) ?? dataValue(...keys)
                    ) : label === "Dominant Foot" ? (
                      reportFoot || dataValue(...keys)
                    ) : (
                      dataValue(...keys)
                    )}
                  </span>
                </div>
              ))}
            </div>
            {playerDataLoading && <p className="sr-player-data-notice" role="status">Loading player details…</p>}
            {!playerDataLoading && playerDataError && <p className="sr-player-data-notice error" role="alert">Could not load player details: {playerDataError}</p>}
            {!playerDataLoading && !playerDataError && !playerData && <p className="sr-player-data-notice">The report is available, but no matching player record was found in the master player database. Add the player there to show their date of birth, age and other details.</p>}
          </section>
          <section className="sr-profile-reports">
            <h2>Reports</h2>
            <div className="sr-report-table">
              <div className="sr-table-head">
                <span>Date</span>
                <span>Fixture</span>
                <span>Scout</span>
                <span>Potential</span>
                <span>Performance</span>
                <span>Status</span>
              </div>
              {[...new Map([
                ...items.filter((x) => x.player === profile.player),
                ...sharedReports.filter((x) => x.player === profile.player).map((x) => ({ ...x, id: x.assignment_id || x.id, report: x.report, status: "Published", club: clubFromReport(x.player_club, x.playerClub, x.team, x.Team, x.report?.player_club, x.report?.playerClub, x.report?.club, x.report?.Club, x.report?.team, x.report?.Team), date: x.completed_at, game: x.fixture_summary, scout: x.scout })),
              ].map((x) => [x.id, x])).values()]
                .map((x) => (
                  <button
                    className="sr-table-row"
                    key={x.id}
                    onClick={() => openReport(x)}
                  >
                    <span>{x.completed_at || x.completedAt || x.date || "—"}</span>
                    <span>{fixtureLabel(x) || "—"}</span>
                    <span>{x.scout || "—"}</span>
                    <span>{x.report?.potential || "—"}</span>
                    <span>{x.report?.performance || "—"}</span>
                    <span>{x.status}</span>
                  </button>
                ))}
            </div>
          </section>
        </section>
        {shortlistPicker && (
          <div className="sr-modal" onClick={() => setShortlistPicker(false)}>
            <section
              className="sr-form sr-shortlist-picker"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="sr-form-head">
                <h2>Add to Shortlist</h2>
                <button
                  className="sr-close"
                  onClick={() => setShortlistPicker(false)}
                >
                  ×
                </button>
              </div>
              <label className="sr-field">
                <span>Shortlist</span>
                <select
                  value={selectedShortlist}
                  onChange={(e) => setSelectedShortlist(e.target.value)}
                >
                  <option value="">Select shortlist</option>
                  {(appState?.shortlists || []).map((x) => (
                    <option key={x.id} value={x.id}>
                      {x.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="sr-field">
                <span>Position</span>
                <select
                  value={selectedPosition}
                  onChange={(e) => setSelectedPosition(e.target.value)}
                >
                  {shortlistPositions.map((x) => (
                    <option key={x} value={`${x}-0`}>
                      {x}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="sr-cyan"
                disabled={!selectedShortlist}
                onClick={addProfileToShortlist}
              >
                Add Player
              </button>
            </section>
          </div>
        )}
        {reportPage}
      </main>
    );
  return (
    <main className="sr-page">
      <section className="sr-dashboard-head">
        <div>
          <div className="sr-kicker">SCOUTING PLATFORM</div>
          <h1>Reports and Assignments</h1>
          <p>Manage player observations and complete your scouting reports.</p>
        </div>
        <div className="sr-head-actions">
          <div className="sr-player-search-wrap">
            <input
              className="sr-player-search"
              placeholder="Search player..."
              value={playerSearch}
              onChange={(e) => setPlayerSearch(e.target.value)}
            />
            {!!playerMatches.length && (
              <div className="sr-player-search-results">
                {playerMatches.map((p) => {
                  const name = p.Name || p.name;
                  return (
                    <button
                      key={p.id || name}
                      onClick={() => {
                        setProfile({
                          ...p,
                          player: name,
                          club: p.club || p.Club || p.team || p.Team,
                        });
                        setPlayerSearch("");
                        setPlayerMatches([]);
                        localStorage.setItem("scoutingProfileOrigin", "assignments");
                      }}
                    >
                      {name}
                      <small>
                        <ClubName club={p.club || p.Club || p.team || p.Team} size={16} />
                      </small>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
          <div className="sr-page-actions-menu" ref={actionsMenuRef}>
            <button type="button" className={`sr-outline sr-page-actions-trigger ${actionsMenuOpen ? "selected" : ""}`} aria-haspopup="menu" aria-expanded={actionsMenuOpen} onClick={() => setActionsMenuOpen((open) => !open)}>
              More <span aria-hidden="true">▾</span>
            </button>
            {actionsMenuOpen && <div className="sr-page-actions-dropdown" role="menu">
              <button type="button" role="menuitem" onClick={() => { setDashboardView(dashboardView === "checker" ? "reports" : "checker"); setActionsMenuOpen(false); }}>{dashboardView === "checker" ? "Reports and Assignments" : "Profile Checker"}</button>
              <button type="button" role="menuitem" onClick={() => { setActionsMenuOpen(false); nav("/shortlists"); }}>View Shortlists</button>
              <button type="button" role="menuitem" onClick={() => { setActionsMenuOpen(false); setReportImportOpen(true); }}>Import Old Reports</button>
            </div>}
          </div>
          <button className="sr-cyan" onClick={() => nav("/create-assignment")}>
            Create New Assignment
          </button>
        </div>
      </section>
      {importNotice && <div className="sr-import-notice" role="status"><span>{importNotice}</span><button type="button" onClick={() => { setDashboardView("reports"); setTab("My Assignments"); setImportNotice(""); }}>Review drafts</button><button type="button" className="sr-import-notice-close" onClick={() => setImportNotice("")} aria-label="Dismiss">×</button></div>}
      {dashboardView === "reports" && <><input
        className="sr-search"
        placeholder="Search scout or player..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") searchPlayer();
        }}
      />
      <section className="sr-tabs">
        {["My Assignments", "All Assigned", "Published"].map((x) => (
          <button className={tab === x ? "selected" : ""} onClick={() => setTab(x)} key={x}>{x}</button>
        ))}
      </section>
      <div className="sr-section-line"><h2>{tab}</h2><span>{shown.length} assignments</span></div>
      <section className="sr-grid">
        {shown.map((x) => (
          <article className="sr-card" key={x.id} onClick={() => { setProfile(null); openReport(x); }}>
            <div className="sr-card-top"><span className="sr-card-status">{x.status}</span><button className="sr-trash" onClick={(e) => { e.stopPropagation(); deleteAssignment(x); }}>Delete</button></div>
            <button type="button" className="sr-assignment-player-link" onClick={(e) => { e.stopPropagation(); setProfile(null); openReport(x); }}>{x.player}</button>
            <p><ClubName club={clubFromReport(x.player_club, x.playerClub, x.club, x.Club, x.team, x.Team) || profilePlayerDirectory[normalizePlayerName(x.player)]?.club} /> · {x.position || "Position not added"}</p>
            <div className="sr-fixture">{cardFixtureLabel(x)}</div>
            <div className="sr-card-meta"><span>{x.date || "Date not added"}</span><span>{x.viewing}</span><span>Scout: {x.scout || "Unassigned"}</span></div>
          </article>
        ))}
      </section>
      {!shown.length && <div className="sr-empty">No assignments found.</div>}</>}
      {dashboardView === "checker" && <section className="sr-profile-checker">
        <button type="button" className="sr-checker-back" onClick={() => { setDashboardView("reports"); setProfileResults([]); setProfileQuery(""); }}>← Back to Assignments</button>
        <div className="sr-profile-checker-head">
          <div>
            <div className="sr-kicker">PROFILE CHECKER</div>
            <h2>Find players from your reports</h2>
            <p>Describe the profile you need and we’ll rank published reports against those criteria.</p>
          </div>
          <span className="sr-profile-checker-count">{profileCandidates.length} reports scanned</span>
        </div>
        <div className="sr-profile-checker-search">
          <input
            className="sr-search"
            placeholder="e.g. left-footed centre-back strong in duels and good in possession"
            value={profileQuery}
            onChange={(e) => setProfileQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") runProfileCheck(); }}
          />
          <button className="sr-cyan" onClick={runProfileCheck}>Check Profile</button>
        </div>
        <div className="sr-profile-filters">
          <select value={profileFilters.foot} onChange={(e) => setProfileFilters({ ...profileFilters, foot: e.target.value })}><option value="">Any foot</option><option>Right</option><option>Left</option><option>Both</option></select>
          <select value={profileFilters.age} onChange={(e) => setProfileFilters({ ...profileFilters, age: e.target.value })}><option value="">Any age</option><option value="under21">Under 21</option><option value="21to24">21–24</option><option value="25to29">25–29</option><option value="30plus">30+</option></select>
          <select value={profileFilters.position} onChange={(e) => setProfileFilters({ ...profileFilters, position: e.target.value })}><option value="">Any position</option>{["GK","LB","LCB","CB","RCB","RB","DM","LM","LCM","CM","RCM","RM","LW","AM","RW","CF","ST"].map((x) => <option key={x}>{x}</option>)}</select>
          <details className="sr-profile-grade-filter">
            <summary>{profileFilters.performance.length ? `Performance grade${profileFilters.performance.length > 1 ? "s" : ""}: ${profileFilters.performance.join(", ")}` : "Any performance grade"}</summary>
            <div className="sr-profile-grade-menu" role="group" aria-label="Choose performance grades">
              {["1", "2", "3", "4", "5"].map((grade) => <label key={grade}><input type="checkbox" checked={profileFilters.performance.includes(grade)} onChange={(event) => setProfileFilters({ ...profileFilters, performance: event.target.checked ? [...profileFilters.performance, grade] : profileFilters.performance.filter((value) => value !== grade) })} />{grade}/5</label>)}
              {profileFilters.performance.length > 0 && <button type="button" onClick={() => setProfileFilters({ ...profileFilters, performance: [] })}>Clear grades</button>}
            </div>
          </details>
          <details className="sr-profile-grade-filter">
            <summary>{profileFilters.potential.length ? `Potential grade${profileFilters.potential.length > 1 ? "s" : ""}: ${profileFilters.potential.join(", ")}` : "Any potential grade"}</summary>
            <div className="sr-profile-grade-menu" role="group" aria-label="Choose potential grades">
              {["A", "B", "C", "D", "E", "F"].map((grade) => <label key={grade}><input type="checkbox" checked={profileFilters.potential.includes(grade)} onChange={(event) => setProfileFilters({ ...profileFilters, potential: event.target.checked ? [...profileFilters.potential, grade] : profileFilters.potential.filter((value) => value !== grade) })} />{grade} grade</label>)}
              {profileFilters.potential.length > 0 && <button type="button" onClick={() => setProfileFilters({ ...profileFilters, potential: [] })}>Clear grades</button>}
            </div>
          </details>
        </div>
        {!!profileResults.length && (
          <><div className="sr-profile-results-actions"><strong>{profileResults.length} matching players</strong><button className="sr-outline" onClick={exportProfileNames}>Export Names</button></div><div className="sr-profile-results">
            {profileResults.slice(0, 12).map((match) => (
              <button className="sr-profile-result" key={match.player || match.id} onClick={() => { setProfile(match); setProfileQuery(""); }}>
                <span className="sr-profile-result-main"><strong>{match.player || "Unnamed player"}</strong><small><ClubName club={match.club} size={16} />{match.position ? ` · ${match.position}` : ""}</small></span>
                <span className="sr-profile-result-score">{match.score}% match<small>{match.reports} report{match.reports === 1 ? "" : "s"}</small></span>
                <span className="sr-profile-result-evidence">{match.evidence}</span>
              </button>
            ))}
          </div></>
        )}
        {profileQuery.trim() && !profileResults.length && <div className="sr-profile-no-results">No published reports match those criteria yet.</div>}
      </section>}
      {reportImportOpen && <ReportImportModal storageKey={reportImportStorageKey} onClose={() => setReportImportOpen(false)} onImport={importOldReports} />}
      {reportPage}
    </main>
  );
}
