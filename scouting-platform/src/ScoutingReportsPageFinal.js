import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "./supabaseClient";
import { useAuth } from "./AuthContext";
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
  return direct || playerPhoto(record?.player || record?.Name || record?.name);
};
const photoCandidates = (name) => {
  const raw = String(name || "").trim();
  const display = raw.split(/\s+/).length > 2 ? `${raw.split(/\s+/)[0]} ${raw.split(/\s+/).at(-1)}` : raw;
  const rawBases = [raw, display].map((value) => value.replace(/[^\p{L}\p{N}]+/gu, "_").replace(/^_+|_+$/g, ""));
  const bases = [...new Set([...rawBases, ...rawBases.map((value) => value.normalize("NFD").replace(/[̀-ͯ]/g, ""))].filter(Boolean))];
  return [...new Set(bases.flatMap((base) => [base, base.toLowerCase(), base.toUpperCase(), `_${base}`, `_${base.toLowerCase()}`, `__${base}`]).map((base) => `${photoBase}${base}.png`))];
};
const retryPhoto = (e, name) => {
  const image = e.currentTarget;
  const candidates = photoCandidates(name);
  const next = Number(image.dataset.photoFallback || 0) + 1;
  if (candidates[next - 1]) { image.dataset.photoFallback = String(next); image.src = candidates[next - 1]; }
  else image.style.display = "none";
};
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
const fixtureRecords = (item) => Array.isArray(item?.games) && item.games.length ? item.games : (Array.isArray(item?.report?.__fixtures) ? item.report.__fixtures : []);
const fixtureLabel = (item) => {
  const fixtures = fixtureRecords(item);
  if (fixtures.length > 1) return "Multiple";
  if (fixtures.length === 1) return typeof fixtures[0] === "string" ? fixtures[0] : fixtures[0]?.name || "Fixture not added";
  return item?.game || item?.fixture_summary || "Fixture not added";
};
export default function ScoutingReportsPageFinal() {
  const nav = useNavigate();
  const { appState, updateAppState, user, profile: accountProfile } = useAuth();
  const [items, setItems] = useState(() =>
    JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"),
  );
  const [tab, setTab] = useState("My Assignments");
  const [query, setQuery] = useState("");
  const [playerSearch, setPlayerSearch] = useState("");
  const [playerMatches, setPlayerMatches] = useState([]);
  const [profile, setProfile] = useState(() => {
    const n = JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"),
      name = localStorage.getItem("scoutingProfilePlayer");
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
  const [photoUploading, setPhotoUploading] = useState(false);
  const [photoStatus, setPhotoStatus] = useState("");
  const [profileQuery, setProfileQuery] = useState(() => sessionStorage.getItem("profileCheckerQuery") || "");
  const [profileResults, setProfileResults] = useState([]);
  const [profilePlayerDirectory, setProfilePlayerDirectory] = useState({});
  const [dashboardView, setDashboardView] = useState(() => sessionStorage.getItem("scoutingDashboardView") || "reports");
  const [profileFilters, setProfileFilters] = useState(() => JSON.parse(sessionStorage.getItem("profileCheckerFilters") || '{"foot":"","age":"","position":"","performance":"","potential":""}'));
  useEffect(() => { sessionStorage.setItem("scoutingDashboardView", dashboardView); }, [dashboardView]);
  useEffect(() => { sessionStorage.setItem("profileCheckerFilters", JSON.stringify(profileFilters)); }, [profileFilters]);
  useEffect(() => { sessionStorage.setItem("profileCheckerQuery", profileQuery); }, [profileQuery]);
  useEffect(() => {
    const selectedPlayer = profile || active;
    if (!selectedPlayer?.player) {
      setPlayerData(null);
      return;
    }
    supabase
      .from("players")
      .select("*")
      .ilike("Name", selectedPlayer.player)
      .limit(5)
      .then(({ data }) => {
        const exact = (data || []).find(
          (row) =>
            String(row.Name || row.name || "").toLowerCase() ===
            selectedPlayer.player.toLowerCase(),
        );
        setPlayerData(exact || data?.[0] || null);
      })
      .catch(() => setPlayerData(null));
  }, [profile, active]);
  const uploadProfilePhoto = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !profile?.player) return;
    const filename = photoSlug(profile.player, true);
    const path = `player-photos/${filename}.png`;
    try {
      setPhotoUploading(true); setPhotoStatus("");
      const { error: uploadError } = await supabase.storage.from("player-photos").upload(path, file, { upsert: true, contentType: file.type || "image/png" });
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
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (active) sessionStorage.setItem("scoutingActiveReport", JSON.stringify({ active, report }));
  }, [active, report]);
  const publishedAssignmentIds = useMemo(
    () => new Set(sharedReports.map((x) => String(x.assignment_id || x.id))),
    [sharedReports],
  );
  useEffect(() => {
    const localAssignments = appState?.assignments || [];
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
      const published = data || [];
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
    updateAppState({ assignments: n, shortlists: appState?.shortlists || [], tags: appState?.tags || [] });
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
    updateAppState({ assignments: appState?.assignments || [], shortlists: lists.map((x) => (String(x.id) === String(list.id) ? next : x)), tags: appState?.tags || [] });
    setShortlistPicker(false);
  };
  const shown = useMemo(
    () => {
      const shared = sharedReports.map((x) => ({ ...x, id: x.assignment_id || x.id, report: x.report, status: "Published", club: x.player_club || "Club not added", position: x.position || x.report?.playedPosition || "", date: x.completed_at, game: x.fixture_summary, scout: x.scout }));
      const source = tab === "Published" ? shared : items;
      return [...new Map(source.map((x) => [String(x.id), x])).values()]
        .filter((x) => tab !== "My Assignments" || !x.scoutId || x.scoutId === user?.id)
        .filter((x) =>
          `${x.player} ${x.club} ${x.scout}`
            .toLowerCase()
            .includes(query.toLowerCase()),
        )
        .filter((x) =>
          tab === "Published"
            ? x.status === "Published"
            : x.status !== "Published",
        );
    },
    [items, query, tab, sharedReports, user?.id],
  );
  const profileCandidates = useMemo(() => {
    const publishedLocal = items.filter((item) => item.status === "Published");
    const publishedShared = sharedReports.map((item) => ({ ...item, id: item.assignment_id || item.id, report: item.report, status: "Published", club: item.player_club || item.club || "Club not added", position: item.position || item.report?.playedPosition || "" }));
    return [...new Map([...publishedLocal, ...publishedShared].map((item) => [String(item.id), item])).values()];
  }, [items, sharedReports]);
  useEffect(() => {
    const names = profileCandidates.map((item) => String(item.player || "").trim().toLowerCase()).filter(Boolean);
    if (!names.length) return;
    supabase.from("players").select("*").limit(2000).then(({ data }) => {
      const directory = {};
      (data || []).forEach((player) => {
        const playerName = String(player.Name || player.name || "").trim().toLowerCase();
        const ageKey = Object.keys(player).find((key) => key.toLowerCase() === "age");
        if (playerName && names.includes(playerName) && ageKey) directory[playerName] = player[ageKey];
      });
      setProfilePlayerDirectory(directory);
    }).catch(() => setProfilePlayerDirectory({}));
  }, [profileCandidates]);
  const runProfileCheck = () => {
    const terms = profileTerms(profileQuery);
    if (!terms.length && !Object.values(profileFilters).some(Boolean)) { setProfileResults([]); return; }
    const grouped = new Map();
    const matchesFilters = (item) => {
      const report = item.report || {};
      const foot = String(report.foot || item.foot || item.preferred_foot || item["Preferred Foot"] || "").toLowerCase();
      const age = Number(profilePlayerDirectory[String(item.player || "").trim().toLowerCase()] ?? report.age ?? item.age ?? item.Age ?? item.player_age);
      const performance = String(report.performance || item.performance || "");
      const potential = String(report.potential || item.potential || "");
      const position = String(report.playedPosition || item.position || item.primary_position || "").toLowerCase();
      const potentialRank = { A: 1, B: 2, C: 3, D: 4, E: 5, F: 6 };
      return (!profileFilters.foot || foot === profileFilters.foot.toLowerCase()) &&
        (!profileFilters.age || (Number.isFinite(age) && (profileFilters.age === "under21" ? age < 21 : profileFilters.age === "21to24" ? age >= 21 && age <= 24 : profileFilters.age === "25to29" ? age >= 25 && age <= 29 : age >= 30))) &&
        (!profileFilters.position || position === profileFilters.position.toLowerCase()) &&
        (!profileFilters.performance || (Number(performance) >= Number(profileFilters.performance))) &&
        (!profileFilters.potential || (potentialRank[potential.toUpperCase()] >= potentialRank[profileFilters.potential]));
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
    if (profileQuery.trim() || Object.values(profileFilters).some(Boolean)) runProfileCheck();
    // Keep results in sync when a filter or the player-profile ages finish loading.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileFilters, profilePlayerDirectory]);
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
    setEditing(x.status !== "Published");
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
    const completedAt = new Date().toISOString().slice(0, 10);
    const reportWithFixtures = {
      ...report,
      __fixtures: active.games || [],
      __fixtureDates: active.fixtureDates || [],
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
      games: active.games || [],
      game: active.game || active.fixture_summary || "",
      fixtureDates: active.fixtureDates || [],
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
        updated_at: new Date().toISOString(), completed_at: active.completedAt || active.date || new Date().toISOString().slice(0, 10), scout: active.scout || "", fixture_summary: (active.games || []).length > 1 ? "Multiple" : (active.games?.[0] ? `${active.games[0].date || ""} · ${active.games[0].name || active.games[0]}` : active.game || ""),
      }, { onConflict: "id" }).then(async ({ error }) => {
        if (error) {
          console.error("Could not publish shared report", error);
          return;
        }
        const { error: assignmentError } = await supabase
          .from("club_assignments")
          .update({ status: "Published", assignment: { ...active, report: reportWithFixtures, status: "Published", games: active.games || [], fixtureDates: active.fixtureDates || [] } })
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
        onChange={(e) => setReport({ ...report, [k]: e.target.value })}
      />
    </label>
  );
  const dataValue = (...keys) => {
    const source = playerData || profile || active || {};
    const key = keys.find(
      (k) => source[k] !== undefined && source[k] !== null && source[k] !== "",
    );
    return key ? source[key] : "—";
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
        className={`sr-form sr-report ${active?.status === "Published" && !editing ? "readonly" : ""}`}
      >
        <button className="sr-report-back" onClick={() => { setActive(null); setProfile(null); sessionStorage.removeItem("scoutingActiveReport"); }}>‹ Back to Assignments</button>
        <div className="sr-form-head sr-report-banner">
          <div className="sr-report-banner-main">
            <img
              className="sr-report-banner-photo"
              src={playerPhotoFor({ ...active, ...(playerData || {}) })}
              alt=""
              onError={(e) => retryPhoto(e, active.player)}
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
          {(!active || active.status !== "Published" || editing) && (
            <button className="sr-cyan sr-banner-publish" onClick={saveReport}>
              Publish Report
            </button>
          )}
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
          {(!active.author_id || active.author_id === user?.id) && <button
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
              nav("/shortlists");
            } else {
              localStorage.removeItem("scoutingProfilePlayer");
              localStorage.removeItem("scoutingProfileOrigin");
              setProfile(null);
            }
          }}
        >
          ‹{" "}
          {localStorage.getItem("scoutingProfileOrigin") === "shortlists"
            ? "Back to previous page"
            : "Back to assignments"}
        </button>
        <section className="sr-profile-head">
          <div>
            <div className="sr-kicker">PLAYER PROFILE</div>
            <h1>{profile.player}</h1>
            <p>
              {clubName} ·{" "}
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
              onError={(e) => retryPhoto(e, profile.player)}
            />
            <span>{profile.player.slice(0, 2).toUpperCase()}</span>
            {!profileHasDirectPhoto && <label className="sr-add-photo" title="Add player photo"><input type="file" accept="image/*" onChange={uploadProfilePhoto} disabled={photoUploading} />{photoUploading ? "…" : "+"}</label>}
          </div>
          <div>
            <h2>{profile.player}</h2>
            <p className="sr-player-club">
              {clubBadge !== "—" && <img src={clubBadge} alt="" onError={(e) => { e.currentTarget.style.display = "none"; }} />}
              <span>{clubName}</span>
            </p>
            <span>
              {[...new Map([
                ...items.filter((x) => x.player === profile.player && x.status === "Published"),
                ...sharedReports.filter((x) => x.player === profile.player).map((x) => ({ ...x, id: x.assignment_id || x.id, report: x.report, status: "Published", club: x.player_club || x.club, date: x.completed_at, game: x.fixture_summary, scout: x.scout })),
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
        <section className="sr-profile-layout">
          <section className="sr-player-details">
            <h2>Player Information</h2>
            <div className="sr-detail-list">
              {[
                ["Name", ["Name", "name"]],
                ["Club", ["club", "Club", "team", "Team"]],
                ["DOB", ["DOB", "Date of Birth", "date_of_birth"]],
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
                    {label === "Dominant Foot"
                      ? reportFoot || dataValue(...keys)
                      : dataValue(...keys)}
                  </span>
                </div>
              ))}
            </div>
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
                ...sharedReports.filter((x) => x.player === profile.player).map((x) => ({ ...x, id: x.assignment_id || x.id, report: x.report, status: "Published", club: x.player_club || x.club, date: x.completed_at, game: x.fixture_summary, scout: x.scout })),
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
                        localStorage.setItem(
                          "scoutingProfileOrigin",
                          "assignments",
                        );
                      }}
                    >
                      {name}
                      <small>
                        {p.club ||
                          p.Club ||
                          p.team ||
                          p.Team ||
                          "Club not added"}
                      </small>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
          <button className={`sr-outline sr-profile-tab ${dashboardView === "checker" ? "selected" : ""}`} onClick={() => setDashboardView(dashboardView === "checker" ? "reports" : "checker")}>
            Profile Checker
          </button>
          <button className="sr-outline" onClick={() => nav("/shortlists")}>
            View Shortlists
          </button>
          <button className="sr-cyan" onClick={() => nav("/create-assignment")}>
            Create New Assignment
          </button>
        </div>
      </section>
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
          <article className="sr-card" key={x.id} onClick={() => { setProfile(null); setActive(x); }}>
            <div className="sr-card-top"><span className="sr-card-status">{x.status}</span><button className="sr-trash" onClick={(e) => { e.stopPropagation(); deleteAssignment(x); }}>Delete</button></div>
            <button type="button" className="sr-assignment-player-link" onClick={(e) => { e.stopPropagation(); setProfile(null); setActive(x); }}>{x.player}</button>
            <p>{x.club || "Club not added"} · {x.position || "Position not added"}</p><div className="sr-fixture">{fixtureLabel(x)}</div>
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
          <select value={profileFilters.performance} onChange={(e) => setProfileFilters({ ...profileFilters, performance: e.target.value })}><option value="">Any performance grade</option>{[1,2,3,4,5].map((x) => <option key={x} value={String(x)}>{x}/5 or higher</option>)}</select>
          <select value={profileFilters.potential} onChange={(e) => setProfileFilters({ ...profileFilters, potential: e.target.value })}><option value="">Any potential grade</option>{["A","B","C","D","E","F"].map((x) => <option key={x} value={x}>{x} grade and below</option>)}</select>
        </div>
        {!!profileResults.length && (
          <><div className="sr-profile-results-actions"><strong>{profileResults.length} matching players</strong><button className="sr-outline" onClick={exportProfileNames}>Export Names</button></div><div className="sr-profile-results">
            {profileResults.slice(0, 12).map((match) => (
              <button className="sr-profile-result" key={match.player || match.id} onClick={() => { setProfile(match); setProfileQuery(""); }}>
                <span className="sr-profile-result-main"><strong>{match.player || "Unnamed player"}</strong><small>{match.club || "Club not added"}{match.position ? ` · ${match.position}` : ""}</small></span>
                <span className="sr-profile-result-score">{match.score}% match<small>{match.reports} report{match.reports === 1 ? "" : "s"}</small></span>
                <span className="sr-profile-result-evidence">{match.evidence}</span>
              </button>
            ))}
          </div></>
        )}
        {profileQuery.trim() && !profileResults.length && <div className="sr-profile-no-results">No published reports match those criteria yet.</div>}
      </section>}
      {reportPage}
    </main>
  );
}
