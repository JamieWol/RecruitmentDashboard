import React, { useEffect, useMemo, useState } from "react";
import { useAuth } from "./AuthContext";
import { supabase } from "./supabaseClient";

const formations = ["4-2-3-1", "4-3-3", "4-4-2", "3-5-2", "3-4-3", "4-1-4-1"];
const formationRows = {
  "4-2-3-1": [["ST"], ["LW", "AM", "RW"], ["DM", "DM"], ["LB", "LCB", "RCB", "RB"], ["GK"]],
  "4-3-3": [["LW", "CF", "RW"], ["LCM", "CM", "RCM"], ["LB", "LCB", "RCB", "RB"], ["GK"]],
  "4-4-2": [["ST", "ST"], ["LM", "LCM", "RCM", "RM"], ["LB", "LCB", "RCB", "RB"], ["GK"]],
  "3-5-2": [["ST", "ST"], ["LWB", "CM", "CM", "RWB"], ["LCB", "CB", "RCB"], ["GK"]],
  "3-4-3": [["LW", "CF", "RW"], ["LWB", "LCM", "RCM", "RWB"], ["LCB", "CB", "RCB"], ["GK"]],
  "4-1-4-1": [["ST"], ["LM", "CM", "CM", "RM"], ["DM"], ["LB", "LCB", "RCB", "RB"], ["GK"]],
};
const positionAliases = { goalkeeper: "GK", gk: "GK", defender: "CB", "centre-back": "CB", "center-back": "CB", fullback: "RB", "right-back": "RB", "left-back": "LB", wingback: "RWB", "right wing-back": "RWB", "left wing-back": "LWB", midfielder: "CM", "central midfielder": "CM", winger: "RW", forward: "ST", striker: "ST" };
const normalise = (value) => String(value || "").trim().toLowerCase();
const playerName = (p) => p.Name || p.name || p.player || [p.first_name, p.last_name].filter(Boolean).join(" ") || "Unnamed player";
const playerClub = (p) => p.Team || p.team || p.club || p.Club || "";
const playerPosition = (p) => {
  const raw = p.primary_position || p.position || p.PrimaryPosition || p["Primary Position"] || p["Position"] || "";
  const text = normalise(raw);
  return positionAliases[text] || String(raw).toUpperCase().split(/[ /]/)[0] || "CM";
};
const photoFor = (name) => `https://syjsmvvsvvprxibqoizw.supabase.co/storage/v1/object/public/player-photos/player-photos/${String(name || "").trim().split(/\s+/).filter(Boolean).map((x) => x.normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-zA-Z0-9]+/g, "_").toLowerCase()).join("_")}.png`;

export default function SquadPlanPage() {
  const { appState, updateAppState, profile: accountProfile } = useAuth();
  const [players, setPlayers] = useState([]);
  const [clubs, setClubs] = useState([]);
  const [club, setClub] = useState(() => localStorage.getItem("squadPlanClub") || accountProfile?.club || "");
  const [formation, setFormation] = useState(() => localStorage.getItem("squadPlanFormation") || "4-2-3-1");
  const [squad, setSquad] = useState(() => JSON.parse(localStorage.getItem("squadPlanPlayers") || "[]"));
  const [loading, setLoading] = useState(true);
  const [picker, setPicker] = useState(null);
  const [pickerSearch, setPickerSearch] = useState("");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      const { data, error } = await supabase.from("players").select("*").limit(10000);
      if (error) console.error("Squad plan player database error", error);
      if (!cancelled) {
        const next = data || [];
        setPlayers(next);
        setClubs([...new Set(next.map(playerClub).filter(Boolean))].sort((a, b) => a.localeCompare(b)));
        setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, []);

  const clubPlayers = useMemo(() => {
    const matching = players.filter((p) => normalise(playerClub(p)) === normalise(club));
    return [...new Map(matching.map((p) => [String(p["Player Id"] || p.player_id || p.playerId || p.id || playerName(p)), p])).values()];
  }, [players, club]);
  const rows = formationRows[formation] || formationRows["4-2-3-1"];
  const slots = useMemo(() => rows.flatMap((row) => row.map((position, index) => `${position}-${index}`)), [rows]);
  const planKey = club;

  useEffect(() => {
    if (!club || !clubPlayers.length) return;
    const existingIds = new Set(squad.filter((p) => p.planKey === planKey).map((p) => String(p.id)));
    const additions = clubPlayers.filter((p) => {
      const id = p["Player Id"] || p.player_id || p.playerId || p.id || playerName(p);
      return !existingIds.has(String(id));
    }).map((p, index) => {
      const id = p["Player Id"] || p.player_id || p.playerId || p.id || playerName(p);
      const preferred = playerPosition(p);
      const slot = slots.find((x) => x.split("-")[0] === preferred) || slots[index % slots.length] || "CM-0";
      return { ...p, id, player: playerName(p), club: playerClub(p), position: preferred, slot, planKey };
    });
    if (!additions.length) return;
    setSquad((current) => {
      const next = [...current.filter((p) => p.planKey !== planKey), ...current.filter((p) => p.planKey === planKey), ...additions];
      localStorage.setItem("squadPlanPlayers", JSON.stringify(next));
      return next;
    });
  }, [club, clubPlayers, planKey, slots, squad]);

  const currentPlayers = squad.filter((p) => p.planKey === planKey);
  const pickerPlayers = currentPlayers.filter((p) => normalise(p.player).includes(normalise(pickerSearch)));
  const persist = (next) => {
    setSquad(next);
    localStorage.setItem("squadPlanPlayers", JSON.stringify(next));
    updateAppState({ assignments: appState?.assignments || [], shortlists: appState?.shortlists || [], tags: appState?.tags || [], squadPlan: { club, formation, players: next } });
  };
  const changeFormation = (value) => {
    const nextSlots = formationRows[value].flatMap((row) => row.map((position, index) => `${position}-${index}`));
    const used = {};
    const next = squad.map((p) => {
      if (p.planKey !== planKey) return p;
      const role = String(p.position || "CM").toUpperCase();
      const matches = nextSlots.filter((slot) => slot.startsWith(`${role}-`));
      const index = used[role] || 0;
      used[role] = index + 1;
      return { ...p, slot: matches[index] || nextSlots[index % nextSlots.length] || "CM-0", planKey: club };
    });
    setFormation(value); localStorage.setItem("squadPlanFormation", value); persist(next);
  };
  const remove = (id) => persist(squad.filter((p) => !(p.planKey === planKey && String(p.id) === String(id))));
  const place = (id, slot) => {
    persist(squad.map((p) => p.planKey === planKey && String(p.id) === String(id) ? { ...p, slot } : p));
    setPicker(null); setPickerSearch("");
  };
  const zone = (position, index) => {
    const slot = `${position}-${index}`;
    const listed = currentPlayers.filter((p) => p.slot === slot);
    return <div className="sr-zone" key={slot} data-count={`${listed.length} player${listed.length === 1 ? "" : "s"}`}><div className="sr-zone-head"><strong>{position}</strong><span>{listed.length ? listed.length : ""}</span><button type="button" onClick={() => { setPicker(picker === slot ? null : slot); setPickerSearch(""); }}>+</button></div><div className="sr-zone-list">{listed.map((p) => <div className="sr-pitch-player-card" key={String(p.id)}><button type="button"><img className="sr-shortlist-player-photo" src={photoFor(p.player)} alt="" onError={(e) => { e.currentTarget.style.display = "none"; }} /><span><strong>{p.player}</strong><small>{p.club}</small></span></button><button className="sr-slot-remove" type="button" onClick={() => remove(p.id)}>×</button></div>)}</div>{picker === slot && <div className="sr-zone-picker"><input autoFocus placeholder="Search club players" value={pickerSearch} onChange={(e) => setPickerSearch(e.target.value)} />{pickerPlayers.slice(0, 12).map((p) => <button type="button" key={String(p.id)} onClick={() => place(p.id, slot)}>{p.player}<small>{p.position || "Player"}</small></button>)}</div>}</div>;
  };

  return <main className="sr-page sr-shortlist-view sr-squad-plan"><button className="sr-back" onClick={() => window.history.back()}>‹ Back</button><section className="sr-dashboard-head"><div><div className="sr-kicker">SQUAD PLAN</div><h1>Plan Your Squad</h1><p>Choose a club to automatically load its players into a formation plan.</p></div><div className="sr-shortlist-head-actions"><label className="sr-field"><span>Club</span><input list="squad-plan-clubs" className="sr-formation-select" placeholder="Start typing your club" value={club} onChange={(e) => { setClub(e.target.value); localStorage.setItem("squadPlanClub", e.target.value); }} /><datalist id="squad-plan-clubs">{clubs.map((x) => <option key={x} value={x} />)}</datalist></label><label className="sr-field"><span>Formation</span><select className="sr-formation-select" value={formation} onChange={(e) => changeFormation(e.target.value)}>{formations.map((x) => <option key={x}>{x}</option>)}</select></label></div></section>{loading && <div className="sr-empty">Loading players…</div>}{!loading && club && !clubPlayers.length && <div className="sr-empty">No players found for this club.</div>}{club && clubPlayers.length > 0 && <div className="sr-tag-legend"><span>{clubPlayers.length} players loaded</span><span>{currentPlayers.length} in squad plan</span></div>}<div className="sr-real-pitch"><div className="sr-goal-box top" />{rows.map((row, i) => <div className="sr-pitch-row" key={i}>{row.map((pos, j) => zone(pos, j))}</div>)}<div className="sr-centre-circle" /><div className="sr-halfway-line" /><div className="sr-goal-box bottom" /></div></main>;
}
