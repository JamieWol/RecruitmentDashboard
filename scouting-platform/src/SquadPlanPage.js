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
const normalise = (value) => String(value || "").trim().toLowerCase();
const playerName = (p) => p.Name || p.name || p.player || [p.first_name, p.last_name].filter(Boolean).join(" ") || "Unnamed player";
const playerClub = (p) => p.Team || p.team || p.club || p.Club || "";
const photoBase = "https://syjsmvvsvvprxibqoizw.supabase.co/storage/v1/object/public/player-photos/player-photos/";
const photoFor = (name) => `${photoBase}${String(name || "").trim().split(/\s+/).filter(Boolean).map((x) => x.normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-zA-Z0-9]+/g, "_").toLowerCase()).join("_")}.png`;
const photoCandidates = (name) => { const raw = String(name || "").trim(); const short = raw.split(/\s+/).length > 2 ? `${raw.split(/\s+/)[0]} ${raw.split(/\s+/).at(-1)}` : raw; const bases = [...new Set([raw, short].map((x) => x.replace(/[^\p{L}\p{N}]+/gu, "_").replace(/^_+|_+$/g, "")))]; return [...new Set(bases.flatMap((x) => [x, x.toLowerCase(), x.toUpperCase(), `_${x}`, `_${x.toLowerCase()}`, `__${x}`, x.normalize("NFD").replace(/[̀-ͯ]/g, "")]).map((x) => `${photoBase}${x}.png`))]; };
const imageSource = (p) => p.Photo || p.photo || p._photoUrl || p.photoUrl || p.photo_url || p.playerPhoto || p.Image || p.image || p["Photo URL"] || p.image_url || photoFor(playerName(p));
const imageFallback = (e, name) => { const image = e.currentTarget; const candidates = photoCandidates(name); const next = Number(image.dataset.photoFallback || 0) + 1; if (candidates[next - 1]) { image.dataset.photoFallback = String(next); image.src = candidates[next - 1]; } else image.style.display = "none"; };
const defaultTags = [];

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
  const [tags, setTags] = useState(() => JSON.parse(localStorage.getItem("squadPlanTags") || JSON.stringify(defaultTags)));
  const [tagPicker, setTagPicker] = useState(null);
  const [newTagName, setNewTagName] = useState("");
  const [newTagColor, setNewTagColor] = useState("#62dcff");
  const [removedIds, setRemovedIds] = useState(() => JSON.parse(localStorage.getItem("squadPlanRemoved") || "{}"));
  const [savedMessage, setSavedMessage] = useState("");
  const [positionLabels, setPositionLabels] = useState(() => JSON.parse(localStorage.getItem("squadPlanPositionLabels") || "{}"));

  useEffect(() => {
    const saved = appState?.squadPlan;
    if (!saved) return;
    const hasLocalPlan = Boolean(localStorage.getItem("squadPlanPlayers"));
    if (!hasLocalPlan && Array.isArray(saved.players)) {
      setSquad(saved.players);
      localStorage.setItem("squadPlanPlayers", JSON.stringify(saved.players));
    }
    if (!localStorage.getItem("squadPlanClub") && saved.club) {
      setClub(saved.club);
      localStorage.setItem("squadPlanClub", saved.club);
    }
    if (!localStorage.getItem("squadPlanFormation") && saved.formation) {
      setFormation(saved.formation);
      localStorage.setItem("squadPlanFormation", saved.formation);
    }
  }, [appState]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      const all = [];
      let offset = 0;
      while (true) {
        const { data, error } = await supabase.from("players").select("*").range(offset, offset + 999);
        if (error) { console.error("Squad plan player database error", error); break; }
        all.push(...(data || []));
        if (!data || data.length < 1000) break;
        offset += 1000;
      }
      if (!cancelled) {
        const next = all;
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
  const planKey = club;
  const labelKey = (slot) => `${planKey}:${formation}:${slot}`;
  const setPositionLabel = (slot, value) => {
    const next = { ...positionLabels, [labelKey(slot)]: value };
    setPositionLabels(next); localStorage.setItem("squadPlanPositionLabels", JSON.stringify(next));
  };

  const currentPlayers = squad.filter((p) => p.planKey === planKey);
  const visibleClubPlayers = clubPlayers.filter((p) => !removedIds[planKey]?.includes(String(p["Player Id"] || p.player_id || p.playerId || p.id || playerName(p))));
  const allUniquePlayers = useMemo(() => [...new Map(players.map((p) => [String(p["Player Id"] || p.player_id || p.playerId || p.id || playerName(p)), p])).values()], [players]);
  const pickerPlayers = allUniquePlayers.filter((p) => normalise(playerName(p)).includes(normalise(pickerSearch)));
  const [dragging, setDragging] = useState(null);
  const persist = (next) => {
    setSquad(next);
    localStorage.setItem("squadPlanPlayers", JSON.stringify(next));
    updateAppState({ assignments: appState?.assignments || [], shortlists: appState?.shortlists || [], tags, squadPlan: { club, formation, players: next } });
  };
  const toggleTag = (id, tagId) => {
    const next = squad.map((p) => p.planKey === planKey && String(p.id) === String(id) ? { ...p, tags: p.tags?.includes(tagId) ? [] : [tagId] } : p);
    persist(next); setTagPicker(null);
  };
  const createTag = () => {
    if (!newTagName.trim()) return;
    const next = [...tags, { id: `squad-${Date.now()}`, name: newTagName.trim(), color: newTagColor }];
    setTags(next); localStorage.setItem("squadPlanTags", JSON.stringify(next)); setNewTagName("");
  };
  const tagMenu = (p) => tagPicker === String(p.id) && <div className="sr-squad-tag-menu">{tags.map((tag) => <button type="button" key={tag.id} onClick={() => toggleTag(p.id, tag.id)}><i style={{ background: tag.color }} />{p.tags?.includes(tag.id) ? "✓ " : ""}{tag.name}</button>)}{p.tags?.length > 0 && <button type="button" className="sr-remove-tag" onClick={() => { persist(squad.map((item) => item.planKey === planKey && String(item.id) === String(p.id) ? { ...item, tags: [] } : item)); setTagPicker(null); }}>Remove tag</button>}</div>;
  const tagStyle = (p) => { const tag = tags.find((item) => p.tags?.includes(item.id)); return tag ? { backgroundColor: tag.color, borderColor: tag.color, color: "#063d74" } : { backgroundColor: "#e9f4fb", borderColor: "#b4d2e8", color: "#063d74" }; };
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
  const remove = (id) => {
    const key = String(id);
    const nextRemoved = { ...removedIds, [planKey]: [...new Set([...(removedIds[planKey] || []), key])] };
    setRemovedIds(nextRemoved); localStorage.setItem("squadPlanRemoved", JSON.stringify(nextRemoved));
    persist(squad.filter((p) => !(p.planKey === planKey && String(p.id) === key)));
  };
  const savePlan = () => { persist(squad); setSavedMessage("Squad plan saved"); window.setTimeout(() => setSavedMessage(""), 2200); };
  const place = (source, slot) => {
    const id = source["Player Id"] || source.player_id || source.playerId || source.id || playerName(source);
    const existing = currentPlayers.find((p) => String(p.id) === String(id));
    const player = existing || { ...source, id, player: playerName(source), club: playerClub(source), planKey };
    const next = existing ? squad.map((p) => p.planKey === planKey && String(p.id) === String(id) ? { ...p, slot } : p) : [...squad, { ...player, slot }];
    persist(next);
    setPicker(null); setPickerSearch("");
  };
  const dropPlayer = (slot) => { if (dragging) place(dragging, slot); setDragging(null); };
  const rosterCard = (p) => {
    const id = p["Player Id"] || p.player_id || p.playerId || p.id || playerName(p);
    const selected = currentPlayers.find((x) => String(x.id) === String(id));
    return <div className="sr-squad-roster-wrap" key={String(id)}><div className="sr-pitch-player-card sr-squad-roster-card" style={selected ? tagStyle(selected) : undefined} draggable onDragStart={() => setDragging(p)}><img className="sr-shortlist-player-photo" src={imageSource(p)} alt="" onError={(e) => imageFallback(e, playerName(p))} /><span><strong>{playerName(p)}</strong><small>{p.primary_position || p.position || "Player"}</small></span>{selected && <button type="button" className="sr-squad-tag-button" onClick={() => setTagPicker(tagPicker === String(id) ? null : String(id))}>●</button>}<button type="button" className="sr-slot-remove" title="Remove from squad plan" onClick={() => remove(id)}>×</button></div>{selected && tagMenu(selected)}</div>;
  };
  const zone = (position, index) => {
    const slot = `${position}-${index}`;
    const listed = currentPlayers.filter((p) => p.slot === slot);
    return <div className="sr-zone" key={slot} data-count={`${listed.length} player${listed.length === 1 ? "" : "s"}`} onDragOver={(e) => e.preventDefault()} onDrop={() => dropPlayer(slot)}><div className="sr-zone-head"><input className="sr-position-label" aria-label={`Edit ${position} label`} value={positionLabels[labelKey(slot)] ?? position} onChange={(e) => setPositionLabel(slot, e.target.value)} /><span>{listed.length ? listed.length : ""}</span><button type="button" onClick={() => { setPicker(picker === slot ? null : slot); setPickerSearch(""); }}>+</button></div><div className="sr-zone-list">{listed.map((p) => <div className="sr-squad-pitch-wrap" key={String(p.id)}><div className="sr-pitch-player-card" style={tagStyle(p)} draggable onDragStart={() => setDragging(p)}><button type="button"><img className="sr-shortlist-player-photo" src={imageSource(p)} alt="" onError={(e) => imageFallback(e, p.player)} /><span><strong>{p.player}</strong><small>{p.club}</small></span></button><button className="sr-squad-tag-button" type="button" onClick={() => setTagPicker(tagPicker === String(p.id) ? null : String(p.id))}>●</button><button className="sr-slot-remove" type="button" onClick={() => remove(p.id)}>×</button></div>{tagMenu(p)}</div>)}</div>{picker === slot && <div className="sr-zone-picker"><input autoFocus placeholder="Search any player" value={pickerSearch} onChange={(e) => setPickerSearch(e.target.value)} />{pickerPlayers.slice(0, 12).map((p) => <button type="button" key={String(p["Player Id"] || p.id || playerName(p))} onClick={() => place(p, slot)}>{playerName(p)}<small>{playerClub(p) || "Player"}</small></button>)}</div>}</div>;
  };

  return <main className="sr-page sr-shortlist-view sr-squad-plan"><button className="sr-back" onClick={() => window.history.back()}>‹ Back</button><section className="sr-dashboard-head"><div><div className="sr-kicker">SQUAD PLAN</div><h1>Plan Your Squad</h1><p>Choose a club, then add players yourself to each position. Use + to add any player from the database.</p></div><div className="sr-shortlist-head-actions"><label className="sr-field"><span>Club</span><input list="squad-plan-clubs" className="sr-formation-select" placeholder="Start typing your club" value={club} onChange={(e) => { setClub(e.target.value); localStorage.setItem("squadPlanClub", e.target.value); }} /><datalist id="squad-plan-clubs">{clubs.map((x) => <option key={x} value={x} />)}</datalist></label><label className="sr-field"><span>Formation</span><select className="sr-formation-select" value={formation} onChange={(e) => changeFormation(e.target.value)}>{formations.map((x) => <option key={x}>{x}</option>)}</select></label><button type="button" className="sr-cyan sr-squad-save" onClick={savePlan}>Save Squad Plan</button>{savedMessage && <small className="sr-saved-message">{savedMessage}</small>}</div></section>{loading && <div className="sr-empty">Loading players…</div>}{!loading && club && !clubPlayers.length && <div className="sr-empty">No players found for this club.</div>}{club && clubPlayers.length > 0 && <div className="sr-tag-legend"><span>{visibleClubPlayers.length} players available</span><span>{currentPlayers.length} added to squad plan</span></div>}<div className="sr-squad-tag-tools"><strong>Custom tags</strong>{tags.map((tag) => <span key={tag.id}><i style={{ background: tag.color }} />{tag.name}</span>)}<input placeholder="Create tag" value={newTagName} onChange={(e) => setNewTagName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); createTag(); } }} /><input className="sr-tag-color" type="color" value={newTagColor} onChange={(e) => setNewTagColor(e.target.value)} /><button type="button" className="sr-outline" onClick={createTag}>Add tag</button></div><div className="sr-squad-plan-layout"><aside className="sr-squad-roster"><h2>{club || "Club"} players</h2><p>Remove players from this plan with ×.</p>{visibleClubPlayers.map(rosterCard)}<h3 className="sr-squad-any-title">Add any player</h3><input className="sr-squad-global-search" placeholder="Search any player" value={pickerSearch} onChange={(e) => setPickerSearch(e.target.value)} />{pickerSearch && pickerPlayers.slice(0, 15).map(rosterCard)}</aside><div className="sr-real-pitch"><div className="sr-goal-box top" />{rows.map((row, i) => <div className="sr-pitch-row" key={i}>{row.map((pos, j) => zone(pos, j))}</div>)}<div className="sr-centre-circle" /><div className="sr-halfway-line" /><div className="sr-goal-box bottom" /></div></div></main>;
}
