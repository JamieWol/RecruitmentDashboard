import { ClubName } from "./ClubBadge";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
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
const photoCandidates = (name) => {
  const raw = String(name || "").trim();
  const parts = raw.split(/\s+/).filter(Boolean);
  const short = parts.length > 2 ? `${parts[0]} ${parts.at(-1)}` : raw;
  const twoNameVariants = parts.length > 2 ? parts.slice(0, -1).map((_, index) => parts.slice(index, index + 2).join(" ")) : [];
  const nameVariants = [raw, ...twoNameVariants, short, raw.replace(/macaulay/gi, "macauley"), ...twoNameVariants.map((x) => x.replace(/macaulay/gi, "macauley")), short.replace(/macaulay/gi, "macauley"), raw.replace(/macauley/gi, "macaulay"), ...twoNameVariants.map((x) => x.replace(/macauley/gi, "macaulay")), short.replace(/macauley/gi, "macaulay")];
  const unicodeBases = nameVariants.map((x) => x.replace(/[^\p{L}\p{N}]+/gu, "_").replace(/^_+|_+$/g, ""));
  const legacyBases = nameVariants.map((x) => x.trim().replace(/[^a-z0-9]/gi, "_").toLowerCase().replace(/^_+|_+$/g, ""));
  const bases = [...new Set([...unicodeBases, ...legacyBases, ...unicodeBases.map((x) => x.normalize("NFD").replace(/[̀-ͯ]/g, ""))])];
  return [...new Set(bases.flatMap((x) => [x, x.toLowerCase(), x.toUpperCase(), `_${x}`, `_${x.toLowerCase()}`, `__${x}`]).map((x) => `${photoBase}${x}.png`))];
};
const imageSource = (p) => p.Photo || p.photo || p._photoUrl || p.photoUrl || p.photo_url || p.playerPhoto || p.player_photo || p.profile_photo || p.profilePhoto || p.headshot || p.headshot_url || p.Image || p.image || p["Photo URL"] || p["Photo Url"] || p["Image URL"] || p.image_url || photoFor(playerName(p));
const imageFallback = (e, name) => { const image = e.currentTarget; const candidates = photoCandidates(name); const next = Number(image.dataset.photoFallback || 0) + 1; if (candidates[next - 1]) { image.dataset.photoFallback = String(next); image.src = candidates[next - 1]; } else image.style.display = "none"; };
const defaultTags = [];

export default function SquadPlanPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { appState, updateAppState, profile: accountProfile, user } = useAuth();
  const query = new URLSearchParams(location.search);
  const planId = query.get("plan");
  const sharedId = query.get("shared");
  const isNewPlan = query.has("new");
  const [players, setPlayers] = useState([]);
  const [clubs, setClubs] = useState([]);
  const [club, setClub] = useState(accountProfile?.club || "");
  const [formation, setFormation] = useState("4-2-3-1");
  const [squad, setSquad] = useState([]);
  const [loading, setLoading] = useState(true);
  const [picker, setPicker] = useState(null);
  const [pickerSearch, setPickerSearch] = useState("");
  const [tags, setTags] = useState(defaultTags);
  const [tagPicker, setTagPicker] = useState(null);
  const [newTagName, setNewTagName] = useState("");
  const [newTagColor, setNewTagColor] = useState("#62dcff");
  const [removedIds, setRemovedIds] = useState({});
  const [savedMessage, setSavedMessage] = useState("");
  const [squadName, setSquadName] = useState("");
  const [positionLabels, setPositionLabels] = useState({});
  const hydratedPlan = useRef("");
  const [sharedPlan, setSharedPlan] = useState(null);
  const [sharedPlanError, setSharedPlanError] = useState("");
  const [sharedPlanLoading, setSharedPlanLoading] = useState(false);
  const readOnly = Boolean(sharedId && sharedPlan?.sharedPermission !== "edit");
  const openPlayer = (p) => { localStorage.setItem("scoutingProfilePlayer", playerName(p)); localStorage.setItem("scoutingProfileOrigin", "squad-plan"); navigate("/scouting-reports"); };

  useEffect(() => {
    if (!sharedId || !user) { setSharedPlan(null); setSharedPlanError(""); return; }
    let cancelled = false;
    const load = async () => {
      setSharedPlanLoading(true); setSharedPlanError("");
      const { data: share, error: shareError } = await supabase.from("shortlist_shares").select("permission,owner_id").eq("shortlist_id", sharedId).eq("member_id", user.id).maybeSingle();
      if (shareError || !share) {
        if (!cancelled) { setSharedPlan(null); setSharedPlanError("You do not have access to this shared squad plan."); setSharedPlanLoading(false); }
        return;
      }
      const { data: snapshot, error: snapshotError } = await supabase.from("shared_shortlists").select("shortlist,owner_id").eq("shortlist_id", sharedId).maybeSingle();
      if (snapshotError || !snapshot?.shortlist || snapshot.shortlist.kind !== "squad-plan") {
        if (!cancelled) { setSharedPlan(null); setSharedPlanError(snapshotError?.message || "This shared squad plan could not be loaded."); setSharedPlanLoading(false); }
        return;
      }
      if (!cancelled) {
        setSharedPlan({ ...snapshot.shortlist, owner_id: snapshot.owner_id, sharedPermission: share.permission || "view" });
        setSharedPlanLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [sharedId, user]);

  useEffect(() => {
    if (sharedId ? !sharedPlan : !appState) return;
    const hydrationKey = `${user?.id || ""}:${sharedId || planId || "draft"}:${isNewPlan}`;
    if (hydratedPlan.current === hydrationKey) return;
    hydratedPlan.current = hydrationKey;
    const state = sharedId ? sharedPlan : appState.squadPlan || {};
    const saved = (state.savedPlans || []).find((item) => String(item.id) === String(planId));
    if (saved) {
      setSquadName(saved.name || ""); setClub(saved.club || ""); setFormation(saved.formation || "4-2-3-1");
      setSquad(Array.isArray(saved.players) ? saved.players : []); setPositionLabels(saved.positionLabels || {});
      setTags(Array.isArray(saved.tags) ? saved.tags : defaultTags); setRemovedIds(saved.removedIds || {});
    } else if (isNewPlan) {
      setSquadName(""); setClub(""); setFormation("4-2-3-1"); setSquad([]);
      setPositionLabels({}); setTags(defaultTags); setRemovedIds({});
    } else {
      setSquadName(state.name || ""); setClub(state.club || accountProfile?.club || "");
      setFormation(state.formation || "4-2-3-1"); setSquad(Array.isArray(state.players) ? state.players : []);
      setPositionLabels(state.positionLabels || {}); setTags(state.tags || defaultTags); setRemovedIds(state.removedIds || {});
    }
  }, [appState, planId, isNewPlan, accountProfile?.club, user, sharedId, sharedPlan]);

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
    if (readOnly) return;
    const next = { ...positionLabels, [labelKey(slot)]: value };
    setPositionLabels(next); persist(squad, { positionLabels: next });
  };

  const currentPlayers = squad.filter((p) => p.planKey === planKey);
  const visibleClubPlayers = clubPlayers.filter((p) => !removedIds[planKey]?.includes(String(p["Player Id"] || p.player_id || p.playerId || p.id || playerName(p))));
  const allUniquePlayers = useMemo(() => [...new Map(players.map((p) => [String(p["Player Id"] || p.player_id || p.playerId || p.id || playerName(p)), p])).values()], [players]);
  const pickerPlayers = allUniquePlayers.filter((p) => normalise(playerName(p)).includes(normalise(pickerSearch)));
  const [dragging, setDragging] = useState(null);
  const persist = (next, extra = {}) => {
    setSquad(next);
    if (sharedId) {
      if (readOnly || !sharedPlan) return;
      const updatedPlan = { ...sharedPlan, club, formation, players: next, tags, removedIds, positionLabels, ...extra, updatedAt: new Date().toISOString() };
      setSharedPlan(updatedPlan);
      supabase.from("shared_shortlists").update({ shortlist: updatedPlan, updated_at: updatedPlan.updatedAt }).eq("shortlist_id", sharedId).eq("owner_id", sharedPlan.owner_id)
        .then(({ error }) => { if (error) console.error("Could not update the shared squad plan", error); });
      return;
    }
    updateAppState({ ...appState, squadPlan: { ...(appState?.squadPlan || {}), club, formation, players: next, tags, removedIds, positionLabels, ...extra } });
  };
  const toggleTag = (id, tagId) => {
    const next = squad.map((p) => p.planKey === planKey && String(p.id) === String(id) ? { ...p, tags: p.tags?.includes(tagId) ? [] : [tagId] } : p);
    persist(next); setTagPicker(null);
  };
  const createTag = () => {
    if (readOnly) return;
    if (!newTagName.trim()) return;
    const next = [...tags, { id: `squad-${Date.now()}`, name: newTagName.trim(), color: newTagColor }];
    setTags(next); persist(squad, { tags: next }); setNewTagName("");
  };
  const tagMenu = (p) => !readOnly && tagPicker === String(p.id) && <div className="sr-squad-tag-menu">{tags.map((tag) => <button type="button" key={tag.id} onClick={() => toggleTag(p.id, tag.id)}><i style={{ background: tag.color }} />{p.tags?.includes(tag.id) ? "✓ " : ""}{tag.name}</button>)}{p.tags?.length > 0 && <button type="button" className="sr-remove-tag" onClick={() => { persist(squad.map((item) => item.planKey === planKey && String(item.id) === String(p.id) ? { ...item, tags: [] } : item)); setTagPicker(null); }}>Remove tag</button>}</div>;
  const tagStyle = (p) => { const tag = tags.find((item) => p.tags?.includes(item.id)); return tag ? { backgroundColor: tag.color, borderColor: tag.color, color: "#063d74" } : { backgroundColor: "#e9f4fb", borderColor: "#b4d2e8", color: "#063d74" }; };
  const changeFormation = (value) => {
    if (readOnly) return;
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
    setFormation(value); persist(next, { formation: value });
  };
  const remove = (id) => {
    if (readOnly) return;
    const key = String(id);
    const nextRemoved = { ...removedIds, [planKey]: [...new Set([...(removedIds[planKey] || []), key])] };
    setRemovedIds(nextRemoved);
    persist(squad.filter((p) => !(p.planKey === planKey && String(p.id) === key)), { removedIds: nextRemoved });
  };
  const savePlan = () => {
    if (readOnly || sharedId) return;
    const enteredName = window.prompt("Name this squad plan", squadName || `${club || "New"} Squad Plan`);
    if (!enteredName?.trim()) return;
    const id = planId || `squad-${Date.now()}`;
    const record = { id, name: enteredName.trim(), club, formation, players: currentPlayers, positionLabels, tags, removedIds, updatedAt: new Date().toISOString() };
    const savedPlans = [...(appState?.squadPlan?.savedPlans || []).filter((item) => String(item.id) !== String(id)), record];
    setSquadName(record.name); persist(squad, { savedPlans }); setSavedMessage("Squad plan saved"); window.setTimeout(() => setSavedMessage(""), 2200); navigate(`/squad-plan?plan=${encodeURIComponent(id)}`, { replace: true });
  };
  const place = (source, slot) => {
    if (readOnly) return;
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
    return <div className="sr-squad-roster-wrap" key={String(id)}><div className="sr-pitch-player-card sr-squad-roster-card" style={selected ? tagStyle(selected) : undefined} draggable={!readOnly} onDragStart={() => !readOnly && setDragging(p)} onClick={() => openPlayer(p)}><img className="sr-shortlist-player-photo" src={imageSource(p)} alt="" onError={(e) => imageFallback(e, playerName(p))} /><span><strong>{playerName(p)}</strong><small>{p.primary_position || p.position || "Player"}</small></span>{selected && !readOnly && <button type="button" className="sr-squad-tag-button" onClick={(e) => { e.stopPropagation(); setTagPicker(tagPicker === String(id) ? null : String(id)); }}>●</button>}{selected && !readOnly && <button type="button" className="sr-slot-remove" title="Remove from squad plan" onClick={(e) => { e.stopPropagation(); remove(id); }}>×</button>}</div>{selected && tagMenu(selected)}</div>;
  };
  const zone = (position, index) => {
    const slot = `${position}-${index}`;
    const listed = currentPlayers.filter((p) => p.slot === slot);
    return <div className="sr-zone" key={slot} data-count={`${listed.length} player${listed.length === 1 ? "" : "s"}`} onDragOver={(e) => e.preventDefault()} onDrop={() => dropPlayer(slot)}><div className="sr-zone-head"><input className="sr-position-label" aria-label={`Edit ${position} label`} value={positionLabels[labelKey(slot)] ?? position} disabled={readOnly} onChange={(e) => setPositionLabel(slot, e.target.value)} /><span>{listed.length ? listed.length : ""}</span>{!readOnly && <button type="button" onClick={() => { setPicker(picker === slot ? null : slot); setPickerSearch(""); }}>+</button>}</div><div className="sr-zone-list">{listed.map((p) => <div className="sr-squad-pitch-wrap" key={String(p.id)}><div className="sr-pitch-player-card" style={tagStyle(p)} draggable={!readOnly} onDragStart={() => !readOnly && setDragging(p)} onClick={() => openPlayer(p)}><button type="button" onClick={(e) => e.stopPropagation()}><img className="sr-shortlist-player-photo" src={imageSource(p)} alt="" onError={(e) => imageFallback(e, p.player)} /><span><strong>{p.player}</strong><small><ClubName club={p.club || playerClub(p)} size={15} /></small></span></button>{!readOnly && <button className="sr-squad-tag-button" type="button" onClick={(e) => { e.stopPropagation(); setTagPicker(tagPicker === String(p.id) ? null : String(p.id)); }}>●</button>}{!readOnly && <button className="sr-slot-remove" type="button" onClick={(e) => { e.stopPropagation(); remove(p.id); }}>×</button>}</div>{tagMenu(p)}</div>)}</div>{!readOnly && picker === slot && <div className="sr-zone-picker"><input autoFocus placeholder="Search any player" value={pickerSearch} onChange={(e) => setPickerSearch(e.target.value)} />{pickerPlayers.slice(0, 12).map((p) => <button type="button" key={String(p["Player Id"] || p.id || playerName(p))} onClick={() => place(p, slot)}>{playerName(p)}<small><ClubName club={playerClub(p)} fallback="Player" size={15} /></small></button>)}</div>}</div>;
  };

  return <main className="sr-page sr-shortlist-view sr-squad-plan">
    <button className="sr-back" onClick={() => window.history.back()}>‹ Back</button>
    {sharedPlanLoading && <div className="sr-empty">Loading shared squad plan…</div>}
    {sharedPlanError && <div className="sr-empty">{sharedPlanError}</div>}
    <section className="sr-dashboard-head"><div><div className="sr-kicker">{sharedId ? "SHARED SQUAD PLAN" : "SQUAD PLAN"}</div><h1>{squadName || "Plan Your Squad"}</h1><p>{sharedId ? readOnly ? "Shared with you by another scout · View only" : "Shared with you · You can edit this plan" : "Choose a club, then add players yourself to each position. Use + to add any player from the database."}</p></div><div className="sr-shortlist-head-actions"><label className="sr-field"><span>Club</span><input list="squad-plan-clubs" className="sr-formation-select" placeholder="Start typing your club" value={club} disabled={readOnly} onChange={(e) => setClub(e.target.value)} onBlur={() => persist(squad, { club })} /><datalist id="squad-plan-clubs">{clubs.map((x) => <option key={x} value={x} />)}</datalist></label><label className="sr-field"><span>Formation</span><select className="sr-formation-select" value={formation} disabled={readOnly} onChange={(e) => changeFormation(e.target.value)}>{formations.map((x) => <option key={x}>{x}</option>)}</select></label>{!sharedId && <button type="button" className="sr-cyan sr-squad-save" onClick={savePlan}>Save Squad Plan</button>}{savedMessage && <small className="sr-saved-message">{savedMessage}</small>}</div></section>
    {loading && <div className="sr-empty">Loading players…</div>}{!loading && club && !clubPlayers.length && <div className="sr-empty">No players found for this club.</div>}{club && clubPlayers.length > 0 && <div className="sr-tag-legend"><span>{visibleClubPlayers.length} players available</span><span>{currentPlayers.length} added to squad plan</span></div>}
    {!readOnly && <div className="sr-squad-tag-tools"><strong>Custom tags</strong>{tags.map((tag) => <span key={tag.id}><i style={{ background: tag.color }} />{tag.name}</span>)}<input placeholder="Create tag" value={newTagName} onChange={(e) => setNewTagName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); createTag(); } }} /><input className="sr-tag-color" type="color" value={newTagColor} onChange={(e) => setNewTagColor(e.target.value)} /><button type="button" className="sr-outline" onClick={createTag}>Add tag</button></div>}
    {(!sharedId || sharedPlan) && !sharedPlanLoading && <div className="sr-squad-plan-layout"><aside className="sr-squad-roster"><h2><ClubName club={club} fallback="Club" /></h2><p>{readOnly ? "Players in this squad plan." : "Remove players from this plan with ×."}</p>{visibleClubPlayers.map(rosterCard)}{!readOnly && <><h3 className="sr-squad-any-title">Add any player</h3><input className="sr-squad-global-search" placeholder="Search any player" value={pickerSearch} onChange={(e) => setPickerSearch(e.target.value)} />{pickerSearch && pickerPlayers.slice(0, 15).map(rosterCard)}</>}</aside><div className="sr-real-pitch"><div className="sr-goal-box top" />{rows.map((row, i) => <div className="sr-pitch-row" key={i}>{row.map((pos, j) => zone(pos, j))}</div>)}<div className="sr-centre-circle" /><div className="sr-halfway-line" /><div className="sr-goal-box bottom" /></div></div>}
  </main>;
}
