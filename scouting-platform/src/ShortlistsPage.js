import React, { useEffect, useMemo, useState } from "react";
import { supabase } from "./supabaseClient";
const formations = [
    "4-1-4-1",
    "4-2-2-2",
    "4-2-3-1",
    "4-3-1-2",
    "4-3-2-1",
    "4-3-3",
    "4-4-1-1",
    "4-4-2",
    "4-4-2 (Diamond)",
    "4-5-1",
    "3-4-3",
    "3-5-2",
  ],
  formationRows = {
    "4-1-4-1": [
      ["LW", "CF", "RW"],
      ["LM", "CM", "CM", "RM"],
      ["DM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-2-2-2": [
      ["CF", "CF"],
      ["AM", "AM"],
      ["DM", "DM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-2-3-1": [
      ["ST"],
      ["LW", "AM", "RW"],
      ["DM", "DM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-3-1-2": [
      ["ST", "ST"],
      ["AM"],
      ["LCM", "CM", "RCM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-3-2-1": [
      ["ST"],
      ["AM", "AM"],
      ["LCM", "CM", "RCM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-3-3": [
      ["LW", "CF", "RW"],
      ["LCM", "CM", "RCM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-4-1-1": [
      ["ST"],
      ["AM"],
      ["LM", "LCM", "RCM", "RM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-4-2": [
      ["ST", "ST"],
      ["LM", "LCM", "RCM", "RM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-4-2 (Diamond)": [
      ["ST", "ST"],
      ["AM"],
      ["DM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-5-1": [
      ["ST"],
      ["LM", "LCM", "CM", "RCM", "RM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "3-4-3": [
      ["LW", "CF", "RW"],
      ["LM", "LCM", "RCM", "RM"],
      ["LCB", "CB", "RCB"],
      ["GK"],
    ],
    "3-5-2": [
      ["ST", "ST"],
      ["LM", "CM", "CM", "RM"],
      ["LCB", "CB", "RCB"],
      ["GK"],
    ],
  },
  flexRows = [
    ["LW", "CF", "RW"],
    ["LM", "AM", "RM"],
    ["LCM", "CM", "RCM"],
    ["LB", "LCB", "CB", "RCB", "RB"],
    ["GK"],
  ];
const shortlistPhoto=(name)=>`https://syjsmvvsvvprxibqoizw.supabase.co/storage/v1/object/public/player-photos/player-photos/${String(name||"").trim().split(/\s+/).filter(Boolean).map(x=>x.normalize("NFD").replace(/[̀-ͯ]/g,"").toLowerCase().replace(/[^a-z0-9]+/g,"_")).join("_")}.png`;
const defaultTags=[{id:"elite",name:"Elite",color:"#62dcff"},{id:"high",name:"High Potential",color:"#142b83"},{id:"value",name:"Value Potential",color:"#a56bc5"},{id:"monitor",name:"Monitor",color:"#ff7416"},{id:"contract",name:"Out of contract",color:"#22c55e"},{id:"deprioritise",name:"Deprioritise",color:"#f2b807"}];
export default function ShortlistsPage() {
  const [databasePlayers, setDatabasePlayers] = useState([]);
  const [lists, setLists] = useState(() =>
    JSON.parse(localStorage.getItem("scoutingShortlists") || "[]"),
  );
  const [current, setCurrent] = useState(null),
    [show, setShow] = useState(false),
    [name, setName] = useState(""),
    [type, setType] = useState("Formation"),
    [formation, setFormation] = useState("4-3-3"),
    [pick, setPick] = useState(null),
    [search, setSearch] = useState(""),
    [tagPlayer, setTagPlayer] = useState(null),
    [dragging, setDragging] = useState(null),
    [newPlayerOpen, setNewPlayerOpen] = useState(false),
    [newPlayerQuery, setNewPlayerQuery] = useState(""),
    [newPlayers, setNewPlayers] = useState([]),
    [tags, setTags] = useState(() => JSON.parse(localStorage.getItem("scoutingTags") || JSON.stringify(defaultTags))),
    [manageTags, setManageTags] = useState(false),
    [newTagName, setNewTagName] = useState(""),
    [newTagColor, setNewTagColor] = useState("#62dcff");
  useEffect(() => {
    if (!search.trim()) {
      setDatabasePlayers([]);
      return;
    }
    supabase
      .from("players")
      .select("*")
      .ilike("Name", search.trim())
      .limit(50)
      .then(({ data, error }) => {
        if (error) console.error("Player database error", error);
        if (data) setDatabasePlayers(data);
      });
  }, [search]);
  const published = useMemo(
    () =>
      JSON.parse(localStorage.getItem("scoutingAssignments") || "[]").filter(
        (x) => x.status === "Published",
      ),
    [],
  );
  const playerPool = useMemo(() => {
    const database = databasePlayers.map((p) => ({
      ...p,
      id:
        p.id || p.playerId || p.player_id || p["Player Id"] || p.Name || p.name,
      player:
        p.Name ||
        p.name ||
        [p.firstName || p.first_name, p.lastName || p.last_name]
          .filter(Boolean)
          .join(" "),
      club: p.club || p.team || p.Team,
    }));
    const reports = published.map((x) => ({
      ...x,
      id: x.playerId || x.id,
      player: x.player,
    }));
    return [
      ...new Map(
        [...database, ...reports]
          .filter((p) => p.player)
          .map((p) => [p.id || p.player, p]),
      ).values(),
    ];
  }, [databasePlayers, published]);
  const persist = (n) => {
    setLists(n);
    localStorage.setItem("scoutingShortlists", JSON.stringify(n));
  };
  const create = (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    const l = { id: Date.now(), name, type, formation, players: [] };
    persist([...lists, l]);
    setName("");
    setShow(false);
    setCurrent(l);
  };
  const add = (p, pos) => {
    if (current.players.some((x) => x.id === p.id && x.slot === pos)) return;
    const n = {
      ...current,
      players: [...current.players, { ...p, slot: pos }],
    };
    persist(lists.map((x) => (x.id === current.id ? n : x)));
    setCurrent(n);
    setPick(null);
    setSearch("");
  };
  const remove = (id, pos) => {
    const n = {
      ...current,
      players: current.players.filter((x) => !(x.id === id && x.slot === pos)),
    };
    persist(lists.map((x) => (x.id === current.id ? n : x)));
    setCurrent(n);
  };
  const toggleTag = (tagId) => {
    const n = {...tagPlayer, tags: tagPlayer.tags?.includes(tagId) ? tagPlayer.tags.filter(x=>x!==tagId) : [...(tagPlayer.tags||[]),tagId]};
    const updated={...current,players:current.players.map(x=>x.id===tagPlayer.id&&x.slot===tagPlayer.slot?n:x)};
    persist(lists.map(x=>x.id===current.id?updated:x)); setCurrent(updated); setTagPlayer(n);
  };
  const createTag = (e) => { e.preventDefault(); if (!newTagName.trim()) return; const n={id:`custom-${Date.now()}`,name:newTagName.trim(),color:newTagColor}; const next=[...tags,n]; setTags(next); localStorage.setItem("scoutingTags",JSON.stringify(next)); setNewTagName(""); };
  const changeFormation = (value) => { const slots=[...(formationRows[value]||[])].flatMap((row,i)=>row.map((pos,j)=>`${pos}-${j}`)); const players=current.players.map(p=>{const label=p.slot.split("-")[0]; const slot=slots.find(s=>s.split("-")[0]===label)||slots[0]; return {...p,slot};}); const n={...current,formation:value,players}; persist(lists.map(x=>x.id===current.id?n:x)); setCurrent(n); };
  const addSelectedPlayers = () => { if(!newPlayers.length)return; const fallback=((formationRows[current.formation]||flexRows).flat()[0]+"-0"); const players=[...current.players,...newPlayers.map(p=>({...p,slot:p.addSlot||fallback}))]; const n={...current,players}; persist(lists.map(x=>x.id===current.id?n:x)); setCurrent(n); setNewPlayers([]); setNewPlayerQuery(""); setNewPlayerOpen(false); };
  const reorder = (pos, targetId) => { if (!dragging || dragging.pos!==pos || dragging.id===targetId) return; const inPos=current.players.filter(x=>x.slot===pos); const from=inPos.findIndex(x=>x.id===dragging.id); const to=inPos.findIndex(x=>x.id===targetId); if(from<0||to<0)return; const ordered=[...inPos]; const [moved]=ordered.splice(from,1); ordered.splice(to,0,moved); const n={...current,players:[...current.players.filter(x=>x.slot!==pos),...ordered]}; persist(lists.map(x=>x.id===current.id?n:x)); setCurrent(n); setDragging(null); };
  const tagManager=manageTags&&<div className="sr-modal" onClick={()=>setManageTags(false)}><form className="sr-form sr-tag-modal" onClick={e=>e.stopPropagation()} onSubmit={createTag}><div className="sr-form-head"><div><div className="sr-kicker">SHORTLIST TAGS</div><h2>Manage Tags</h2></div><button type="button" className="sr-close" onClick={()=>setManageTags(false)}>×</button></div><label className="sr-field"><span>New tag</span><input value={newTagName} onChange={e=>setNewTagName(e.target.value)} placeholder="Tag name"/></label><div className="sr-color-palette">{["#62dcff","#142b83","#a56bc5","#ff7416","#22c55e","#f2b807","#ef4444","#3b82f6","#ec4899","#14b8a6","#64748b","#84cc16"].map(c=><button type="button" key={c} style={{background:c}} className={newTagColor===c?"selected":""} onClick={()=>setNewTagColor(c)} />)}</div><button className="sr-cyan">Add Tag</button><div className="sr-existing-tags">{tags.map(t=><span key={t.id} style={{background:t.color}}>{t.name}</span>)}</div></form></div>;
  const tagModal=tagPlayer&&<div className="sr-modal" onClick={()=>setTagPlayer(null)}><section className="sr-form sr-tag-modal" onClick={e=>e.stopPropagation()}><div className="sr-form-head"><div><div className="sr-kicker">PLAYER TAGS</div><h2>{tagPlayer.player}</h2></div><button className="sr-close" onClick={()=>setTagPlayer(null)}>×</button></div><p>Click a tag to add or remove it.</p><div className="sr-tag-list">{tags.map(t=><button key={t.id} style={{background:t.color,color:t.color==="#f2b807"?"#071d3d":"#fff"}} className={tagPlayer.tags?.includes(t.id)?"active":""} onClick={()=>toggleTag(t.id)}>{t.name}</button>)}</div><button className="sr-cyan" onClick={()=>setTagPlayer(null)}>Done</button></section></div>;
  const zone = (pos) => {
    const players = current.players.filter((x) => x.slot === pos),
      matches = playerPool.filter((x) =>
        x.player.toLowerCase().includes(search.toLowerCase()),
      );
    return (
      <div className="sr-zone" key={pos}>
        <div className="sr-zone-head">
          <strong>{pos}</strong>
          <span>{players.length} Players</span>
          <button
            onClick={() => {
              setPick(pick === pos ? null : pos);
              setSearch("");
            }}
          >
            +
          </button>
        </div>
        <div className="sr-zone-list">
          {players.map((p) => (
            <div className="sr-pitch-player-card" key={p.id} draggable onDragStart={()=>setDragging({id:p.id,pos})} onDragOver={e=>e.preventDefault()} onDrop={()=>reorder(pos,p.id)} style={{"--tag-color":tags.find(t=>(p.tags||[]).includes(t.id))?.color||"#f7fbff"}}>
              <button onClick={() => { localStorage.setItem("scoutingProfilePlayer", p.player); localStorage.setItem("scoutingProfileOrigin", "shortlists"); window.location.href="/scouting-reports"; }}>
                <img className="sr-shortlist-player-photo" src={shortlistPhoto(p.player)} alt="" onError={(e) => { e.currentTarget.style.display = "none"; }} />
                <span><strong>{p.player}</strong><small>{p.club || "Club not added"}</small><em className="sr-report-count">▤ {JSON.parse(localStorage.getItem("scoutingAssignments") || "[]").filter(x=>x.player===p.player&&x.status==="Published").length}</em></span>
              </button>
              <div className="sr-player-highlight" style={{background:tags.find(t=>(p.tags||[]).includes(t.id))?.color||"transparent"}} />
              <div className="sr-card-tags">{(p.tags||[]).map(id=>{const t=tags.find(x=>x.id===id);return t?<i key={id} title={t.name} style={{background:t.color}}/>:null})}</div><button onClick={() => remove(p.id, pos)}>×</button>
            </div>
          ))}
        </div>
        {pick === pos && (
          <div className="sr-zone-picker">
            <input
              autoFocus
              placeholder="Search player..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            {search.trim() &&
              matches.map((p) => (
                <button key={p.id} onClick={() => add(p, pos)}>
                  {p.player}
                  <small>{p.club || "Club not added"}</small>
                </button>
              ))}
          </div>
        )}
      </div>
    );
  };
  const del = (id) => {
    if (window.confirm("Delete this shortlist?")) {
      persist(lists.filter((x) => x.id !== id));
      setCurrent(null);
    }
  };
  const newPlayerMatches = playerPool.filter(p=>p.player.toLowerCase().includes(newPlayerQuery.toLowerCase()));
  const newPlayerModal = newPlayerOpen && <div className="sr-modal" onClick={()=>setNewPlayerOpen(false)}><section className="sr-form sr-new-player-modal" onClick={e=>e.stopPropagation()}><div className="sr-form-head"><h2>Add New Players</h2><button className="sr-close" onClick={()=>setNewPlayerOpen(false)}>×</button></div><input className="sr-search" autoFocus placeholder="Search player..." value={newPlayerQuery} onChange={e=>{setNewPlayerQuery(e.target.value);setSearch(e.target.value)}}/><div className="sr-new-player-results">{newPlayerMatches.map(p=>{const chosen=newPlayers.find(x=>x.id===p.id);return <div className={chosen?"selected":""} key={p.id} onClick={()=>setNewPlayers(chosen?newPlayers.filter(x=>x.id!==p.id):[...newPlayers,{...p,addSlot:(formationRows[current.formation]||flexRows).flat().map((x,i)=>x+"-"+i)[0]}])}><img src={shortlistPhoto(p.player)} alt=""/><span><strong>{p.player}</strong><small>{p.club||"Club not added"}</small></span>{chosen&&<select value={chosen.addSlot} onClick={e=>e.stopPropagation()} onChange={e=>setNewPlayers(newPlayers.map(x=>x.id===p.id?{...x,addSlot:e.target.value}:x))}>{(formationRows[current.formation]||flexRows).flat().map((x,i)=><option key={x+"-"+i} value={x+"-"+i}>{x}</option>)}</select>}</div>})}</div><div className="sr-actions"><button className="sr-outline" onClick={()=>setNewPlayerOpen(false)}>Cancel</button><button className="sr-cyan" onClick={addSelectedPlayers}>Add</button></div></section></div>;
  if (current)
    return (
      <main className="sr-page sr-shortlist-view">
        <button className="sr-back" onClick={() => setCurrent(null)}>
          ‹ Back to Your Shortlists
        </button>
        <section className="sr-dashboard-head">
          <div>
            <div className="sr-kicker">SHORTLIST</div>
            <h1>{current.name}</h1>
            {current.type === "Flex" ? <p>Flex · 14 positions</p> : <select className="sr-formation-select" value={current.formation} onChange={e=>changeFormation(e.target.value)}>{formations.map(f=><option key={f}>{f}</option>)}</select>}
          </div>
          <div className="sr-shortlist-head-actions"><button className="sr-delete-list" onClick={() => del(current.id)}>Delete Shortlist</button><div className="sr-shortlist-actions"><button className="sr-outline" onClick={()=>setManageTags(true)}>Add Tags</button><button className="sr-outline" onClick={()=>setPick("new-player")}>Add New Player</button></div></div>
        </section>
        <div className="sr-tag-legend">{tags.map(t=><span key={t.id}><i style={{background:t.color}}/>{t.name}</span>)}</div>
        <div className="sr-real-pitch">
          <div className="sr-goal-box top" />
          {(current.type === "Flex"
            ? flexRows
            : formationRows[current.formation] || formationRows["4-3-3"]
          ).map((r, i) => (
            <div
              className={"sr-pitch-row row-" + i}
              style={{
                zIndex: r.some((pos) => pick && pick.startsWith(pos + "-"))
                  ? 10
                  : 2,
              }}
              key={i}
            >
              {r.map((pos, j) => zone(pos + "-" + j))}
            </div>
          ))}
          <div className="sr-centre-circle" />
          <div className="sr-halfway-line" />
          <div className="sr-goal-box bottom" />
        </div>
        {tagModal}
        {tagManager}
        {newPlayerModal}
      </main>
    );
  return (
    <main className="sr-page">
      <section className="sr-dashboard-head">
        <div>
          <div className="sr-kicker">SCOUTING PLATFORM</div>
          <h1>Your Shortlists</h1>
          <p>Open a shortlist to view its formation pitch.</p>
        </div>
        <button className="sr-cyan" onClick={() => setShow(true)}>
          Create New Shortlist
        </button>
      </section>
      <section className="sr-list-grid">
        {lists.map((l) => (
          <button
            className="sr-list-card sr-list-card-button"
            key={l.id}
            onClick={() => setCurrent(l)}
          >
            <h2>{l.name}</h2>
            <p>{l.type === "Flex" ? "Flex formation" : l.formation}</p>
            <span>{l.players.length} players</span>
          </button>
        ))}
      </section>
      {!lists.length && (
        <div className="sr-empty">Create your first shortlist.</div>
      )}
      {show && (
        <div className="sr-modal">
          <form className="sr-form sr-shortlist-create" onSubmit={create}>
            <div className="sr-form-head">
              <div>
                <div className="sr-kicker">SHORTLIST MANAGEMENT</div>
                <h2>Create Shortlist</h2>
              </div>
              <button
                type="button"
                className="sr-close"
                onClick={() => setShow(false)}
              >
                ×
              </button>
            </div>
            <div className="sr-choice-grid sr-choice-small">
              <button
                type="button"
                className={type === "Flex" ? "active" : ""}
                onClick={() => setType("Flex")}
              >
                <h2>Flex</h2>
                <p>All 14 positions.</p>
              </button>
              <button
                type="button"
                className={type === "Formation" ? "active" : ""}
                onClick={() => setType("Formation")}
              >
                <h2>Formation</h2>
                <p>Regular 11 positions.</p>
                {type === "Formation" && (
                  <select
                    value={formation}
                    onChange={(e) => setFormation(e.target.value)}
                  >
                    {formations.map((f) => (
                      <option key={f}>{f}</option>
                    ))}
                  </select>
                )}
              </button>
            </div>
            <label className="sr-field">
              <span>Shortlist Name</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Shortlist name"
              />
            </label>
            <div className="sr-actions">
              <button
                type="button"
                className="sr-outline"
                onClick={() => setShow(false)}
              >
                Cancel
              </button>
              <button className="sr-cyan">Save</button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}
