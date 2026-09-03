import React, { useMemo, useState } from "react";
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
export default function ShortlistsPage() {
  const published = useMemo(
    () =>
      JSON.parse(localStorage.getItem("scoutingAssignments") || "[]").filter(
        (x) => x.status === "Published",
      ),
    [],
  );
  const playerPool = useMemo(() => {
    const database = JSON.parse(
      localStorage.getItem("scoutingPlayers") || "[]",
    ).map((p) => ({
      ...p,
      id: p.id || p.playerId || p.player_id || p.name,
      player:
        p.name ||
        [p.firstName || p.first_name, p.lastName || p.last_name]
          .filter(Boolean)
          .join(" "),
      club: p.club || p.team,
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
  }, [published]);
  const [lists, setLists] = useState(() =>
    JSON.parse(localStorage.getItem("scoutingShortlists") || "[]"),
  );
  const [current, setCurrent] = useState(null),
    [show, setShow] = useState(false),
    [name, setName] = useState(""),
    [type, setType] = useState("Formation"),
    [formation, setFormation] = useState("4-3-3"),
    [pick, setPick] = useState(null),
    [search, setSearch] = useState("");
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
            <div className="sr-pitch-player-card" key={p.id}>
              <button onClick={() => window.alert(p.player)}>
                <strong>{p.player}</strong>
                <small>{p.club || "Club not added"}</small>
              </button>
              <button onClick={() => remove(p.id, pos)}>×</button>
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
            {matches.map((p) => (
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
            <p>
              {current.type === "Flex"
                ? "Flex · 14 positions"
                : current.formation}
            </p>
          </div>
          <button className="sr-delete-list" onClick={() => del(current.id)}>
            Delete Shortlist
          </button>
        </section>
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
