import React, { useMemo, useState } from "react";
import { useEffect } from "react";
import { supabase } from "./supabaseClient";
export default function CreateAssignmentPage() {
  const [databasePlayers, setDatabasePlayers] = useState([]);
  const [mode, setMode] = useState(() => localStorage.getItem("editingAssignment") ? "player" : ""),
    [query, setQuery] = useState(""),
    [selected, setSelected] = useState(""),
    [newPlayer, setNewPlayer] = useState(false);
  useEffect(() => {
    if (!query.trim()) {
      setDatabasePlayers([]);
      return;
    }
    supabase
      .from("players")
      .select("*")
      .ilike("Name", query.trim())
      .limit(50)
      .then(({ data, error }) => {
        if (error) console.error("Player database error", error);
        if (data) setDatabasePlayers(data);
      });
  }, [query]);
  const records = useMemo(
    () => JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"),
    [],
  );
  const imported = databasePlayers.map((x) => ({
    ...x,
    name:
      x.Name ||
      x.name ||
      [x.firstName || x.first_name, x.lastName || x.last_name]
        .filter(Boolean)
        .join(" "),
    club: x.club || x.team || x.Team,
    position: x.position || x["Primary Position"],
    id: x.id || x["Player Id"] || x.Name,
  }));
  const players = [
    ...new Map([
      ...imported.map((x) => [x.name, x]),
      ...records.map((x) => [
        x.player,
        { name: x.player, club: x.club, position: x.position },
      ]),
    ]).values(),
  ];
  const [details, setDetails] = useState({
      name: "",
      club: "",
      position: "",
      dob: "",
      nationality: "",
      foot: "",
    }),
    [game, setGame] = useState(""),
    [games, setGames] = useState([]),
    [gameDate, setGameDate] = useState(""),
    [scout, setScout] = useState(""),
    [date, setDate] = useState(""),
    [viewing, setViewing] = useState("Live"),
    [error, setError] = useState("");
  useEffect(() => {
    const editing = JSON.parse(
      localStorage.getItem("editingAssignment") || "null",
    );
    if (!editing) return;
    setSelected(editing.player);
    setQuery(editing.player);
    setDetails({
      name: editing.player,
      club: editing.club || "",
      position: editing.position || "",
      dob: "",
      nationality: "",
      foot: "",
    });
    setGames(
      editing.games || [
        { name: editing.game || "", date: editing.fixtureDates?.[0] || "" },
      ],
    );
    setDate(editing.date || "");
    setViewing(editing.viewing || "Live");
  }, []);
  const matches = players.filter((p) =>
    p.name.toLowerCase().includes(query.toLowerCase()),
  );
  const addGame = () => {
    if (!game.trim() || !gameDate) {
      setError("Enter a fixture before adding it.");
      return;
    }
    setGames([...games, { name: game.trim(), date: gameDate }]);
    setGame("");
    setGameDate("");
    setError("");
  };
  const save = (e) => {
    e.preventDefault();
    const player = selected || details.name;
    const fixtures = games.length
      ? games
      : game.trim() && gameDate
        ? [{ name: game.trim(), date: gameDate }]
        : [];
    if (!player)
      return setError("Select an existing player or create a new player.");
    if (!fixtures.length) return setError("Add at least one fixture / game.");
    const chosen = players.find((p) => p.name === selected);
    const old = JSON.parse(localStorage.getItem("scoutingAssignments") || "[]");
    const assignment = {
      id: Date.now(),
      player,
      playerId: chosen?.id || null,
      club: details.club || chosen?.club || "",
      position: details.position || chosen?.position || "",
      games: fixtures,
      game: fixtures
        .map((x) => (typeof x === "string" ? x : x.name))
        .join(" • "),
      fixtureDates: fixtures.map((x) => (typeof x === "string" ? "" : x.date)),
      scout,
      date,
      viewing,
      status: "Not Started",
      report: null,
    };
    const editing = JSON.parse(
      localStorage.getItem("editingAssignment") || "null",
    );
    localStorage.setItem(
      "scoutingAssignments",
      JSON.stringify(
        editing
          ? old.map((x) =>
              x.id === editing.id ? { ...assignment, id: editing.id } : x,
            )
          : [...old, assignment],
      ),
    );
    localStorage.removeItem("editingAssignment");
    window.history.back();
  };
  return (
    <main className="sr-page sr-wizard">
      <button className="sr-back" onClick={() => window.history.back()}>
        ‹ Back to Scouting Reports
      </button>
      <div className="sr-wizard-title">
        <div className="sr-kicker">ASSIGNMENT MANAGEMENT</div>
        <h1>Create Assignment</h1>
      </div>
      {!mode ? (
        <div className="sr-choice-grid">
          <button onClick={() => setMode("player")}>
            <h2>By Player</h2>
            <p>Select multiple fixtures and report on one player.</p>
          </button>
          <button onClick={() => setMode("fixture")}>
            <h2>By Fixture</h2>
            <p>Select one fixture and report on multiple players.</p>
          </button>
        </div>
      ) : (
        <form className="sr-form sr-wizard-form" onSubmit={save}>
          <button type="button" className="sr-back" onClick={() => setMode("")}>
            ‹ Change assignment type
          </button>
          <h2>
            {mode === "player" ? "Assign by Player" : "Assign by Fixture"}
          </h2>
          <label className="sr-field">
            <span>Search Player</span>
            <input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setSelected("");
                setNewPlayer(false);
              }}
              placeholder="Type player name..."
            />
          </label>
          {query && !newPlayer && (
            <div className="sr-player-results">
              {matches.map((p) => (
                <button
                  type="button"
                  key={p.id || p.name}
                  onClick={() => {
                    setSelected(p.name);
                    setQuery(p.name);
                  }}
                >
                  <>
                    {p.name}
                    <small>
                      {p.club || "Club not added"} ·{" "}
                      {p.position || "Position not added"}
                    </small>
                  </>
                </button>
              ))}
              {!matches.length && (
                <button
                  type="button"
                  onClick={() => {
                    setNewPlayer(true);
                    setDetails({ ...details, name: query });
                  }}
                >
                  + Create New Player
                </button>
              )}
            </div>
          )}
          {selected && (
            <div className="sr-selected-player">
              <strong>{selected}</strong>
              <span>Existing player selected</span>
            </div>
          )}
          {newPlayer && (
            <div className="sr-form-grid">
              {[
                ["Full name", "name"],
                ["Club", "club"],
                ["Position", "position"],
                ["Date of birth", "dob"],
                ["Nationality", "nationality"],
              ].map(([l, k]) => (
                <label className="sr-field" key={k}>
                  <span>{l}</span>
                  <input
                    value={details[k]}
                    onChange={(e) =>
                      setDetails({ ...details, [k]: e.target.value })
                    }
                  />
                </label>
              ))}
              <label className="sr-field">
                <span>Preferred foot</span>
                <select
                  value={details.foot}
                  onChange={(e) =>
                    setDetails({ ...details, foot: e.target.value })
                  }
                >
                  <option value="">Select</option>
                  <option>Right</option>
                  <option>Left</option>
                  <option>Both</option>
                </select>
              </label>
            </div>
          )}
          <label className="sr-field">
            <span>Add Fixtures / Games</span>
            <div className="sr-inline">
              <input
                value={game}
                onChange={(e) => setGame(e.target.value)}
                placeholder="Home v Away"
              />
              <input
                type="date"
                value={gameDate}
                onChange={(e) => setGameDate(e.target.value)}
                aria-label="Fixture date"
              />
              <button type="button" className="sr-outline" onClick={addGame}>
                Add Game
              </button>
            </div>
          </label>
          {games.map((fixture, i) => (
            <div className="sr-game-list" key={fixture + i}>
              <div>
                {typeof fixture === "string"
                  ? fixture
                  : `${fixture.name} · ${fixture.date}`}
                <button
                  type="button"
                  onClick={() => setGames(games.filter((_, n) => n !== i))}
                >
                  ×
                </button>
              </div>
            </div>
          ))}
          <div className="sr-form-grid">
            <label className="sr-field">
              <span>Assign scout</span>
              <input value={scout} onChange={(e) => setScout(e.target.value)} />
            </label>
            <label className="sr-field">
              <span>Due date</span>
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
              />
            </label>
            <label className="sr-field">
              <span>Viewing type</span>
              <select
                value={viewing}
                onChange={(e) => setViewing(e.target.value)}
              >
                <option>Live</option>
                <option>Video</option>
              </select>
            </label>
          </div>
          {error && <p className="sr-error">{error}</p>}
          <button className="sr-cyan">Save Assignment</button>
        </form>
      )}
    </main>
  );
}
