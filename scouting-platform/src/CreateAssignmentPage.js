import { ClubName } from "./ClubBadge";
import React, { useMemo, useState } from "react";
import { useEffect } from "react";
import { supabase } from "./supabaseClient";
import { useAuth } from "./AuthContext";
import { createMasterPlayer } from "./playerDatabase";

export default function CreateAssignmentPage() {
  const { appState, updateAppState, profile, user } = useAuth();
  const [databasePlayers, setDatabasePlayers] = useState([]);
  const [scouts, setScouts] = useState([]);
  const [mode, setMode] = useState(() => localStorage.getItem("editingAssignment") ? "player" : ""),
    [query, setQuery] = useState(""),
    [selected, setSelected] = useState(""),
    [selectedPlayers, setSelectedPlayers] = useState([]),
    [newPlayer, setNewPlayer] = useState(false);
  useEffect(() => {
    if (!profile?.club) return;
    supabase.from("profiles").select("id,full_name,club").eq("club", profile.club).eq("approved", true).then(({ data }) => setScouts(data || []));
  }, [profile?.club]);
  useEffect(() => {
    if (!query.trim()) {
      setDatabasePlayers([]);
      return;
    }
    supabase
      .from("players")
      .select("*")
      .ilike("Name", `%${query.trim()}%`)
      .limit(50)
      .then(({ data, error }) => {
        if (error) console.error("Player database error", error);
        if (data) setDatabasePlayers([...new Map(data.map((p) => [String(p.Name || p.name || "").trim().toLowerCase(), p])).values()]);
      });
  }, [query]);
  const records = useMemo(
    () => appState?.assignments || JSON.parse(localStorage.getItem("scoutingAssignments") || "[]"),
    [appState?.assignments],
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
    // Keep the scout attached to imported reports when their fixtures are edited.
    // Legacy assignments without an assignee default to the signed-in editor.
    setScout(editing.scoutId || user?.id || "");
    setDate(editing.date || "");
    setViewing(editing.viewing || "Live");
  }, [user?.id]);
  const matches = players.filter((p) =>
    p.name.toLowerCase().includes(query.toLowerCase()) &&
    (mode !== "fixture" || !selectedPlayers.some((chosen) => String(chosen.id || chosen.name).toLowerCase() === String(p.id || p.name).toLowerCase())),
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
  const save = async (e) => {
    e.preventDefault();
    const player = selected || details.name;
    const fixtures = games.length
      ? games
      : game.trim() && gameDate
        ? [{ name: game.trim(), date: gameDate }]
        : [];
    if (mode === "fixture" && !selectedPlayers.length)
      return setError("Select at least one player for this fixture.");
    if (mode !== "fixture" && !player)
      return setError("Select an existing player or create a new player.");
    if (!fixtures.length) return setError("Add at least one fixture / game.");
    let chosen = players.find((p) => p.name === selected);
    if (newPlayer) {
      try {
        const savedPlayer = await createMasterPlayer(details);
        chosen = {
          ...savedPlayer,
          name: savedPlayer.Name || savedPlayer.name || player,
          club: savedPlayer.Team || savedPlayer.team || savedPlayer.Club || savedPlayer.club || details.club,
          position: savedPlayer.Position || savedPlayer.position || savedPlayer["Primary Position"] || details.position,
        };
      } catch (savePlayerError) {
        console.error("Could not add player to the master player database", savePlayerError);
        setError(`Could not add this player to the master database: ${savePlayerError.message || "Please check your Supabase permissions and try again."}`);
        return;
      }
    }
    const old = appState?.assignments || JSON.parse(localStorage.getItem("scoutingAssignments") || "[]");
    const editing = JSON.parse(
      localStorage.getItem("editingAssignment") || "null",
    );
    const existing = editing
      ? old.find((x) => String(x.id) === String(editing.id))
      : null;
    const selectedScout = scouts.find((member) => String(member.id) === String(scout));
    const currentScoutName = profile?.full_name || user?.user_metadata?.full_name || user?.email || "";
    const assignment = {
      id: existing?.id || editing?.id || Date.now(),
      player,
      playerId: chosen?.id || chosen?.player_id || chosen?.["Player Id"] || null,
      club: details.club || chosen?.club || "",
      position: details.position || chosen?.position || "",
      games: fixtures,
      game: fixtures
        .map((x) => (typeof x === "string" ? x : x.name))
        .join(" • "),
      fixtureDates: fixtures.map((x) => (typeof x === "string" ? "" : x.date)),
      scout: selectedScout?.full_name || (String(scout) === String(user?.id) ? currentScoutName : scout),
      scoutId: scout,
      date,
      viewing,
      status: "Not Started",
      report: existing?.report || editing?.report || null,
    };
    let nextAssignments;
    if (mode === "fixture") {
      const groupId = String(Date.now());
      const groupPlayers = selectedPlayers.map((person, index) => ({
        assignmentId: Number(groupId) + index,
        player: person.name,
        playerId: person.id || person.player_id || person["Player Id"] || null,
        club: person.club || "",
        position: person.position || "",
      }));
      nextAssignments = [
        ...old,
        ...groupPlayers.map((person) => ({
          ...assignment,
          id: person.assignmentId,
          player: person.player,
          playerId: person.playerId,
          club: person.club,
          position: person.position,
          report: null,
          fixtureGroupId: groupId,
          fixtureGroupFixture: fixtures.map((fixture) => typeof fixture === "string" ? fixture : fixture.name).join(" • "),
          fixtureGroupPlayers: groupPlayers,
        })),
      ];
    } else {
      nextAssignments = editing
        ? old
          .map((x) => String(x.id) === String(editing.id) ? { ...x, ...assignment, id: x.id } : x)
          .filter((x) => String(x.id) === String(editing.id) || String(x.player || "").trim().toLowerCase() !== String(player).trim().toLowerCase())
        : [...old, assignment];
    }
    updateAppState({ ...appState, assignments: nextAssignments });
    if (profile?.club && scout) {
      const createdBy = (await supabase.auth.getUser()).data.user.id;
      const assignmentsToSave = mode === "fixture" ? nextAssignments.slice(old.length) : [assignment];
      await Promise.all(assignmentsToSave.map((item) => supabase.from("club_assignments").upsert({
        id: Number(item.id), club: profile.club, assigned_to: scout, created_by: createdBy,
        assignment: { ...item, id: Number(item.id) }, status: item.status,
      }, { onConflict: "id" })));
    }
    localStorage.removeItem("editingAssignment");
    window.history.back();
  };
  return (
    <main className="sr-page sr-wizard">
      <button className="sr-back" onClick={() => window.history.back()}>
        ‹ Back to Reports & Assignments
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
              if (mode === "fixture") setNewPlayer(false);
              else {
                setSelected("");
                setNewPlayer(false);
              }
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
                    if (mode === "fixture") {
                      setSelectedPlayers((current) => current.some((person) => String(person.id || person.name).toLowerCase() === String(p.id || p.name).toLowerCase()) ? current : [...current, p]);
                      setQuery("");
                    } else {
                      setSelected(p.name);
                      setQuery(p.name);
                    }
                  }}
                >
                  <>
                    {p.name}
                    <small>
                      <ClubName club={p.club} size={16} /> ·{" "}
                      {p.position || "Position not added"}
                    </small>
                  </>
                </button>
              ))}
              {!matches.length && mode !== "fixture" && (
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
          {mode === "fixture" && (
            <div className="sr-fixture-selected-players">
              <strong>Players selected for this fixture ({selectedPlayers.length})</strong>
              {selectedPlayers.length ? selectedPlayers.map((person) => (
                <div className="sr-selected-player" key={person.id || person.name}>
                  <strong>{person.name}</strong>
                  <span><ClubName club={person.club} size={16} /> · {person.position || "Position not added"}</span>
                  <button type="button" aria-label={`Remove ${person.name}`} onClick={() => setSelectedPlayers((current) => current.filter((chosen) => String(chosen.id || chosen.name) !== String(person.id || person.name)))}>×</button>
                </div>
              )) : <p className="sr-muted">Search and select each player you want to report on from this fixture.</p>}
            </div>
          )}
          {mode !== "fixture" && selected && (
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
              <select value={scout} onChange={(e) => setScout(e.target.value)}>
                <option value="">Select scout</option>
                {scouts.map((member) => <option key={member.id} value={member.id}>{member.full_name || member.id}</option>)}
              </select>
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
