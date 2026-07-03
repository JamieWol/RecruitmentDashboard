import React, { useState } from "react";
import Papa from "papaparse";

/* ---------------- POSITION MAP ---------------- */
export const POSITION_MAP = {
  "Goalkeeper": { role: "GK", side: "C" },

  "Centre Back": { role: "CB", side: "C" },
  "Left Centre Back": { role: "CB", side: "L" },
  "Right Centre Back": { role: "CB", side: "R" },

  "Left Back": { role: "WB", side: "L" },
  "Right Back": { role: "WB", side: "R" },
  "Left Wing Back": { role: "WB", side: "L" },
  "Right Wing Back": { role: "WB", side: "R" },

  "Centre Defensive Midfielder": { role: "CM", side: "C" },
  "Left Defensive Midfielder": { role: "CM", side: "L" },
  "Right Defensive Midfielder": { role: "CM", side: "R" },
  "Left Centre Midfielder": { role: "CM", side: "L" },
  "Right Centre Midfielder": { role: "CM", side: "R" },

  "Centre Attacking Midfielder": { role: "10", side: "C" },
  "Left Attacking Midfielder": { role: "10", side: "L" },
  "Right Attacking Midfielder": { role: "10", side: "R" },
  "Left Midfielder": { role: "10", side: "L" },
  "Right Midfielder": { role: "10", side: "R" },
  "Left Wing": { role: "10", side: "L" },
  "Right Wing": { role: "10", side: "R" },

  "Centre Forward": { role: "CF", side: "C" },
  "Left Centre Forward": { role: "CF", side: "L" },
  "Right Centre Forward": { role: "CF", side: "R" },
  "Striker": { role: "CF", side: "C" },
};

/* ---------------- FORMATIONS ---------------- */
const FORMATIONS = {
  "3-4-3": [
    { id: "GK", base: "GK", side: "C" },

    { id: "LCB", base: "CB", side: "L" },
    { id: "CB", base: "CB", side: "C" },
    { id: "RCB", base: "CB", side: "R" },

    { id: "LWB", base: "WB", side: "L" },
    { id: "RCM", base: "CM", side: "R" },
    { id: "LCM", base: "CM", side: "L" },
    { id: "RWB", base: "WB", side: "R" },

    { id: "L10", base: "10", side: "L" },
    { id: "R10", base: "10", side: "R" },
    { id: "CF", base: "CF", side: "C" },
  ],
};

/* ---------------- COMPONENT ---------------- */
export default function ScoutDashboard({ shadowSquad, setShadowSquad }) {
  const [players, setPlayers] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState("");
  const [formation] = useState("3-4-3");

  /* ---------------- CSV UPLOAD ---------------- */
  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        setPlayers(res.data);
        setSelectedIndex("");
        console.log("CSV loaded:", res.data.length, "players");
      },
    });
  };

  /* ---------------- AUTO ASSIGN ---------------- */
  const autoAssignSlot = (playerMeta, formationSlots, currentSquad) => {
    const taken = currentSquad.map(p => p.position).filter(Boolean);

    let slot = formationSlots.find(
      s =>
        s.base === playerMeta.role &&
        s.side === playerMeta.side &&
        !taken.includes(s.id)
    );
    if (slot) return slot.id;

    slot = formationSlots.find(
      s =>
        s.base === playerMeta.role &&
        s.side === "C" &&
        !taken.includes(s.id)
    );
    if (slot) return slot.id;

    slot = formationSlots.find(
      s =>
        s.base === playerMeta.role &&
        !taken.includes(s.id)
    );
    if (slot) return slot.id;

    return null; // bench
  };

  /* ---------------- ADD TO SHADOW SQUAD ---------------- */
  const addToShadowSquad = () => {
    if (selectedIndex === "") return;

    const player = players[selectedIndex];
    if (!player || !player["Player Name"]) return;

    const mapping =
      POSITION_MAP[player["Primary Position"]] || { role: "CF", side: "C" };

    const formationSlots = FORMATIONS[formation];

    const assignedSlot = autoAssignSlot(
      mapping,
      formationSlots,
      shadowSquad
    );

    const playerToAdd = {
      id: player["Player Id"] || player["Player Name"],
      playerName: player["Player Name"],
      team: player["Team"] || "Unknown",

      role: mapping.role,
      side: mapping.side,
      position: assignedSlot,

      fullPosition: player["Primary Position"],
      raw: player,
    };

    setShadowSquad(prev => {
      if (prev.some(p => p.id === playerToAdd.id)) return prev;
      console.log("✅ Auto-assigned:", playerToAdd);
      return [...prev, playerToAdd];
    });
  };

  /* ---------------- UI ---------------- */
  return (
    <div style={{ padding: 24 }}>
      <h2>Scout Dashboard</h2>

      <input type="file" accept=".csv" onChange={handleUpload} />

      {players.length > 0 && (
        <>
          <select
            value={selectedIndex}
            onChange={(e) => setSelectedIndex(e.target.value)}
          >
            <option value="">Select Player</option>
            {players.map((p, i) => (
              <option key={i} value={i}>
                {p["Player Name"]}
              </option>
            ))}
          </select>

          <button onClick={addToShadowSquad}>
            ➕ Add to Shadow Squad
          </button>
        </>
      )}

      <ul>
        {shadowSquad.map(p => (
          <li key={p.id}>
            {p.playerName} — <strong>{p.position || "BENCH"}</strong>
          </li>
        ))}
      </ul>
    </div>
  );
}






