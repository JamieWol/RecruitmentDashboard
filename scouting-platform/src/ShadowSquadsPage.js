import React, { useEffect, useState } from "react";

/* ---------------- FORMATIONS ---------------- */
const FORMATIONS = {
  "3-4-3": [
    { id: "GK",  label: "GK",  base: "GK", side: "C", x: 50, y: 92 },

    { id: "LCB", label: "LCB", base: "CB", side: "L", x: 30, y: 72 },
    { id: "CB",  label: "CB",  base: "CB", side: "C", x: 50, y: 72 },
    { id: "RCB", label: "RCB", base: "CB", side: "R", x: 70, y: 72 },

    { id: "LWB", label: "LWB", base: "WB", side: "L", x: 18, y: 52 },
    { id: "LCM", label: "L6",  base: "CM", side: "L", x: 40, y: 52 },
    { id: "RCM", label: "R6",  base: "CM", side: "R", x: 60, y: 52 },
    { id: "RWB", label: "RWB", base: "WB", side: "R", x: 82, y: 52 },

    { id: "L10", label: "L10", base: "10", side: "L", x: 28, y: 28 },
    { id: "CF",  label: "CF",  base: "CF", side: "C", x: 50, y: 22 },
    { id: "R10", label: "R10", base: "10", side: "R", x: 72, y: 28 },
  , positions],
};



/* ---------------- PRIORITY COLOURS ---------------- */
const STATUS_COLOURS = {
  neutral: "#eeeeee",
  priority: "#66bb6a",
  maybe: "#ffeb3b",
  monitor: "#ef5350",
};

export default function ShadowSquadsPage({ shadowSquad, setShadowSquad }) {
  const [formation, setFormation] = useState("3-4-3");
  const [assignments, setAssignments] = useState({});
  const [statuses, setStatuses] = useState({});

  const positions = FORMATIONS[formation];

  /* ---------------- AUTO ASSIGN ---------------- */
    useEffect(() => {
      const autoAssignSlot = (player, slots, taken) => {
        // 1️⃣ exact role + side
        let slot = slots.find(
          s =>
            s.base === player.role &&
            s.side === player.side &&
            !taken.includes(s.id)
        );
        if (slot) return slot.id;

        // 2️⃣ same role, centre
        slot = slots.find(
          s =>
            s.base === player.role &&
            s.side === "C" &&
            !taken.includes(s.id)
        );
        if (slot) return slot.id;

        // 3️⃣ same role, any side
        slot = slots.find(
          s =>
            s.base === player.role &&
            !taken.includes(s.id)
        );
        if (slot) return slot.id;

        return null; // bench
      };

      setAssignments(() => {
        const next = {};
        const taken = [];

        shadowSquad.forEach(player => {
          const slotId = autoAssignSlot(player, positions, taken);
          if (slotId) {
            next[player.id] = slotId;
            taken.push(slotId);
          }
        });

        return next;
      });
    }, [shadowSquad, formation]);


  /* ---------------- HELPERS ---------------- */
  const assignedIds = Object.keys(assignments);

  const unassignedPlayers = shadowSquad.filter(
    (p) => !assignedIds.includes(p.id)
  );

  const cycleStatus = (playerId) => {
    const order = ["neutral", "priority", "maybe", "monitor"];
    const current = statuses[playerId] || "neutral";
    const next = order[(order.indexOf(current) + 1) % order.length];
    setStatuses((p) => ({ ...p, [playerId]: next }));
  };

  const removeFromPitch = (playerId) => {
    setAssignments((p) => {
      const copy = { ...p };
      delete copy[playerId];
      return copy;
    });
  };

  const removeFromShadowSquad = (playerId) => {
    setShadowSquad((prev) => prev.filter((p) => p.id !== playerId));
    removeFromPitch(playerId);
    setStatuses((s) => {
      const copy = { ...s };
      delete copy[playerId];
      return copy;
    });
  };

  /* ---------------- SAVE / LOAD ---------------- */
  const saveSquad = () => {
    localStorage.setItem(
      "shadowSquadData",
      JSON.stringify({ shadowSquad, assignments, statuses, formation })
    );
    alert("Shadow Squad saved");
  };

  const loadSquad = () => {
    const saved = localStorage.getItem("shadowSquadData");
    if (!saved) return alert("No saved squad found");

    const data = JSON.parse(saved);
    setShadowSquad(data.shadowSquad);
    setAssignments(data.assignments);
    setStatuses(data.statuses);
    setFormation(data.formation);
  };

  /* ---------------- EXPORT ---------------- */
  const exportJSON = () => {
    const blob = new Blob(
      [JSON.stringify({ shadowSquad, assignments, statuses }, null, 2)],
      { type: "application/json" }
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "shadow-squad.json";
    a.click();
  };

  /* ---------------- UI ---------------- */
  return (
    <div style={{ padding: 20 }}>
      <h2>Shadow Squad</h2>

      <div style={{ marginBottom: 12 }}>
        <select value={formation} onChange={(e) => setFormation(e.target.value)}>
          <option value="3-4-3">3-4-3</option>
        </select>

        <button onClick={saveSquad} style={btn}>💾 Save</button>
        <button onClick={loadSquad} style={btn}>📂 Load</button>
        <button onClick={exportJSON} style={btn}>📤 Export</button>
      </div>

      <div style={{ display: "flex", gap: 20 }}>
        {/* BENCH */}
        <div style={bench}>
          <h4>Unassigned</h4>

          {unassignedPlayers.map((p) => (
            <div
              key={p.id}
              onDoubleClick={() => cycleStatus(p.id)}
              style={{
                ...card,
                background: STATUS_COLOURS[statuses[p.id] || "neutral"],
              }}
            >
              <a href={p.reportUrl} target="_blank" rel="noreferrer">
                {p.playerName}
              </a>
              <div style={team}>{p.team}</div>

              <button
                style={removeAllBtn}
                onClick={() => removeFromShadowSquad(p.id)}
              >
                🗑
              </button>
            </div>
          ))}
        </div>

        {/* PITCH */}
        <div style={pitch}>
          <div style={halfway} />
          <div style={circle} />
          <div style={boxTop} />
          <div style={boxBottom} />
          <div style={sixTop} />
          <div style={sixBottom} />
          <div style={goalTop} />
          <div style={goalBottom} />

          {positions.map((pos) => (
            <div
              key={pos.id}
              style={{
                position: "absolute",
                left: `${pos.x}%`,
                top: `${pos.y}%`,
                transform: "translate(-50%, -50%)",
                width: 180,
                textAlign: "center",
              }}
            >
              <div style={circlePos}>{pos.label}</div>

              {shadowSquad
                .filter((p) => assignments[p.id] === pos.id)
                .map((p) => (
                  <div
                    key={p.id}
                    onDoubleClick={() => cycleStatus(p.id)}
                    style={{
                      ...card,
                      background:
                        STATUS_COLOURS[statuses[p.id] || "neutral"],
                    }}
                  >
                    <a href={p.reportUrl} target="_blank" rel="noreferrer">
                      {p.playerName}
                    </a>
                    <div style={team}>{p.team}</div>

                    <button
                      style={removeAllBtn}
                      onClick={() => removeFromShadowSquad(p.id)}
                    >
                      🗑
                    </button>
                  </div>
                ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ---------------- STYLES ---------------- */

const btn = { marginLeft: 8, padding: "6px 10px", fontWeight: 700 };

const pitch = {
  position: "relative",
  width: 700,
  height: 980,
  background:
    "repeating-linear-gradient(0deg,#dfeee0,#dfeee0 40px,#d7e8d8 40px,#d7e8d8 80px)",
  border: "3px solid #fff",
  borderRadius: 10,
};

const halfway = { position: "absolute", top: "50%", width: "100%", height: 2, background: "#fff" };
const circle = { position: "absolute", top: "50%", left: "50%", width: 140, height: 140, border: "2px solid #fff", borderRadius: "50%", transform: "translate(-50%,-50%)" };
const boxTop = { position: "absolute", top: 0, left: "25%", width: "50%", height: 150, border: "2px solid #fff" };
const boxBottom = { position: "absolute", bottom: 0, left: "25%", width: "50%", height: 150, border: "2px solid #fff" };
const sixTop = { position: "absolute", top: 0, left: "40%", width: "20%", height: 70, border: "2px solid #fff" };
const sixBottom = { position: "absolute", bottom: 0, left: "40%", width: "20%", height: 70, border: "2px solid #fff" };
const goalTop = { position: "absolute", top: -16, left: "44%", width: "12%", height: 16, border: "2px solid #fff" };
const goalBottom = { position: "absolute", bottom: -16, left: "44%", width: "12%", height: 16, border: "2px solid #fff" };

const circlePos = {
  width: 40,
  height: 40,
  borderRadius: "50%",
  background: "#fff",
  fontWeight: 800,
  margin: "0 auto 6px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};

const bench = { width: 240, background: "#f4f4f4", padding: 10 };

const card = {
  padding: 6,
  marginBottom: 6,
  borderRadius: 4,
  position: "relative",
};

const team = { fontSize: 11, opacity: 0.7 };

const removeAllBtn = {
  position: "absolute",
  top: 4,
  right: 4,
  border: "none",
  background: "transparent",
  cursor: "pointer",
};
