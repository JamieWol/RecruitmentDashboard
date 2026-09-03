export function parsePlayerCsv(text) {
  const rows = [];
  let row = [],
    cell = "",
    quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i],
      n = text[i + 1];
    if (c === '"' && quoted && n === '"') {
      cell += '"';
      i++;
    } else if (c === '"') {
      quoted = !quoted;
    } else if (c === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((c === "\n" || c === "\r") && !quoted) {
      if (c === "\r" && n === "\n") i++;
      row.push(cell);
      if (row.some(Boolean)) rows.push(row);
      row = [];
      cell = "";
    } else cell += c;
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const headers = rows.shift() || [],
    index = Object.fromEntries(headers.map((h, i) => [h.trim(), i]));
  const get = (r, h) => String(r[index[h]] || "").trim();
  return rows
    .map((r) => ({
      id:
        get(r, "Player Id") ||
        get(r, "Player SBData Id") ||
        `${get(r, "Player Name")}-${get(r, "Team")}`,
      name: get(r, "Player Name") || get(r, "Name"),
      club: get(r, "Team"),
      position: get(r, "Primary Position"),
      secondaryPosition: get(r, "Secondary Position"),
      dob: get(r, "Date of Birth") || get(r, "Birth Date"),
      nationality: get(r, "Nationality") || get(r, "Country"),
      height: get(r, "Height"),
      weight: get(r, "Weight"),
      season: get(r, "Season"),
      competition: get(r, "Competition"),
    }))
    .filter((p) => p.name);
}
