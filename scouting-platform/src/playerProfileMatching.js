export const normalizePlayerName = (value) => String(value || "")
  .normalize("NFD")
  .replace(/[\u0300-\u036f]/g, "")
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, " ")
  .trim()
  .replace(/\s+/g, " ");

const tokenMatches = (left, right) => left === right ||
  (left.length === 1 && right.startsWith(left)) ||
  (right.length === 1 && left.startsWith(right));

export const playerNameMatchScore = (reportedName, databaseName) => {
  const reported = normalizePlayerName(reportedName);
  const canonical = normalizePlayerName(databaseName);
  if (!reported || !canonical) return 0;
  if (reported === canonical) return 100;

  const queryTokens = reported.split(" ");
  const playerTokens = canonical.split(" ");
  let cursor = 0;
  for (const token of queryTokens) {
    let found = false;
    while (cursor < playerTokens.length) {
      if (tokenMatches(token, playerTokens[cursor])) {
        found = true;
        cursor += 1;
        break;
      }
      cursor += 1;
    }
    if (!found) return 0;
  }
  return queryTokens.length >= 2 ? 90 + queryTokens.length : 0;
};

const clubFor = (row) => row?.club || row?.Club || row?.team || row?.Team || row?.Squad || row?.squad || "";
const populatedFields = (row) => [
  row?.DOB, row?.dob, row?.["Date of Birth"], row?.date_of_birth, row?.Age, row?.age,
  row?.Nationality, row?.nationality, row?.Position, row?.["Primary Position"],
  row?.playing_position, row?.["Preferred Foot"], row?.preferred_foot,
  row?.Photo, row?.photo, clubFor(row),
].filter((value) => value !== null && value !== undefined && String(value).trim() !== "").length;

export const choosePlayerRecord = (rows, reportedName, expectedClub = "") => {
  const uniqueRows = [...new Map((rows || []).map((row) => [
    String(row?.id || `${row?.Name || row?.name || row?.player_name}:${clubFor(row)}:${row?.Photo || row?.photo || ""}`), row,
  ])).values()];
  const ranked = uniqueRows
    .map((row) => ({ row, score: playerNameMatchScore(reportedName, row?.Name || row?.name || row?.player_name || row?.player) }))
    .filter((candidate) => candidate.score > 0)
    .sort((left, right) => right.score - left.score);
  if (!ranked.length) return null;

  const bestScore = ranked[0].score;
  let candidates = ranked.filter((candidate) => candidate.score === bestScore);
  const expectedClubKey = normalizePlayerName(expectedClub);
  if (expectedClubKey) {
    const matchingClub = candidates.filter(({ row }) => normalizePlayerName(clubFor(row)) === expectedClubKey);
    if (matchingClub.length) candidates = matchingClub;
  }
  candidates.sort((left, right) => populatedFields(right.row) - populatedFields(left.row));

  if (candidates.length > 1) {
    const exactNameMatches = bestScore === 100;
    const sameClub = new Set(candidates.map(({ row }) => normalizePlayerName(clubFor(row)))).size === 1;
    const richerWinner = populatedFields(candidates[0].row) > populatedFields(candidates[1].row);
    if (exactNameMatches && !sameClub && !expectedClubKey) return null;
    if (!exactNameMatches && !sameClub && !richerWinner) return null;
  }

  // Duplicate database rows for the same player can split their bio fields.
  // Prefer the best-matched row, then fill only its missing values from exact
  // name duplicates with the same club so a complete record is displayed.
  const chosen = { ...candidates[0].row };
  const chosenClub = normalizePlayerName(clubFor(chosen));
  for (const candidate of candidates) {
    if (candidate.score !== 100) continue;
    const candidateClub = normalizePlayerName(clubFor(candidate.row));
    if (chosenClub && candidateClub && chosenClub !== candidateClub) continue;
    Object.entries(candidate.row).forEach(([key, value]) => {
      if ((chosen[key] === null || chosen[key] === undefined || String(chosen[key]).trim() === "") && value !== null && value !== undefined && String(value).trim() !== "") {
        chosen[key] = value;
      }
    });
  }
  return chosen;
};
