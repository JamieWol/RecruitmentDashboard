import { supabase } from "./supabaseClient";
import { normalizePlayerName } from "./playerProfileMatching";

export const createMasterPlayer = async (details) => {
  const name = String(details.name || "").trim();
  const { data: existingRows, error: lookupError } = await supabase
    .from("players")
    .select("*")
    .ilike("Name", name)
    .limit(20);
  if (lookupError) throw lookupError;

  const existing = (existingRows || []).find((row) =>
    normalizePlayerName(row.Name || row.name || row.player_name || row.player) === normalizePlayerName(name),
  );
  if (existing) return existing;

  const payload = {
    Name: name,
    Team: String(details.club || "").trim(),
    Position: String(details.position || "").trim(),
    DOB: String(details.dob || "").trim(),
    Nationality: String(details.nationality || "").trim(),
    "Dominant Foot": String(details.foot || "").trim(),
  };
  Object.keys(payload).forEach((key) => {
    if (!payload[key]) delete payload[key];
  });

  // Player sheets vary slightly in their column names. If PostgREST reports a
  // missing column, retry with its common aliases while retaining other fields.
  const aliases = {
    Team: ["team", "club", "Club"],
    Position: ["Playing Position", "Primary Position", "playing_position", "position"],
    DOB: ["Date of Birth", "date_of_birth", "dateOfBirth", "dob", "birth_date", "birthDate"],
    Nationality: ["nationality"],
    "Dominant Foot": ["Preferred Foot", "preferred_foot"],
  };
  const triedAliases = new Set();
  for (let attempt = 0; attempt < 12; attempt += 1) {
    const { data, error } = await supabase.from("players").insert(payload).select("*").single();
    if (!error) return data;
    if (error.code === "42501" || /row-level security/i.test(error.message || "")) {
      throw new Error("Supabase is blocking player inserts. Apply the supplied players RLS migration in the Supabase SQL Editor, then try again.");
    }
    if (!["PGRST204", "42703"].includes(error.code)) throw error;

    const errorMessage = String(error.message || "");
    const missingColumn = errorMessage.match(/column\s+['"]([^'"]+)['"]/i)?.[1]
      || errorMessage.match(/['"]([^'"]+)['"]\s+column/i)?.[1];
    const key = Object.keys(payload).find((candidate) => candidate.toLowerCase() === String(missingColumn || "").toLowerCase());
    if (!key || key.toLowerCase() === "name") throw error;
    const canonicalKey = Object.keys(aliases).find((candidate) => candidate.toLowerCase() === key.toLowerCase()) || key;
    const alias = (aliases[canonicalKey] || []).find((candidate) => !triedAliases.has(candidate.toLowerCase()));
    if (alias) {
      triedAliases.add(alias.toLowerCase());
      payload[alias] = payload[key];
    }
    delete payload[key];
  }
  throw new Error("Could not add this player using the available player database fields.");
};
