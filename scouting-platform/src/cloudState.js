import { supabase } from "./supabaseClient";

const keys = ["scoutingAssignments", "scoutingShortlists", "scoutingTags"];
const readLocal = (key, fallback) => {
  try { return JSON.parse(localStorage.getItem(key) || JSON.stringify(fallback)); } catch { return fallback; }
};

export async function migrateAndLoadState(user) {
  if (!user) return null;
  const local = {
    assignments: readLocal("scoutingAssignments", []),
    shortlists: readLocal("scoutingShortlists", []),
    tags: readLocal("scoutingTags", []),
  };
  const { data: existing, error: readError } = await supabase.from("user_app_state").select("*").eq("user_id", user.id).maybeSingle();
  if (readError) throw readError;
  const state = existing || { user_id: user.id, assignments: local.assignments, shortlists: local.shortlists, tags: local.tags };
  if (!existing && (local.assignments.length || local.shortlists.length || local.tags.length)) {
    const { error } = await supabase.from("user_app_state").upsert(state, { onConflict: "user_id" });
    if (error) throw error;
    keys.forEach((key) => localStorage.removeItem(key));
  }
  return { assignments: state.assignments || [], shortlists: state.shortlists || [], tags: state.tags || [] };
}

export async function saveCloudState(user, state) {
  if (!user) return;
  const { error } = await supabase.from("user_app_state").upsert({ user_id: user.id, ...state, updated_at: new Date().toISOString() }, { onConflict: "user_id" });
  if (error) throw error;
}

export async function syncPublishedReports(user, state, club) {
  if (!user || !club) return;
  const reports = (state?.assignments || []).filter((x) => ["Published", "Complete"].includes(x.status) && x.report).map((x) => ({
    id: Number(x.id), assignment_id: Number(x.id), player_id: x.playerId || null,
    player: x.player, club, player_club: x.club || "", author_id: user.id, report: x.report, status: "Published", completed_at: x.completedAt || x.date || new Date().toISOString().slice(0, 10), scout: x.scout || "", fixture_summary: (x.games || []).length > 1 ? "Multiple" : (x.games?.[0] ? `${x.games[0].date || ""} · ${x.games[0].name || x.games[0]}` : x.game || ""),
  }));
  if (reports.length) {
    const { error } = await supabase.from("club_reports").upsert(reports, { onConflict: "id" });
    if (error) console.error("Could not migrate published reports", error);
  }
}
