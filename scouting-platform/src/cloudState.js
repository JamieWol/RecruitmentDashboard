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
