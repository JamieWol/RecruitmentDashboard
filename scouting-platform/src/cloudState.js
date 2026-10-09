import { supabase } from "./supabaseClient";

const keys = ["scoutingAssignments", "scoutingShortlists", "scoutingTags"];
const legacySquadKeys = ["squadPlans", "squadPlanClub", "squadPlanFormation", "squadPlanPlayers", "squadPlanTags", "squadPlanRemoved", "squadPlanPositionLabels", "squadPlanActiveId"];
const readLocal = (key, fallback) => {
  try { return JSON.parse(localStorage.getItem(key) || JSON.stringify(fallback)); } catch { return fallback; }
};

export async function migrateAndLoadState(user) {
  if (!user) return null;
  const local = {
    assignments: readLocal("scoutingAssignments", []),
    shortlists: readLocal("scoutingShortlists", []),
    tags: readLocal("scoutingTags", []),
    squadPlan: {
      club: localStorage.getItem("squadPlanClub") || "",
      formation: localStorage.getItem("squadPlanFormation") || "4-2-3-1",
      players: readLocal("squadPlanPlayers", []),
      tags: readLocal("squadPlanTags", []),
      removedIds: readLocal("squadPlanRemoved", {}),
      positionLabels: readLocal("squadPlanPositionLabels", {}),
      savedPlans: readLocal("squadPlans", []),
    },
  };
  const { data: existing, error: readError } = await supabase.from("user_app_state").select("*").eq("user_id", user.id).maybeSingle();
  if (readError) throw readError;
  const hasLocalData = local.assignments.length || local.shortlists.length || local.tags.length;
  let state = existing || {
    user_id: user.id,
    assignments: local.assignments,
    shortlists: local.shortlists,
    tags: local.tags,
    squadPlan: { savedPlans: [] },
  };
  if (!existing && hasLocalData) {
    const { error } = await supabase.from("user_app_state").upsert(state, { onConflict: "user_id" });
    if (error) throw error;
    keys.forEach((key) => localStorage.removeItem(key));
  } else if (existing?.squadPlan && local.squadPlan.savedPlans.length) {
    // Saved squad plans used to live in a browser-wide key. Attach only plans that
    // match this account's already-cloud-saved active plan, then discard the legacy key.
    const planPlayerIds = (plan) => (plan?.players || []).map((player) => String(player.id || player.playerId || player.player_id || player.player || player.Name || player.name || "")).filter(Boolean).sort();
    const activeIds = planPlayerIds(existing.squadPlan);
    const matchingLegacyPlans = local.squadPlan.savedPlans.filter((plan) => {
      const sameClub = String(plan.club || "").trim().toLowerCase() === String(existing.squadPlan.club || "").trim().toLowerCase();
      const ids = planPlayerIds(plan);
      return sameClub && plan.formation === existing.squadPlan.formation && ids.length === activeIds.length && ids.every((id, index) => id === activeIds[index]);
    });
    if (matchingLegacyPlans.length) {
      state = { ...existing, squadPlan: { ...existing.squadPlan, savedPlans: [...(existing.squadPlan.savedPlans || []), ...matchingLegacyPlans.filter((plan) => !(existing.squadPlan.savedPlans || []).some((saved) => String(saved.id) === String(plan.id)))] } };
      const { error } = await supabase.from("user_app_state").upsert(state, { onConflict: "user_id" });
      if (error) throw error;
    }
  }
  // Squad plan draft/list data used to be shared by every account on this device.
  // Remove those browser-wide keys so another account cannot inherit them.
  legacySquadKeys.forEach((key) => localStorage.removeItem(key));
  return {
    assignments: state.assignments || [],
    shortlists: state.shortlists || [],
    tags: state.tags || [],
    squadPlan: state.squadPlan || { savedPlans: [] },
    playerSummaries: state.playerSummaries || {},
  };
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
