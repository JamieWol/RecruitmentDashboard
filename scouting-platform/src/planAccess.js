const FULL_TOOL_PLANS = new Set(["independent_bundle", "club", "organisation"]);

export function hasFullToolAccess(profile, entitlement) {
  if (!profile) return false;
  if (profile.role === "platform_admin") return true;
  if (entitlement) return entitlement.status === "active" && FULL_TOOL_PLANS.has(entitlement.plan_code);
  if (profile.workspace_type === "club") return true;
  if (!profile.approved) return false;
  const requestedPlan = String(profile.requested_plan || "").toLowerCase();
  if (profile.workspace_type === "organization") return requestedPlan.includes("club workspace") || requestedPlan.includes("larger organisation");
  return profile.workspace_type === "independent" && requestedPlan.includes("full tools bundle");
}
