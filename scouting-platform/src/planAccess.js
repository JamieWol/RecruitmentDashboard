const FULL_TOOL_PLANS = new Set(["independent_bundle", "club", "organisation"]);

export function normalizeWorkspaceKey(value) {
  return String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
}

export function hasFullToolAccess(profile, entitlement, workspaceEntitlement) {
  if (!profile) return false;
  if (profile.role === "platform_admin") return true;
  if (entitlement) return entitlement.status === "active" && FULL_TOOL_PLANS.has(entitlement.plan_code);
  if (workspaceEntitlement) return workspaceEntitlement.status === "active" && FULL_TOOL_PLANS.has(workspaceEntitlement.plan_code);
  if (profile.workspace_type === "club") return true;
  return false;
}
