import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { supabase } from "./supabaseClient";
import { migrateAndLoadState, saveCloudState, syncPublishedReports } from "./cloudState";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  const [profile, setProfile] = useState(null);
  const [entitlement, setEntitlement] = useState(null);
  const [loading, setLoading] = useState(true);
  const [appState, setAppState] = useState(null);
  const [profileError, setProfileError] = useState("");
  const cloudSaveQueue = useRef(Promise.resolve());
  const loadProfile = async (user) => {
    if (!user) { setProfile(null); setEntitlement(null); setAppState(null); setProfileError(""); return; }
    const { data, error } = await supabase.from("profiles").select("*").eq("id", user.id).maybeSingle();
    if (error) { console.error("Could not load account profile", error); setProfile(null); setProfileError(error.message || String(error)); return; }
    if (!data) { setProfile(null); setProfileError("No profile row was returned for this user."); return; }
    setProfileError("");
    setProfile(data || null);
    const { data: planData, error: planError } = await supabase.from("account_entitlements")
      .select("plan_code,status,trial_started_at,trial_ends_at,trial_report_limit,published_reports_used")
      .eq("user_id", user.id).maybeSingle();
    if (planError) {
      if (!/account_entitlements|schema cache|does not exist/i.test(planError.message || "")) console.error("Could not load plan access", planError);
      setEntitlement(null);
    } else setEntitlement(planData || null);
    if (data?.approved) {
      try { const state = await migrateAndLoadState(user); setAppState(state); await syncPublishedReports(user, state, data.club); } catch (error) { console.error("Could not load cloud data", error); }
    }
  };
  useEffect(() => {
    supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data: listener } = supabase.auth.onAuthStateChange((_event, next) => setSession(next));
    return () => listener.subscription.unsubscribe();
  }, []);
  useEffect(() => {
    if (!session?.user) { setProfile(null); setEntitlement(null); setAppState(null); setLoading(false); return; }
    setLoading(true);
    loadProfile(session.user).finally(() => setLoading(false));
  }, [session]);
  const updateAppState = (next) => {
    setAppState(next);
    const saveUser = session?.user;
    if (!saveUser) return Promise.resolve();
    // Keep full-row app-state upserts in order. Rapid squad edits can otherwise
    // finish out of order and let an older snapshot overwrite the saved plan.
    const save = cloudSaveQueue.current.catch(() => {}).then(() => saveCloudState(saveUser, next));
    cloudSaveQueue.current = save;
    return save.catch((error) => {
      console.error("Could not save cloud data", error);
      throw error;
    });
  };
  const refreshAppState = useCallback(async () => {
    const activeUser = session?.user;
    if (!activeUser) return null;
    const { data, error } = await supabase.from("user_app_state").select("*").eq("user_id", activeUser.id).maybeSingle();
    if (error) throw error;
    if (!data) return null;
    const latest = {
      assignments: data.assignments || [],
      shortlists: data.shortlists || [],
      shortlistFolders: data.shortlistFolders || data.squadPlan?.shortlistFolders || [],
      tags: data.tags || [],
      squadPlan: data.squadPlan || { savedPlans: [] },
      playerSummaries: data.playerSummaries || {},
    };
    setAppState(latest);
    return latest;
  }, [session?.user]);
  return <AuthContext.Provider value={{ session, user: session?.user || null, profile, entitlement, profileError, appState, updateAppState, refreshAppState, loading, refreshProfile: () => loadProfile(session?.user) }}>{children}</AuthContext.Provider>;
}
export const useAuth = () => useContext(AuthContext);
