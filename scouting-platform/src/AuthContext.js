import React, { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "./supabaseClient";
import { migrateAndLoadState, saveCloudState, syncPublishedReports } from "./cloudState";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [appState, setAppState] = useState(null);
  const [profileError, setProfileError] = useState("");
  const loadProfile = async (user) => {
    if (!user) { setProfile(null); setAppState(null); setProfileError(""); return; }
    const { data, error } = await supabase.from("profiles").select("*").eq("id", user.id).maybeSingle();
    if (error) { console.error("Could not load account profile", error); setProfile(null); setProfileError(error.message || String(error)); return; }
    if (!data) { setProfile(null); setProfileError("No profile row was returned for this user."); return; }
    setProfileError("");
    setProfile(data || null);
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
    if (!session?.user) { setProfile(null); setAppState(null); setLoading(false); return; }
    setLoading(true);
    loadProfile(session.user).finally(() => setLoading(false));
  }, [session]);
  const updateAppState = (next) => { setAppState(next); saveCloudState(session?.user, next).catch((error) => console.error("Could not save cloud data", error)); };
  return <AuthContext.Provider value={{ session, user: session?.user || null, profile, profileError, appState, updateAppState, loading, refreshProfile: () => loadProfile(session?.user) }}>{children}</AuthContext.Provider>;
}
export const useAuth = () => useContext(AuthContext);
