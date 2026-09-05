import React, { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "./supabaseClient";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const loadProfile = async (user) => {
    if (!user) { setProfile(null); return; }
    const { data } = await supabase.from("profiles").select("*").eq("id", user.id).maybeSingle();
    setProfile(data || null);
  };
  useEffect(() => {
    supabase.auth.getSession().then(async ({ data }) => { setSession(data.session); await loadProfile(data.session?.user); setLoading(false); });
    const { data: listener } = supabase.auth.onAuthStateChange(async (_event, next) => { setSession(next); await loadProfile(next?.user); setLoading(false); });
    return () => listener.subscription.unsubscribe();
  }, []);
  return <AuthContext.Provider value={{ session, user: session?.user || null, profile, loading, refreshProfile: () => loadProfile(session?.user) }}>{children}</AuthContext.Provider>;
}
export const useAuth = () => useContext(AuthContext);
