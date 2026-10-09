import { ClubName } from "./ClubBadge";
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "./AuthContext";
import { supabase } from "./supabaseClient";

const formatDate = (value) => {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : `Updated ${date.toLocaleDateString()}`;
};

export default function SquadPlansPage() {
  const navigate = useNavigate();
  const { appState, updateAppState, profile, user } = useAuth();
  const [scouts, setScouts] = useState([]);
  const [sharedPlans, setSharedPlans] = useState([]);
  const [showShared, setShowShared] = useState(false);
  const [sharePlan, setSharePlan] = useState(null);
  const [shareScout, setShareScout] = useState("");
  const [sharePermission, setSharePermission] = useState("view");
  const [shareError, setShareError] = useState("");
  const plans = [...(appState?.squadPlan?.savedPlans || [])].sort((a, b) => new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0));

  useEffect(() => {
    if (!profile?.club) { setScouts([]); return; }
    supabase.from("profiles").select("id,full_name").eq("club", profile.club).eq("approved", true)
      .then(({ data, error }) => {
        if (error) { console.error("Could not load scouts for squad sharing", error); return; }
        setScouts((data || []).filter((scout) => scout.id !== user?.id));
      });
  }, [profile?.club, user?.id]);

  useEffect(() => {
    if (!user) return;
    let cancelled = false;
    const loadSharedPlans = async () => {
      const { data: shares, error: shareLoadError } = await supabase.from("shortlist_shares").select("shortlist_id,owner_id,permission").eq("member_id", user.id);
      if (shareLoadError) { console.error("Could not load shared squad plans", shareLoadError); return; }
      const squadShares = (shares || []).filter((share) => String(share.shortlist_id).startsWith("squad-plan:"));
      if (!squadShares.length) { if (!cancelled) setSharedPlans([]); return; }
      const { data: snapshots, error: snapshotError } = await supabase.from("shared_shortlists").select("shortlist_id,owner_id,shortlist,updated_at").in("shortlist_id", squadShares.map((share) => String(share.shortlist_id)));
      if (snapshotError) { console.error("Could not load shared squad plan details", snapshotError); return; }
      const ownerIds = [...new Set((snapshots || []).map((snapshot) => snapshot.owner_id).filter(Boolean))];
      const { data: owners } = ownerIds.length ? await supabase.from("profiles").select("id,full_name").in("id", ownerIds) : { data: [] };
      const ownerNames = new Map((owners || []).map((owner) => [String(owner.id), owner.full_name]));
      const plans = (snapshots || []).filter((snapshot) => snapshot.shortlist?.kind === "squad-plan").map((snapshot) => {
        const share = squadShares.find((item) => String(item.shortlist_id) === String(snapshot.shortlist_id));
        return { ...snapshot.shortlist, shared: true, shareId: snapshot.shortlist_id, owner_id: snapshot.owner_id, sharedBy: ownerNames.get(String(snapshot.owner_id)) || "Another scout", sharedPermission: share?.permission || "view" };
      });
      if (!cancelled) setSharedPlans(plans.sort((a, b) => new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0)));
    };
    loadSharedPlans();
    window.addEventListener("focus", loadSharedPlans);
    return () => { cancelled = true; window.removeEventListener("focus", loadSharedPlans); };
  }, [user]);

  const createPlan = () => navigate(`/squad-plan?new=${Date.now()}`);
  const deletePlan = (event, id) => {
    event.stopPropagation();
    if (!window.confirm("Delete this squad plan?")) return;
    const squadPlan = appState?.squadPlan || {};
    updateAppState({ ...appState, squadPlan: { ...squadPlan, savedPlans: (squadPlan.savedPlans || []).filter((plan) => String(plan.id) !== String(id)) } });
  };
  const share = async (event) => {
    event.preventDefault();
    setShareError("");
    if (!sharePlan || !shareScout || !user || !profile?.club) return setShareError("Choose a scout to share this plan with.");
    const shareId = `squad-plan:${sharePlan.id}`;
    const snapshot = { ...sharePlan, kind: "squad-plan" };
    const { error: snapshotError } = await supabase.from("shared_shortlists").upsert({ shortlist_id: shareId, owner_id: user.id, club: profile.club, shortlist: snapshot, updated_at: new Date().toISOString() }, { onConflict: "shortlist_id" });
    if (snapshotError) return setShareError(snapshotError.message || "Could not save the shared squad plan.");
    const { data: existingShare, error: readShareError } = await supabase.from("shortlist_shares").select("shortlist_id").eq("shortlist_id", shareId).eq("member_id", shareScout).maybeSingle();
    if (readShareError) return setShareError(readShareError.message || "Could not check the existing share.");
    const sharePayload = { owner_id: user.id, member_id: shareScout, club: profile.club, permission: sharePermission };
    const result = existingShare
      ? await supabase.from("shortlist_shares").update({ permission: sharePermission }).eq("shortlist_id", shareId).eq("member_id", shareScout)
      : await supabase.from("shortlist_shares").insert({ shortlist_id: shareId, ...sharePayload });
    if (result.error) return setShareError(result.error.message || "Could not share this squad plan.");
    setSharePlan(null); setShareScout(""); setSharePermission("view");
  };

  const card = (plan, shared = false) => <article className="sr-card sr-squad-plan-card" key={`${shared ? "shared" : "owned"}-${String(plan.id)}`} onClick={() => navigate(shared ? `/squad-plan?shared=${encodeURIComponent(plan.shareId)}` : `/squad-plan?plan=${encodeURIComponent(plan.id)}`)}>
    <div className="sr-card-top"><span className="sr-card-status">{shared ? "SHARED SQUAD PLAN" : "SQUAD PLAN"}</span>{shared ? <small>Shared by {plan.sharedBy}</small> : <div className="sr-card-actions"><button type="button" className="sr-outline sr-small-action" onClick={(event) => { event.stopPropagation(); setShareError(""); setSharePlan(plan); }}>Share</button><button type="button" className="sr-trash" onClick={(event) => deletePlan(event, plan.id)}>Delete</button></div>}</div>
    <h3>{plan.name || "Unnamed squad"}</h3><p><ClubName club={plan.club} fallback="No club selected" /></p><small>{plan.formation || "No formation selected"}</small><small>{Array.isArray(plan.players) ? plan.players.length : 0} players</small>{formatDate(plan.updatedAt) && <small>{formatDate(plan.updatedAt)}</small>}
  </article>;

  return <main className="sr-page sr-shortlist-view sr-squad-plans-page">
    <section className="sr-dashboard-head"><div><div className="sr-kicker">SQUAD PLANS</div><h1>{showShared ? "Shared Squad Plans" : "Your Squad Plans"}</h1><p>{showShared ? "Squad plans other scouts have shared with you." : "Open a saved squad plan or create a new one."}</p></div><div className="sr-shortlist-head-actions"><button type="button" className={showShared ? "sr-outline" : "sr-cyan"} onClick={() => setShowShared(false)}>My Squad Plans</button><button type="button" className={showShared ? "sr-cyan" : "sr-outline"} onClick={() => setShowShared(true)}>Shared with Me</button>{!showShared && <button type="button" className="sr-cyan" onClick={createPlan}>Create New Squad</button>}</div></section>
    {!appState ? <div className="sr-empty">Loading your squad plans…</div> : showShared ? sharedPlans.length ? <div className="sr-grid">{sharedPlans.map((plan) => card(plan, true))}</div> : <div className="sr-empty">No one has shared a squad plan with you yet.</div> : plans.length === 0 ? <div className="sr-empty"><p>You have not saved a squad plan yet.</p><button type="button" className="sr-cyan" onClick={createPlan}>Create your first squad</button></div> : <div className="sr-grid">{plans.map((plan) => card(plan))}</div>}
    {sharePlan && <div className="sr-modal" onClick={() => setSharePlan(null)}><form className="sr-form sr-shortlist-create" onClick={(event) => event.stopPropagation()} onSubmit={share}><div className="sr-form-head"><div><div className="sr-kicker">SQUAD PLAN SHARING</div><h2>Share {sharePlan.name}</h2></div><button type="button" className="sr-close" onClick={() => setSharePlan(null)}>×</button></div><label className="sr-field"><span>Scout</span><select value={shareScout} onChange={(event) => setShareScout(event.target.value)} required><option value="">Select scout</option>{scouts.map((scout) => <option key={scout.id} value={scout.id}>{scout.full_name || scout.id}</option>)}</select></label><label className="sr-field"><span>Permission</span><select value={sharePermission} onChange={(event) => setSharePermission(event.target.value)}><option value="view">View only</option><option value="edit">View and edit</option></select></label>{shareError && <p className="sr-auth-error">{shareError}</p>}<div className="sr-actions"><button type="button" className="sr-outline" onClick={() => setSharePlan(null)}>Cancel</button><button className="sr-cyan">Share Squad Plan</button></div></form></div>}
  </main>;
}
