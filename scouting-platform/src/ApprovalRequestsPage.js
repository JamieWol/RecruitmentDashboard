import { useCallback, useEffect, useState } from "react";
import { useAuth } from "./AuthContext";
import { supabase } from "./supabaseClient";
import "./approvalRequests.css";

export default function ApprovalRequestsPage() {
  const { profile } = useAuth();
  const [requests, setRequests] = useState([]);
  const [approvedAccounts, setApprovedAccounts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [workingId, setWorkingId] = useState(null);
  const [notice, setNotice] = useState("");
  const isPlatformAdmin = profile?.role === "platform_admin";
  const isClubAdmin = profile?.role === "club_admin";
  const canManageAccess = isPlatformAdmin || isClubAdmin;

  const loadRequests = useCallback(async () => {
    setLoading(true);
    setError("");
    const [pendingResult, approvedResult] = await Promise.all([
      supabase.rpc("list_pending_access_requests"),
      canManageAccess ? supabase.rpc("list_approved_access_accounts") : Promise.resolve({ data: [], error: null }),
    ]);
    if (pendingResult.error) {
      setError(pendingResult.error.message || "Could not load access requests.");
      setRequests([]);
    } else setRequests(pendingResult.data || []);
    if (approvedResult.error) {
      setError((current) => [current, approvedResult.error.message || "Could not load active accounts."].filter(Boolean).join(" "));
      setApprovedAccounts([]);
    } else setApprovedAccounts(approvedResult.data || []);
    setLoading(false);
  }, [canManageAccess]);

  useEffect(() => { loadRequests(); }, [loadRequests]);

  const review = async (request, approve, makeClubAdmin = false) => {
    setWorkingId(request.id);
    setError("");
    setNotice("");
    const { error: rpcError } = await supabase.rpc("review_access_request", {
      p_user_id: request.id,
      p_approve: approve,
      p_make_club_admin: makeClubAdmin,
    });
    setWorkingId(null);
    if (rpcError) {
      setError(rpcError.message || "Could not update this request.");
      return;
    }
    let trialSetupError = "";
    let trialStarted = false;
    if (approve && isPlatformAdmin && !makeClubAdmin) {
      const { data: trialResult, error: trialError } = await supabase.rpc("start_independent_trial", { p_user_id: request.id });
      if (trialError) trialSetupError = trialError.message || "Trial activation failed.";
      else trialStarted = Boolean(trialResult?.started);
    }
    await loadRequests();
    setNotice(`${request.full_name || request.email || "Account"} ${approve ? (makeClubAdmin ? "approved as club admin" : "approved") : "declined"}.${trialStarted ? " Their 30-day Independent Scout trial is now active (up to 4 published reports)." : ""}`);
    if (trialSetupError) setError(`The account was approved, but trial activation failed. Apply the Independent Trial migration and retry: ${trialSetupError}`);
  };

  const revokeAccess = async (account) => {
    const name = account.full_name || account.email || "this account";
    if (!window.confirm(`Revoke ScoutPro access for ${name}? They will lose app access on their next refresh or sign-in. This does not delete their account or data, or cancel a Stripe subscription.`)) return;
    setWorkingId(account.id);
    setError("");
    setNotice("");
    const { error: revokeError } = await supabase.rpc("revoke_user_access", { p_user_id: account.id });
    setWorkingId(null);
    if (revokeError) {
      setError(revokeError.message || "Could not revoke this account's access.");
      return;
    }
    await loadRequests();
    setNotice(`${name} no longer has access to ScoutPro.`);
  };

  if (!isPlatformAdmin && !isClubAdmin) {
    return <main className="sr-page sr-approvals-page"><h1>Access approvals</h1><p className="sr-approval-alert">Your account does not have access to this page.</p></main>;
  }

  return <main className="sr-page sr-approvals-page">
    <header className="sr-approval-heading">
      <div><span className="eyebrow">ACCOUNT SETTINGS</span><h1>Access approvals</h1><p>{isPlatformAdmin ? "Review new account requests across clubs and appoint each club’s first admin." : `Review account requests for ${profile?.club || "your club"}.`}</p></div>
      <button type="button" className="sr-outline" onClick={loadRequests} disabled={loading}>Refresh</button>
    </header>
    {error && <p className="sr-approval-alert" role="alert">{error}</p>}
    {notice && <p className="sr-approval-notice" role="status">{notice}</p>}
    {loading ? <div className="sr-approval-empty">Loading requests…</div> : requests.length === 0 ? <div className="sr-approval-empty"><strong>No pending requests</strong><span>New signups will appear here after they request access.</span></div> : <section className="sr-approval-list" aria-label="Pending access requests">
      {requests.map((request) => <article className="sr-approval-card" key={request.id}>
        <div className="sr-approval-person"><div className="sr-approval-avatar" aria-hidden="true">{(request.full_name || request.email || "?").trim().charAt(0).toUpperCase()}</div><div><h2>{request.full_name || "Name not provided"}</h2><p>{request.email}</p></div></div>
        <div className="sr-approval-meta"><span><small>Club</small><strong>{request.club || "Not selected"}</strong></span><span><small>Requested</small><strong>{request.requested_at ? new Date(request.requested_at).toLocaleDateString() : "—"}</strong></span></div>
        <div className="sr-approval-actions">
          <button type="button" className="sr-cyan" disabled={workingId === request.id} onClick={() => review(request, true)}>{workingId === request.id ? "Updating…" : isPlatformAdmin ? "Approve as scout" : "Approve member"}</button>
          {isPlatformAdmin && <button type="button" className="sr-approve-admin" disabled={workingId === request.id} onClick={() => review(request, true, true)}>Approve as club admin</button>}
          <button type="button" className="sr-reject-button" disabled={workingId === request.id} onClick={() => review(request, false)}>Decline</button>
        </div>
      </article>)}
    </section>}
    {canManageAccess && <section className="sr-approved-accounts" aria-labelledby="sr-approved-accounts-title">
      <div className="sr-approved-accounts-heading"><div><h2 id="sr-approved-accounts-title">Active accounts</h2><p>{isPlatformAdmin ? "Revoke access for any approved scout or club admin." : `Manage approved scouts at ${profile?.club || "your club"}.`} Their account and data remain, and any Stripe subscription must be canceled separately.</p></div><span>{approvedAccounts.length} accounts</span></div>
      {loading ? <div className="sr-approval-empty">Loading active accounts…</div> : approvedAccounts.length === 0 ? <div className="sr-approval-empty"><strong>No active accounts</strong><span>Approved non-admin accounts will appear here.</span></div> : <section className="sr-approval-list" aria-label="Active accounts">
        {approvedAccounts.map((account) => <article className="sr-approval-card" key={account.id}>
          <div className="sr-approval-person"><div className="sr-approval-avatar" aria-hidden="true">{(account.full_name || account.email || "?").trim().charAt(0).toUpperCase()}</div><div><h2>{account.full_name || "Name not provided"}</h2><p>{account.email}</p></div></div>
          <div className="sr-approval-meta"><span><small>Club</small><strong>{account.club || "Independent"}</strong></span><span><small>Role</small><strong>{account.role === "club_admin" ? "Club admin" : "Scout"}</strong></span></div>
          <div className="sr-approval-actions"><button type="button" className="sr-revoke-access" disabled={workingId === account.id} onClick={() => revokeAccess(account)}>{workingId === account.id ? "Revoking…" : "Revoke access"}</button></div>
        </article>)}
      </section>}
    </section>}
  </main>;
}
