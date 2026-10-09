import { useCallback, useEffect, useState } from "react";
import { useAuth } from "./AuthContext";
import { supabase } from "./supabaseClient";
import "./approvalRequests.css";

export default function ApprovalRequestsPage() {
  const { profile } = useAuth();
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [workingId, setWorkingId] = useState(null);
  const [notice, setNotice] = useState("");
  const isPlatformAdmin = profile?.role === "platform_admin";
  const isClubAdmin = profile?.role === "club_admin";

  const loadRequests = useCallback(async () => {
    setLoading(true);
    setError("");
    const { data, error: rpcError } = await supabase.rpc("list_pending_access_requests");
    if (rpcError) {
      setError(rpcError.message || "Could not load access requests.");
      setRequests([]);
    } else {
      setRequests(data || []);
    }
    setLoading(false);
  }, []);

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
    setNotice(`${request.full_name || request.email || "Account"} ${approve ? (makeClubAdmin ? "approved as club admin" : "approved") : "declined"}.`);
    await loadRequests();
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
  </main>;
}
