import { useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { supabase } from "./supabaseClient";
import { useAuth } from "./AuthContext";

export default function BillingPage() {
  const { user, profile, entitlement, workspaceEntitlement, refreshEntitlements } = useAuth();
  const [searchParams] = useSearchParams();
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [showBundleConfirm, setShowBundleConfirm] = useState(false);
  const [showPremiumCancelConfirm, setShowPremiumCancelConfirm] = useState(false);
  const checkoutState = searchParams.get("checkout");
  const isIndependent = profile?.workspace_type === "independent";
  const isWorkspaceOwner = profile?.workspace_type === "organization" || (profile?.workspace_type === "club" && profile?.role === "club_admin");
  const workspaceName = profile?.club || "Your Workspace";
  const currentSubscription = isIndependent ? entitlement : workspaceEntitlement;
  const hasStripeSubscription = Boolean(currentSubscription?.stripe_subscription_id && ["active", "pending"].includes(currentSubscription.status));
  const hasActiveIndependentSubscription = isIndependent && entitlement?.status === "active" && Boolean(entitlement?.stripe_subscription_id);
  const independentBundleActive = isIndependent && entitlement?.plan_code === "independent_bundle" && entitlement?.status === "active";
  const currentPlanName = useMemo(() => {
    const plan = currentSubscription?.plan_code;
    if (plan === "independent_trial") return "Free Trial";
    if (plan === "independent") return "Independent Scout";
    if (plan === "independent_bundle") return "Independent Scout Premium";
    if (plan === "club") return "Club Workspace";
    if (plan === "organisation") return "Larger Organisation";
    if (profile?.workspace_type === "club") return "Club Workspace";
    return "No Paid Plan Active";
  }, [currentSubscription?.plan_code, profile?.workspace_type]);

  useEffect(() => {
    if (checkoutState === "success") {
      setNotice("Checkout completed. Stripe is confirming your subscription; your access will update automatically shortly.");
      let checks = 0;
      const timer = window.setInterval(() => {
        checks += 1;
        refreshEntitlements().catch(() => {});
        if (checks >= 10) window.clearInterval(timer);
      }, 2500);
      return () => window.clearInterval(timer);
    }
    if (checkoutState === "canceled") setNotice("Checkout was canceled. No subscription was started.");
    return undefined;
  }, [checkoutState, refreshEntitlements]);

  const beginCheckout = useCallback(async (plan, interval) => {
    setBusy(`${plan}:${interval}`);
    setError("");
    setNotice("");
    const { data, error: invokeError } = await supabase.functions.invoke("create-checkout-session", { body: { plan, interval } });
    setBusy("");
    if (invokeError || !data?.url) {
      setError(data?.error || invokeError?.message || "Could not open secure checkout. Please try again.");
      return;
    }
    window.location.assign(data.url);
  }, []);

  const openPortal = useCallback(async () => {
    setBusy("portal");
    setError("");
    const { data, error: invokeError } = await supabase.functions.invoke("create-billing-portal-session", { body: {} });
    setBusy("");
    if (invokeError || !data?.url) {
      setError(data?.error || invokeError?.message || "Could not open billing management.");
      return;
    }
    window.location.assign(data.url);
  }, []);

  const addToolsBundle = useCallback(async () => {
    if (!hasActiveIndependentSubscription || entitlement?.plan_code !== "independent") {
      await beginCheckout("independent_bundle", "monthly");
      return;
    }
    setShowBundleConfirm(false);
    setBusy("bundle-upgrade");
    setError("");
    const { data, error: invokeError } = await supabase.functions.invoke("add-independent-tools-bundle", { body: {} });
    setBusy("");
    if (invokeError || (!data?.updated && !data?.already_enabled)) {
      setError(data?.error || invokeError?.message || "Could not add Premium.");
      return;
    }
    setNotice("Your Premium upgrade was sent to Stripe. The extra tools will unlock after Stripe confirms it.");
    window.setTimeout(() => refreshEntitlements().catch(() => {}), 2000);
  }, [beginCheckout, entitlement?.plan_code, hasActiveIndependentSubscription, refreshEntitlements]);

  const requestToolsBundle = useCallback(async () => {
    if (!hasActiveIndependentSubscription || entitlement?.plan_code !== "independent") {
      await addToolsBundle();
      return;
    }
    setShowBundleConfirm(true);
  }, [addToolsBundle, entitlement?.plan_code, hasActiveIndependentSubscription]);

  const cancelPremium = useCallback(async () => {
    setBusy("premium-cancel");
    setError("");
    setShowPremiumCancelConfirm(false);
    const { data, error: invokeError } = await supabase.functions.invoke("remove-independent-tools-bundle", { body: {} });
    setBusy("");
    if (invokeError || (!data?.updated && !data?.already_removed)) {
      setError(data?.error || invokeError?.message || "Could not remove Premium from your subscription.");
      return;
    }
    setNotice("Premium has been removed from your subscription. Independent Scout stays active, and Stripe will apply any prorated credit to a future invoice where applicable.");
    window.setTimeout(() => refreshEntitlements().catch(() => {}), 2000);
  }, [refreshEntitlements]);

  if (!profile?.approved || !user) return <main className="sr-page"><p>Your account must be approved before billing is available.</p></main>;
  return <main className="sr-page sr-billing-page">
    <header className="sr-billing-heading"><div><span className="sr-kicker">ACCOUNT</span><h1>Plans And Billing</h1><p>Review your current plan, start a subscription or manage billing through Stripe.</p></div></header>
    {error && <p className="sr-billing-alert" role="alert">{error}</p>}
    {notice && <p className="sr-billing-notice" role="status">{notice}</p>}
    <section className="sr-billing-current"><div><small>Current Plan</small><h2>{currentPlanName}</h2><p>{isWorkspaceOwner ? workspaceName : isIndependent ? "Independent Scout workspace" : `Managed by ${workspaceName}`}</p></div><div className="sr-billing-status"><strong>{currentSubscription?.status ? currentSubscription.status.replaceAll("_", " ") : profile.workspace_type === "club" ? "Club access" : "Not active"}</strong>{currentSubscription?.current_period_end && <small>Renews or ends {new Date(currentSubscription.current_period_end).toLocaleDateString()}</small>}</div></section>

    {isIndependent ? <section className="sr-billing-plans" aria-label="Independent Scout plans">
      <article className="sr-billing-plan"><span>CORE PLAN</span><h2>Independent Scout</h2><p>Player reports, assignments and summaries.</p><strong>£9.99 <small>/ month</small></strong>{hasStripeSubscription ? <div className="sr-billing-active-plan">Current plan active</div> : <div className="sr-billing-actions"><button type="button" className="sr-cyan" disabled={Boolean(busy)} onClick={() => beginCheckout("independent", "monthly")}>{busy === "independent:monthly" ? "Opening Checkout…" : "Subscribe Monthly · £9.99"}</button><button type="button" className="sr-outline" disabled={Boolean(busy)} onClick={() => beginCheckout("independent", "yearly")}>{busy === "independent:yearly" ? "Opening Checkout…" : "Subscribe Yearly · £99"}</button></div>}</article>
      <article className="sr-billing-plan featured"><span>PREMIUM ADD-ON</span><h2>Independent Scout Premium</h2><p>Independent Scout plus shortlists, squad plans and team analysis.</p><strong>£17.98 <small>/ month total</small></strong><small className="sr-billing-plan-note">Includes the £9.99 Independent Scout plan and £7.99 Premium add-on.</small>{independentBundleActive ? <><div className="sr-billing-active-plan">Premium is active</div><button type="button" className="sr-outline" disabled={Boolean(busy)} onClick={() => setShowPremiumCancelConfirm(true)}>Cancel Premium · Keep Independent Scout</button></> : <button type="button" className="sr-cyan" disabled={Boolean(busy)} onClick={requestToolsBundle}>{busy === "bundle-upgrade" ? "Updating Subscription…" : busy === "independent_bundle:monthly" ? "Opening Checkout…" : entitlement?.plan_code === "independent" && hasActiveIndependentSubscription ? "Add Premium · £7.99/month" : "Subscribe With Premium"}</button>}</article>
      {currentSubscription?.stripe_customer_id && <button type="button" className="sr-outline sr-billing-manage" disabled={Boolean(busy)} onClick={openPortal}>{busy === "portal" ? "Opening Billing…" : "Manage Or Cancel Subscription"}</button>}
    </section> : isWorkspaceOwner ? <section className="sr-billing-plans" aria-label="Club Workspace plans">
      <article className="sr-billing-plan featured"><span>FOR {workspaceName.toUpperCase()}</span><h2>Club Workspace</h2><p>Shared reports, shortlists, squad plans and team analysis for up to seven users.</p><strong>£49 <small>/ month</small></strong><button type="button" className="sr-cyan" disabled={Boolean(busy) || hasStripeSubscription} onClick={() => beginCheckout("club", "monthly")}>{busy === "club:monthly" ? "Opening Checkout…" : hasStripeSubscription ? "Already Subscribed" : "Subscribe To Club Workspace"}</button></article>
      {currentSubscription?.stripe_customer_id && <button type="button" className="sr-outline sr-billing-manage" disabled={Boolean(busy)} onClick={openPortal}>{busy === "portal" ? "Opening Billing…" : "Manage Or Cancel Subscription"}</button>}
    </section> : <section className="sr-billing-current"><div><h2>Billing Is Managed By Your Club</h2><p>Ask your club administrator to manage the workspace subscription.</p></div></section>}
    <p className="sr-billing-footnote">Payments are securely processed by Stripe. Your card details are not stored in ScoutPro. Workspace access updates after Stripe confirms the subscription.</p>
    {showBundleConfirm && <div className="sr-billing-confirm-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) setShowBundleConfirm(false); }}>
      <section className="sr-billing-confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="sr-bundle-confirm-title" aria-describedby="sr-bundle-confirm-copy">
        <span className="sr-kicker">SUBSCRIPTION UPDATE</span>
        <h2 id="sr-bundle-confirm-title">Add Premium?</h2>
        <p id="sr-bundle-confirm-copy">This adds £7.99/month to your existing £9.99/month Independent Scout plan, bringing the total to <strong>£17.98/month</strong>. Stripe will update your subscription now; any prorated amount due today will be shown by Stripe.</p>
        <div className="sr-billing-confirm-actions">
          <button type="button" className="sr-outline" disabled={Boolean(busy)} onClick={() => setShowBundleConfirm(false)}>Cancel</button>
          <button type="button" className="sr-cyan" disabled={Boolean(busy)} onClick={addToolsBundle}>{busy === "bundle-upgrade" ? "Updating Subscription…" : "Confirm And Add Premium"}</button>
        </div>
      </section>
    </div>}
    {showPremiumCancelConfirm && <div className="sr-billing-confirm-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) setShowPremiumCancelConfirm(false); }}>
      <section className="sr-billing-confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="sr-premium-cancel-title" aria-describedby="sr-premium-cancel-copy">
        <span className="sr-kicker">SUBSCRIPTION CHANGE</span>
        <h2 id="sr-premium-cancel-title">Cancel Premium?</h2>
        <p id="sr-premium-cancel-copy">This removes the £7.99/month Premium add-on and keeps your Independent Scout plan active at £9.99/month. Premium tools will switch off once Stripe confirms the change. Stripe applies any prorated credit to a future invoice where applicable.</p>
        <div className="sr-billing-confirm-actions">
          <button type="button" className="sr-outline" disabled={Boolean(busy)} onClick={() => setShowPremiumCancelConfirm(false)}>Keep Premium</button>
          <button type="button" className="sr-cyan" disabled={Boolean(busy)} onClick={cancelPremium}>{busy === "premium-cancel" ? "Removing Premium…" : "Confirm Cancel Premium"}</button>
        </div>
      </section>
    </div>}
  </main>;
}
