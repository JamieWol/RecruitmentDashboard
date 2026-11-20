import Stripe from "npm:stripe@17.7.0";
import { createClient } from "npm:@supabase/supabase-js@2";

const stripeSecretKey = Deno.env.get("STRIPE_SECRET_KEY") || "";
const webhookSecret = Deno.env.get("STRIPE_WEBHOOK_SIGNING_SECRET") || "";
const stripe = stripeSecretKey ? new Stripe(stripeSecretKey, { httpClient: Stripe.createFetchHttpClient() }) : null;
const supabaseUrl = Deno.env.get("SUPABASE_URL") || "";
const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";
const admin = supabaseUrl && serviceRoleKey ? createClient(supabaseUrl, serviceRoleKey, { auth: { persistSession: false, autoRefreshToken: false } }) : null;

const toIso = (seconds?: number | null) => typeof seconds === "number" ? new Date(seconds * 1000).toISOString() : null;
const entitlementStatus = (status: Stripe.Subscription.Status) => {
  if (status === "active" || status === "trialing") return "active";
  if (status === "canceled" || status === "incomplete_expired") return "canceled";
  return "pending";
};
const customerId = (customer: string | Stripe.Customer | Stripe.DeletedCustomer | null) => typeof customer === "string" ? customer : customer?.id || null;

Deno.serve(async (request) => {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  if (!stripe || !webhookSecret || !admin) return new Response("Webhook is not configured", { status: 500 });
  const signature = request.headers.get("stripe-signature");
  if (!signature) return new Response("Missing Stripe signature", { status: 400 });

  const rawBody = await request.text();
  let event: Stripe.Event;
  try {
    event = await stripe.webhooks.constructEventAsync(
      rawBody,
      signature,
      webhookSecret,
      undefined,
      Stripe.createSubtleCryptoProvider(),
    );
  } catch (error) {
    console.error("Invalid Stripe webhook signature", error);
    return new Response("Invalid signature", { status: 400 });
  }

  if (!event.type.startsWith("customer.subscription.")) return Response.json({ received: true });
  const subscription = event.data.object as Stripe.Subscription;
  const metadata = subscription.metadata || {};
  const planCode = metadata.plan_code || "";
  const bundleAddonPriceId = Deno.env.get("STRIPE_PRICE_TOOLS_BUNDLE_MONTHLY");
  const hasBundleAddon = subscription.items.data.some((item) => item.price.id === bundleAddonPriceId);
  const status = planCode === "independent_bundle" && (!hasBundleAddon || subscription.pending_update)
    ? "pending"
    : entitlementStatus(subscription.status);
  const customer = customerId(subscription.customer);
  const priceId = subscription.items.data[0]?.price?.id || null;
  const currentPeriodEnd = toIso(subscription.current_period_end ?? subscription.items.data[0]?.current_period_end);
  const entitlement = {
    status,
    stripe_customer_id: customer,
    stripe_subscription_id: subscription.id,
    stripe_price_id: priceId,
    current_period_end: currentPeriodEnd,
    cancel_at_period_end: Boolean(subscription.cancel_at_period_end),
    stripe_event_created_at: event.created,
  };

  let result;
  if (["independent", "independent_bundle"].includes(planCode) && metadata.user_id) {
    const { data: existing } = await admin.from("account_entitlements").select("plan_code,status,stripe_event_created_at").eq("user_id", metadata.user_id).maybeSingle();
    if (existing?.stripe_event_created_at && existing.stripe_event_created_at > event.created) return Response.json({ received: true, stale: true });
    if (planCode === "independent_bundle" && status === "pending" && existing?.plan_code === "independent" && existing.status === "active") {
      return Response.json({ received: true, upgrade_pending: true });
    }
    result = await admin.from("account_entitlements").upsert({
      user_id: metadata.user_id,
      plan_code: planCode,
      ...entitlement,
      updated_at: new Date().toISOString(),
    }, { onConflict: "user_id" });
  } else if (["club", "organisation"].includes(planCode) && metadata.workspace_key && metadata.workspace_name) {
    const { data: existing } = await admin.from("workspace_entitlements").select("stripe_event_created_at").eq("workspace_key", metadata.workspace_key).maybeSingle();
    if (existing?.stripe_event_created_at && existing.stripe_event_created_at > event.created) return Response.json({ received: true, stale: true });
    result = await admin.from("workspace_entitlements").upsert({
      workspace_key: metadata.workspace_key,
      workspace_name: metadata.workspace_name,
      plan_code: planCode,
      ...entitlement,
      seat_limit: 7,
      updated_at: new Date().toISOString(),
    }, { onConflict: "workspace_key" });
  } else {
    console.error("Ignoring Stripe subscription without valid ScoutPro plan metadata", event.id);
    return Response.json({ received: true, ignored: true });
  }

  if (result?.error) {
    console.error("Could not update subscription entitlement", event.id, result.error);
    return new Response("Could not update plan access", { status: 500 });
  }
  return Response.json({ received: true });
});
