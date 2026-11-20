import Stripe from "npm:stripe@17.7.0";
import { createClient } from "npm:@supabase/supabase-js@2";

const stripeSecretKey = Deno.env.get("STRIPE_SECRET_KEY") || "";
const stripe = stripeSecretKey ? new Stripe(stripeSecretKey, { httpClient: Stripe.createFetchHttpClient() }) : null;
const supabaseUrl = Deno.env.get("SUPABASE_URL") || "";
const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";
const appBaseUrl = (Deno.env.get("APP_BASE_URL") || "").replace(/\/$/, "");
const admin = supabaseUrl && serviceRoleKey ? createClient(supabaseUrl, serviceRoleKey, { auth: { persistSession: false, autoRefreshToken: false } }) : null;

Deno.serve(async (request) => {
  const origin = request.headers.get("origin") || "";
  const allowedOrigins = [appBaseUrl, ...(Deno.env.get("DEV_ALLOWED_ORIGINS") || "").split(",").map((item) => item.trim())].filter(Boolean);
  const headers = {
    "Access-Control-Allow-Origin": allowedOrigins.includes(origin) ? origin : appBaseUrl,
    "Access-Control-Allow-Headers": "authorization, apikey, content-type, x-client-info",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Vary": "Origin",
    "Content-Type": "application/json",
  };
  if (request.method === "OPTIONS") return new Response("ok", { headers });
  if (request.method !== "POST") return Response.json({ error: "Method not allowed." }, { status: 405, headers });
  if (!origin || !allowedOrigins.includes(origin)) return Response.json({ error: "This app origin is not allowed." }, { status: 403, headers });
  if (!stripe || !admin || !appBaseUrl) return Response.json({ error: "Billing is not fully configured." }, { status: 503, headers });
  const token = (request.headers.get("authorization") || "").match(/^Bearer\s+(.+)$/i)?.[1];
  if (!token) return Response.json({ error: "Sign in before changing your plan." }, { status: 401, headers });
  const { data: authData, error: authError } = await admin.auth.getUser(token);
  if (authError || !authData.user) return Response.json({ error: "Your sign-in has expired. Sign in again." }, { status: 401, headers });
  const { data: profile } = await admin.from("profiles").select("workspace_type,approved").eq("id", authData.user.id).maybeSingle();
  if (!profile?.approved || profile.workspace_type !== "independent") return Response.json({ error: "This upgrade is only available to approved independent accounts." }, { status: 403, headers });
  const { data: entitlement } = await admin.from("account_entitlements").select("plan_code,status,stripe_subscription_id").eq("user_id", authData.user.id).maybeSingle();
  if (!entitlement?.stripe_subscription_id || entitlement.status !== "active") return Response.json({ error: "Start an Independent Scout subscription before adding the tools bundle." }, { status: 409, headers });
  if (entitlement.plan_code === "independent_bundle") return Response.json({ already_enabled: true }, { headers });
  if (entitlement.plan_code !== "independent") return Response.json({ error: "Your current plan cannot be upgraded with this add-on." }, { status: 409, headers });

  try {
    const subscription = await stripe.subscriptions.retrieve(entitlement.stripe_subscription_id);
    const basePriceId = Deno.env.get("STRIPE_PRICE_INDEPENDENT_MONTHLY");
    const addonPriceId = Deno.env.get("STRIPE_PRICE_TOOLS_BUNDLE_MONTHLY");
    if (!basePriceId || !addonPriceId) return Response.json({ error: "The tools bundle prices are not configured." }, { status: 503, headers });
    const baseItem = subscription.items.data.find((item) => item.price.id === basePriceId);
    if (!baseItem) return Response.json({ error: "The bundle add-on is currently available with monthly Independent Scout billing. Change your base subscription to monthly in Manage Billing first." }, { status: 409, headers });
    if (subscription.items.data.some((item) => item.price.id === addonPriceId)) return Response.json({ already_enabled: true }, { headers });
    const update = await stripe.subscriptions.update(subscription.id, {
      items: [{ id: baseItem.id, price: basePriceId }, { price: addonPriceId, quantity: 1 }],
      proration_behavior: "always_invoice",
      payment_behavior: "pending_if_incomplete",
      metadata: { ...subscription.metadata, plan_code: "independent_bundle" },
    });
    return Response.json({ updated: true, subscription_status: update.status }, { headers });
  } catch (error) {
    console.error("Could not add the Full Tools Bundle", error);
    return Response.json({ error: error instanceof Error ? error.message : "Could not update your subscription." }, { status: 500, headers });
  }
});
