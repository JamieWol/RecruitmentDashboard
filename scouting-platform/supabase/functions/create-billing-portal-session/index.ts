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
  if (!token) return Response.json({ error: "Sign in before managing billing." }, { status: 401, headers });
  const { data: authData, error: authError } = await admin.auth.getUser(token);
  if (authError || !authData.user) return Response.json({ error: "Your sign-in has expired. Sign in again." }, { status: 401, headers });
  const { data: profile } = await admin.from("profiles").select("club,workspace_type,approved,role").eq("id", authData.user.id).maybeSingle();
  if (!profile?.approved) return Response.json({ error: "Your account must be approved before managing billing." }, { status: 403, headers });

  let customerId: string | null = null;
  if (profile.workspace_type === "independent") {
    const { data } = await admin.from("account_entitlements").select("stripe_customer_id").eq("user_id", authData.user.id).maybeSingle();
    customerId = data?.stripe_customer_id || null;
  } else if (["club", "organization"].includes(profile.workspace_type)) {
    if (profile.workspace_type === "club" && !["club_admin", "platform_admin"].includes(profile.role)) {
      return Response.json({ error: "Only a club administrator can manage the workspace subscription." }, { status: 403, headers });
    }
    const workspaceKey = String(profile.club || "").trim().toLowerCase().replace(/\s+/g, " ");
    const { data } = await admin.from("workspace_entitlements").select("stripe_customer_id").eq("workspace_key", workspaceKey).maybeSingle();
    customerId = data?.stripe_customer_id || null;
  }
  if (!customerId) return Response.json({ error: "No Stripe billing profile is linked to this account yet." }, { status: 404, headers });

  try {
    const portal = await stripe.billingPortal.sessions.create({ customer: customerId, return_url: `${appBaseUrl}/billing` });
    return Response.json({ url: portal.url }, { headers });
  } catch (error) {
    console.error("Stripe billing portal error", error);
    return Response.json({ error: error instanceof Error ? error.message : "Could not open billing management." }, { status: 500, headers });
  }
});
