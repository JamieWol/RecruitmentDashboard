import Stripe from "npm:stripe@17.7.0";
import { createClient } from "npm:@supabase/supabase-js@2";

const stripeSecretKey = Deno.env.get("STRIPE_SECRET_KEY") || "";
const stripe = stripeSecretKey ? new Stripe(stripeSecretKey, { httpClient: Stripe.createFetchHttpClient() }) : null;
const supabaseUrl = Deno.env.get("SUPABASE_URL") || "";
const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";
const appBaseUrl = (Deno.env.get("APP_BASE_URL") || "").replace(/\/$/, "");
const admin = supabaseUrl && serviceRoleKey ? createClient(supabaseUrl, serviceRoleKey, { auth: { persistSession: false, autoRefreshToken: false } }) : null;

const price = (name: string) => {
  const value = Deno.env.get(name);
  if (!value) throw new Error(`Missing Supabase function secret: ${name}`);
  return value;
};

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
  if (!origin || !allowedOrigins.includes(origin)) {
    console.warn("Checkout blocked: app origin is not allowed", { origin, allowedOrigins });
    return Response.json({ error: "This app origin is not allowed." }, { status: 403, headers });
  }
  if (!stripe || !admin || !appBaseUrl) {
    return Response.json({ error: "Billing is not fully configured. Ask the platform administrator." }, { status: 503, headers });
  }

  const authorization = request.headers.get("authorization") || "";
  const accessToken = authorization.match(/^Bearer\s+(.+)$/i)?.[1];
  if (!accessToken) return Response.json({ error: "Sign in before starting checkout." }, { status: 401, headers });
  const { data: authData, error: authError } = await admin.auth.getUser(accessToken);
  if (authError || !authData.user) return Response.json({ error: "Your sign-in has expired. Sign in again." }, { status: 401, headers });
  const user = authData.user;
  let requested: { plan?: string; interval?: string };
  try { requested = await request.json(); } catch { return Response.json({ error: "Invalid checkout request." }, { status: 400, headers }); }
  const plan = requested.plan;
  const interval = requested.interval;
  const { data: profile, error: profileError } = await admin.from("profiles").select("id,full_name,club,workspace_type,requested_plan,approved,role").eq("id", user.id).maybeSingle();
  if (profileError || !profile) {
    console.warn("Checkout blocked: profile lookup failed", { userId: user.id, profileError: profileError?.message || null });
    return Response.json({ error: "Your ScoutPro profile could not be found." }, { status: 403, headers });
  }
  const pendingPaidIndependentSignup = !profile.approved
    && profile.workspace_type === "independent"
    && ((profile.requested_plan === "Independent Scout" && plan === "independent" && ["monthly", "yearly"].includes(interval || ""))
      || (profile.requested_plan === "Independent Scout + Full Tools Bundle" && plan === "independent_bundle" && interval === "monthly"));
  if (!profile.approved && !pendingPaidIndependentSignup) {
    console.warn("Checkout blocked: account approval or requested plan does not match", {
      userId: user.id,
      workspaceType: profile.workspace_type,
      requestedPlan: profile.requested_plan,
      approved: profile.approved,
      plan,
      interval,
    });
    return Response.json({ error: "Your account must be approved before checkout." }, { status: 403, headers });
  }

  let planCode = "";
  let lineItems: Stripe.Checkout.SessionCreateParams.LineItem[] = [];
  let workspaceKey = "";
  let workspaceName = "";
  let customerId: string | undefined;
  let existingSubscriptionId: string | null = null;

  try {
    if (profile.workspace_type === "independent") {
      const { data: current } = await admin.from("account_entitlements").select("stripe_customer_id,stripe_subscription_id,status,current_period_end").eq("user_id", user.id).maybeSingle();
      customerId = current?.stripe_customer_id || undefined;
      existingSubscriptionId = current?.stripe_subscription_id && ["active", "pending"].includes(current.status) ? current.stripe_subscription_id : null;
      if (plan === "independent" && interval === "monthly") {
        planCode = "independent";
        lineItems = [{ price: price("STRIPE_PRICE_INDEPENDENT_MONTHLY"), quantity: 1 }];
      } else if (plan === "independent" && interval === "yearly") {
        planCode = "independent";
        lineItems = [{ price: price("STRIPE_PRICE_INDEPENDENT_YEARLY"), quantity: 1 }];
      } else if (plan === "independent_bundle" && interval === "monthly") {
        planCode = "independent_bundle";
        lineItems = [
          { price: price("STRIPE_PRICE_INDEPENDENT_MONTHLY"), quantity: 1 },
          { price: price("STRIPE_PRICE_TOOLS_BUNDLE_MONTHLY"), quantity: 1 },
        ];
      } else {
        return Response.json({ error: "That plan option is not available for this account." }, { status: 400, headers });
      }
    } else if (["organization", "club"].includes(profile.workspace_type) && (profile.workspace_type === "organization" || profile.role === "club_admin") && plan === "club" && interval === "monthly") {
      planCode = "club";
      workspaceName = String(profile.club || "").trim();
      workspaceKey = workspaceName.toLowerCase().replace(/\s+/g, " ");
      if (!workspaceKey) return Response.json({ error: "Add a workspace name before starting checkout." }, { status: 400, headers });
      const { data: current } = await admin.from("workspace_entitlements").select("stripe_customer_id,stripe_subscription_id,status,current_period_end").eq("workspace_key", workspaceKey).maybeSingle();
      customerId = current?.stripe_customer_id || undefined;
      existingSubscriptionId = current?.stripe_subscription_id && ["active", "pending"].includes(current.status) ? current.stripe_subscription_id : null;
      lineItems = [{ price: price("STRIPE_PRICE_CLUB_MONTHLY"), quantity: 1 }];
    } else {
      console.warn("Checkout blocked: unsupported workspace or plan", {
        userId: user.id,
        workspaceType: profile.workspace_type,
        role: profile.role,
        plan,
        interval,
      });
      return Response.json({ error: "Checkout is available to approved Independent Scout and Club Workspace accounts." }, { status: 403, headers });
    }

    if (existingSubscriptionId) {
      return Response.json({ error: "This account already has a Stripe subscription. Use Manage Billing to change or cancel it." }, { status: 409, headers });
    }
    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      line_items: lineItems,
      ...(customerId ? { customer: customerId } : { customer_email: profile.email || user.email }),
      client_reference_id: user.id,
      metadata: {
        user_id: user.id,
        plan_code: planCode,
        workspace_key: workspaceKey,
        workspace_name: workspaceName,
        workspace_type: profile.workspace_type,
      },
      subscription_data: {
        metadata: {
          user_id: user.id,
          plan_code: planCode,
          workspace_key: workspaceKey,
          workspace_name: workspaceName,
          workspace_type: profile.workspace_type,
        },
      },
      success_url: `${appBaseUrl}/billing?checkout=success`,
      cancel_url: `${appBaseUrl}/billing?checkout=canceled`,
      allow_promotion_codes: true,
    });
    return Response.json({ url: session.url }, { headers });
  } catch (error) {
    console.error("Stripe checkout session error", error);
    return Response.json({ error: error instanceof Error ? error.message : "Could not start checkout." }, { status: 500, headers });
  }
});
