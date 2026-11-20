# Stripe Checkout Setup

This integration keeps Stripe secret keys in Supabase Edge Function secrets. Do not add them to React environment variables or commit them to source control.

## Required Supabase secrets

Set these with `supabase secrets set NAME=value`:

- `STRIPE_SECRET_KEY` — Stripe restricted or secret key for the same Stripe account as the price IDs.
- `STRIPE_WEBHOOK_SIGNING_SECRET` — signing secret for the Stripe webhook endpoint.
- `STRIPE_PRICE_INDEPENDENT_MONTHLY` — £9.99 recurring monthly price.
- `STRIPE_PRICE_INDEPENDENT_YEARLY` — £99 recurring yearly price.
- `STRIPE_PRICE_TOOLS_BUNDLE_MONTHLY` — £7.99 recurring monthly add-on price.
- `STRIPE_PRICE_CLUB_MONTHLY` — £49 recurring monthly Club Workspace price.
- `APP_BASE_URL` — public ScoutPro origin, with no trailing slash.
- `DEV_ALLOWED_ORIGINS` — optional comma-separated local development origins.

Supabase provides `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to Edge Functions automatically. Do not add a custom secret whose name starts with `SUPABASE_`.

The Full Tools Bundle Checkout creates one monthly subscription with both the Independent Scout monthly price and the tools add-on price. The add-on flow for an existing monthly Independent Scout subscription adds the add-on item with proration. Existing yearly subscribers need to switch to monthly in Stripe Billing Portal before adding the monthly add-on.

## Apply and deploy

1. Apply `supabase/migrations/202611200002_stripe_subscriptions.sql` after the Independent Trial migration.
2. Set the secrets above in the Supabase project.
3. Deploy `create-checkout-session`, `create-billing-portal-session`, `add-independent-tools-bundle`, and `stripe-webhook` Edge Functions.
4. In Stripe, create a webhook endpoint at `https://YOUR_PROJECT_REF.supabase.co/functions/v1/stripe-webhook` for `customer.subscription.created`, `customer.subscription.updated`, and `customer.subscription.deleted`. Add its signing secret to Supabase.
5. Enable Stripe Billing Portal and allow customers to update payment methods and cancel subscriptions. Product switching is handled by ScoutPro's bundle flow; keep arbitrary product changes disabled so Stripe metadata continues to match the selected plan.

`stripe-webhook` is configured without JWT verification because Stripe authenticates webhook deliveries with its signature. The function verifies that signature over the raw request body before processing subscription updates. The checkout and portal functions require a Supabase user JWT and re-check account and workspace details server-side. Paid Independent Scout signups can open Checkout before approval; the signed Stripe webhook activates the account after an active subscription is confirmed. Free trials and club setup requests continue through the review process.

The Club Workspace seat limit is seven approved profiles sharing the workspace name. An eighth approval is blocked until the subscribed workspace seat limit is changed.
