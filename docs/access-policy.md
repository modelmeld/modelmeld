# Pro feed access policy

Who is entitled to fetch the live curated routing feed, when. Source of
truth for both marketing copy and the future feed-server implementation.

Decided 2026-05-23.

## The rule

A tenant qualifies for Pro feed access (`feed.modelmeld.ai`) if and only
if **at least one** of the following is true:

### A. Active hosted customer

- Tenant has spent **$20+ in total credit purchases within the last 30 days**, OR
- Tenant's current credit balance is **> $10**

The OR is intentional. The first clause catches "just topped up but hasn't
used credits yet"; the second catches "regular customer who tops up
periodically but currently between top-ups."

A new customer who buys $20 once and never tops up loses Pro feed access
~30 days after their balance hits $10 (which usually means after they've
used roughly half their initial purchase).

### B. Active Pro subscriber

- Tenant has a Stripe subscription with `status = "active"`, `plan = "pro_subscriber"` or its successors

Pro subscribers retain feed access for as long as their monthly billing
stays current. Cancellation → access lapses at the end of the current
billing period.

## What about everyone else

If neither A nor B applies:

- The feed server returns HTTP 403 with a clear reason
- The customer's `RegistryFeedClient` automatically falls back to the
  bundled snapshot (as it does for every failure mode — see
  `src/modelmeld/scout/feed.py`)
- Nothing else breaks; the gateway keeps routing on snapshot data

## Why this rule

It aligns the customer-facing positioning ("Pro feed bundled with active
hosted accounts") with what's enforceable in code. The two conditions in
A correctly capture "currently a paying customer": one is a leading
indicator (recent spend), the other is a coincident indicator (balance
on hand). Either one means revenue is flowing.

The 30-day window matches the typical credit-pack consumption cadence
for active users. The $10 balance threshold prevents the "I bought $20
once 6 months ago and never came back" customer from leeching the feed
indefinitely.

## Implementation: deferred to v0.2 (feed-server build)

As of this writing, the feed server (`feed.modelmeld.ai`) doesn't exist
yet. The `RegistryFeedClient` in `modelmeld` knows how to fetch from one
when it exists; until then, all gateways use the bundled snapshot.

When the feed server is implemented (planned post-OSS-launch):

1. `GET https://feed.modelmeld.ai/v1/registry` validates the bearer JWT
   (signature + not expired + not revoked)
2. Then queries enterprise-control for the tenant's account state
3. Applies the rule above (cached per-tenant for ~5 minutes to bound DB load)
4. Returns 200 with signed payload OR 403 with reason

The JWT itself can stay long-lived (currently 365 days). It's an identity
token, not an access token — access is determined at fetch time by the
tenant's current account state.

## Edge cases worth thinking about now

### New-customer race

Customer purchases at T=0. Webhook fires, license issued, email sent. If
the customer immediately tries to fetch the feed (T=0+5s), the
account-state query should see their recent purchase. Implementation
detail: the balance credit and billing-event row must be committed before
the email is sent.

### Refund / chargeback

If we refund a customer (rare per ToS) or absorb a Stripe chargeback,
their balance drops. If it drops below $10 and they have no other recent
$20+ purchases, they fall out of Pro feed access at the next cached-check
refresh (5 min lag). This is correct.

### Subscription cancellation

Pro subscriber cancels mid-month. Stripe keeps the subscription active
until end of billing period. Our cached check would still consider them
active. When the period ends, Stripe sends `customer.subscription.deleted`
or similar; we update the local subscription record; next cached check
returns 403.

### Account dispute

Customer claims they didn't authorize the charge. We mark the
subscription as `disputed` (or similar). Access check should fail
immediately on disputed status. (Future work: webhook for
`charge.dispute.created`.)

### Multi-tenant abuse

Tenant resells their license JWT to multiple downstream consumers. Each
JWT fetch hits our feed server. We could rate-limit per-JWT at the
server. Not a v1 problem.

## Marketing implications

Pricing-page copy must reflect this honestly. The promise is "Pro feed
included with active hosted account" — not "free forever after one $20
purchase."

Current copy needs adjustment until the feed actually ships in v0.2.
Then re-evaluate.

---

_Owner: ops + engineering. Last reviewed: 2026-05-23._
