# Registry data — license terms

The files in this directory contain benchmark data, model metadata, and
routing-decision inputs. They are **separately licensed** from the
surrounding Python code. The AGPL-3.0-or-later license in the package root applies
to the code; the terms below apply to the data.

## What's in here

| File | Contents | License |
|---|---|---|
| `default_registry.json` | Stale seed snapshot of the curated model registry, frozen at package release time | **CC-BY-4.0** (see below) |
| `LICENSE.md` | This file | CC-BY-4.0 |

## Bundled seed — `default_registry.json`

Distributed under the **Creative Commons Attribution 4.0 International**
license (CC-BY-4.0). You may use, copy, redistribute, and adapt the seed
under the following condition:

> **Attribution** — give appropriate credit to Stretto Labs LLC, provide a
> link to this LICENSE, and indicate if changes were made.

The seed is provided so individual developers can use `modelmeld`
standalone for free with reasonable routing quality. It is **explicitly
stale** — the underlying benchmark scores represent a fixed point in time
(see the `last_updated` field) and are not updated between package releases.
Loading this file in non-test environments emits a one-time WARN log.

Use case: dev / evaluation / open-source contribution / non-commercial
testing. **Not appropriate for production routing decisions** because
benchmark gaming, model deprecation, and frontier price moves render
stale scores misleading within months.

## Live curated feed — separate subscription

The production model registry is served continuously from
`feed.modelmeld.ai` (or a self-hosted equivalent — see
`docs/feed-server-contract.md` once published). It is:

- Updated daily from Artificial Analysis API, Aider Polyglot leaderboard,
  LiveBench monthly snapshots, LMArena daily snapshots, and aggregated
  customer-workload signals.
- Editorially curated — we manually weight benchmark sources, flag suspected
  benchmark gaming, and remove deprecated models.
- Signed with an Ed25519 key; clients verify before consuming.

Access requires a ModelMeld subscription. The feed itself is **NOT
licensed under CC-BY-4.0**; it is governed by the subscription agreement,
which restricts:

- Redistribution (you may not republish the curated feed as your own)
- Reverse engineering of the curation methodology
- Sharing the license key (per-tenant; rotation procedure in admin docs)

What the subscription agreement **does NOT** restrict:

- Using the data internally to route your own production traffic — that's
  the whole point
- Building applications on top of `modelmeld` that consume the feed
- Forking `modelmeld` itself (the code is AGPL-3.0-or-later — your
  fork just won't have access to the live feed without a separate
  subscription; commercial dual-licensing available, contact
  `hello@modelmeld.ai`)

## Why this split

The code is the implementation; the curated data is the operational
product. Treating them as one licensing unit forces a false choice between
"give everything away" and "lock everything down." Splitting them — code
under network-copyleft (AGPL), data subscription-gated — preserves the
open-source adoption flywheel while keeping the moat sustainable.

This is the same model used by:

- **MaxMind GeoIP** — code (and a legacy free database) are MIT/CC-BY;
  the live commercial database (`GeoIP2`) is a paid subscription with
  restricted redistribution.
- **Have I Been Pwned** — code is BSD; the breach database access is via
  a paid API.
- **Tessera Mapping** / **OpenStreetMap commercial tiles** — software
  permissive, tile data has commercial terms.

## Contact

Questions about data licensing or subscribing to the live feed:
`hello@modelmeld.ai`.

---

_Last reviewed: 2026-05-20. Maintained by: Stretto Labs LLC_
