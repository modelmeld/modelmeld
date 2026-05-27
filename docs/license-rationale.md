# License rationale — why AGPL-3.0, not Apache-2.0

This document exists to answer the question every prospective user, contributor, or integration partner asks within the first 60 seconds of looking at ModelMeld: **"why AGPL? am I going to get pulled into a copyleft mess?"**

Short answer: **no, you're fine.** The longer answer is below.

---

## The TL;DR for different audiences

| You are... | What this means for you |
|---|---|
| **An individual developer** running modelmeld locally for personal use | Nothing changes. AGPL imposes obligations only when you OFFER a modified gateway as a service to OTHER people. Your local install is yours. |
| **A company** running modelmeld internally for your engineering team | Same as individuals. If you modify the gateway for your team's use, that's internal use — no AGPL obligations. |
| **A dev tool** (Cursor, Aider, Claude Code, Continue, Cline, etc.) calling the gateway over HTTP | **You are not affected.** HTTP-call clients are not derivative works of the gateway. Your codebase stays under whatever license you already have. AGPL stops at our process boundary. |
| **A framework** (AutoGen, CrewAI, LangGraph, OpenClaw, MetaGPT) routing requests through ModelMeld | Same as dev tools — you call us over the network; you're unaffected. |
| **An author of a Python package** that imports `modelmeld` and embeds it in their distribution | This IS a derived work. Your package would also need to be AGPL (or a compatible copyleft license) to distribute it. Most integrations don't need this — they call us over HTTP, not import us. If you have a real need to embed: contact `hello@modelmeld.ai` for a commercial license. |
| **A service provider** offering "managed ModelMeld" or "ModelMeld plus our proprietary tuning" to third-party customers, with modifications you don't want to share | **This is the scenario AGPL is designed to prevent.** Either publish your modifications under AGPL, or contact `hello@modelmeld.ai` for a commercial license. |

If you're not in the last row, you don't need to think about AGPL.

---

## Why we made this choice

### What Apache-2.0 doesn't protect against

ModelMeld's most valuable assets are not the routing code per se — that's a few thousand lines of Python anyone could re-implement. The actual moat is:

1. **Continuously curated routing data** (the live feed)
2. **Customer relationships + distribution within dev tools**
3. **The accumulated knowledge of which routing decisions work in production**

Under Apache-2.0, a well-funded competitor could fork modelmeld, throw their own compute at producing comparable routing data (it's not expensive — it's "run HumanEval against 20 models monthly + tune"), embed the result into their existing product with their existing distribution, and ship something we couldn't catch up to. This is the pattern that played out with Elasticsearch (AWS), Redis (AWS), Terraform (HashiCorp), Sentry — successful Apache-licensed projects that eventually had to relicense because hyperscalers were eating them.

Relicensing later is painful. The community feels betrayed, forks emerge (OpenSearch, Valkey, OpenTofu), and the optics damage outlasts the legal change. We'd rather make the right call upfront.

### What AGPL-3.0 actually requires

AGPL is identical to GPL-3.0 except for **one clause** (section 13): if you modify the software AND make it available to users over a network, you must offer those users the source code of your modifications.

That's it. That's the entire delta.

The two things people incorrectly worry about:

1. **"AGPL is viral and will infect my project."** False. AGPL applies to **derived works of the AGPL software**. A program that calls an AGPL service over a network is NOT a derived work. The court precedent on this is settled (Microsoft, IBM, and Google all have explicit policies acknowledging this). Cursor, Aider, Claude Code, Continue, OpenAI SDK, anthropic SDK, AutoGen, CrewAI, LangGraph, OpenClaw — none of them become AGPL by calling our gateway over HTTP. The viral-AGPL fear is FUD from a 2010-era misreading.

2. **"AGPL prevents commercial use."** False. AGPL explicitly permits commercial use. What it requires is that if you USE the software TO OFFER A SERVICE, you must share your modifications. Companies running AGPL software internally have zero obligations beyond the standard GPL ones. Companies offering AGPL software as a managed service must share modifications OR negotiate a commercial license.

### Why not BUSL or FSL (Terraform's / Sentry's approach)

BUSL (Business Source License) and FSL (Functional Source License) are popular newer "source-available" licenses with explicit non-compete clauses that convert to OSS after 2-4 years. They're slightly stronger protection than AGPL in some ways.

We chose AGPL over BUSL/FSL because:

1. **AGPL is OSI-approved.** BUSL and FSL are not (the OSI considers their non-compete clauses incompatible with the Open Source Definition). For procurement reviews at enterprise customers, "OSI-approved license" is often a hard line.

2. **AGPL has 17+ years of legal precedent.** BUSL is from 2015 (HashiCorp adopted it in 2023); FSL is from 2024. Less battle-tested.

3. **AGPL works cleanly with dual-licensing.** When a customer needs an AGPL exemption, the commercial-license path is well-understood. BUSL/FSL relicensing-to-commercial paths are murkier.

4. **AGPL deters the specific threat we care about** (hyperscaler forks offering managed services) without imposing additional restrictions on individual users.

### Why not just stay Apache-2.0 and trust the moat

Honest answer: we considered it. The argument for Apache is that it maximizes adoption velocity and the moat is supposed to be in distribution + data, not code license. The argument against Apache, the one we found more persuasive, is that "distribution + data moats" only matter if you survive the first 18 months — and during those months, an Apache fork is exactly the existential threat that has killed comparable companies.

We can always relax to Apache later if no one cares about the AGPL clause. We can't tighten to AGPL after launch without burning community trust. **Start more restrictive, easier to relax** — that's the heuristic.

---

## Mechanical Q&A

**Q: My company has a "no AGPL" internal policy. What do I do?**
A: Contact `hello@modelmeld.ai` for a commercial license. Common pattern; we'll work with you.

**Q: I want to contribute a PR. Does signing a CLA mean ModelMeld can relicense my code commercially?**
A: We don't currently require a CLA on contributions because we're AGPL-only with no dual-license path active. If/when we add commercial licensing, contributors will be asked to sign a CLA going forward. Prior contributions remain under AGPL.

**Q: I want to fork modelmeld, add features, and offer my fork as a free OSS project.**
A: That's allowed and encouraged. Your fork must also be AGPL (or a compatible copyleft). You can absolutely keep the code free and open.

**Q: I want to fork modelmeld and offer it as a commercial managed service to third parties, with my own proprietary improvements I don't want to share.**
A: That triggers AGPL section 13 — your improvements must also be shared under AGPL. If you don't want that, you'd need a commercial license from us.

**Q: I'm building a startup whose product calls modelmeld over HTTP from our backend. Does AGPL affect us?**
A: No. Network-call clients are not derived works. You're fine.

**Q: Can I embed modelmeld inside my packaged desktop application?**
A: If you distribute the embedded gateway, you're distributing AGPL code — your application would need to also be AGPL (or commercially license from us). If your application calls a separately-installed gateway over HTTP, you're fine.

**Q: Does AGPL apply to the bundled `default_registry.json` data file?**
A: No. The data file is CC-BY-4.0 (see `core-engine/src/modelmeld/scout/data/LICENSE.md`). Only the Python code is AGPL.

**Q: Does AGPL apply to the live registry feed at `feed.modelmeld.ai`?**
A: No. The live feed is a subscription product with its own terms — neither AGPL nor CC-BY-4.0. The feed data is licensed only with an active subscription.

---

## If you're a lawyer reading this on behalf of a client

The relevant legal text is:
- **License file**: `core-engine/LICENSE` — verbatim GNU AGPL-3.0 from gnu.org
- **SPDX identifier on every source file**: `AGPL-3.0-or-later`
- **NOTICE file**: `core-engine/NOTICE` — covers the data exception and third-party components

ModelMeld is the copyright holder of all original modelmeld code as of the OSS launch. No external contributions had been merged before launch. The CC-BY-4.0 data file and AGPL code file are tagged separately and distributed in the same Python package; downstream packagers should preserve both tags.

For commercial-license inquiries: `hello@modelmeld.ai`.
For security disclosures: `security@modelmeld.ai`.
