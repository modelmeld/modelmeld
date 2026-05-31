# Codex CLI / ChatGPT Subscription Passthrough — Feasibility Memo

**Author:** Research pass for ModelMeld Sprint 5.5 scoping
**Date:** 2026-05-30
**Status:** Research-only; no code written.

---

## 1. TL;DR

**Technically feasible, and arguably *less* gray-area than the Claude Max passthrough.** OpenAI's Codex CLI authenticates to a non-public endpoint at `https://chatgpt.com/backend-api/codex/responses` using OAuth bearer tokens cached in `~/.codex/auth.json`. Unlike Anthropic — which in Jan–Feb 2026 explicitly banned third-party OAuth reuse of Claude Pro/Max — OpenAI has issued public-facing endorsement of subscription reuse across third-party tooling (Romain Huet: *"We want people to be able to use Codex, and their ChatGPT subscription, wherever they like"*). However, the endpoint is undocumented, has no SLA, and OpenAI's official guidance still steers programmatic workflows toward API keys, so the safe shape is identical to Claude Max: self-host only, headers preserved verbatim, opt-in / power-user framing, no hosted multi-tenant offering.

---

## 2. Codex CLI auth model

The Codex CLI (https://github.com/openai/codex) supports **three** auth methods, selectable at login:

- **Sign in with ChatGPT** (default; browser OAuth) — used to grant Codex access tied to a ChatGPT Plus/Pro/Business/Edu/Enterprise plan. Tokens auto-refresh.
- **API key** (`printenv OPENAI_API_KEY | codex login --with-api-key`) — standard `sk-…` API pricing.
- **Access token / device-code** (`codex login --device-auth`, or pipe `CODEX_ACCESS_TOKEN` to `codex login --with-access-token`) — headless/CI variant of the OAuth path.

Credentials are cached at **`~/.codex/auth.json`** (or in OS keychain if `cli_auth_credentials_store=keyring`; `CODEX_HOME` overrides the directory). The docs warn: *"Treat `~/.codex/auth.json` like a password."* When a ChatGPT-authenticated session calls the model, the CLI hits **`https://chatgpt.com/backend-api/codex/responses`**, not `api.openai.com`. API-key sessions go through the documented `api.openai.com` surface and pay standard API rates.

Method selection: `forced_login_method` config restricts available options to `"chatgpt"` or `"api"`; otherwise the CLI picks the cached session, preferring ChatGPT when both are available (this has caused user-reported confusion — see `openai/codex` issues #2733 and #3286).

Sources:
- https://developers.openai.com/codex/auth
- https://developers.openai.com/codex/cli/reference
- https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan
- https://help.openai.com/en/articles/11381614 (Codex CLI + Sign in with ChatGPT)
- https://github.com/openai/codex/issues/3820 (headless auth request)

---

## 3. ChatGPT subscription → API gap

The historical answer is **still "no" for `api.openai.com`** — ChatGPT Plus/Pro subscriptions do not grant programmatic access to the documented OpenAI API and have never shipped a bearer-token bridge to `api.openai.com/v1/*`. API and subscription billing remain entirely separate ([AIonX guide](https://aionx.co/chatgpt-reviews/chatgpt-plus-api-access/); OpenAI help docs).

**However**, a parallel surface has emerged: the Codex CLI backend at `chatgpt.com/backend-api/codex/responses` *is* a subscription-authenticated model endpoint, just not via `api.openai.com`. The April 2026 GPT-5.5 launch made this concretely visible — GPT-5.5 was initially exposed *only* via the Codex backend, not the public API ([Daniel Vaughan analysis](https://codex.danielvaughan.com/2026/04/24/codex-subscription-api-programmatic-access-gpt-5-5-chatgpt-plan/); [Simon Willison, Apr 23 2026](https://simonwillison.net/2026/Apr/23/gpt-5-5/)). Simon Willison's `llm-openai-via-codex` plugin (https://github.com/simonw/llm-openai-via-codex) reads `~/.codex/auth.json` and forwards prompts to that endpoint with the same headers Codex CLI uses.

So the picture is: subscription → standard OpenAI API = still gapped; subscription → Codex backend = exists, undocumented, no SLA, but reachable and (currently) blessed.

---

## 4. ToS comparison: OpenAI vs Anthropic

The comparison flipped over the last 6 months in our favor on OpenAI:

**Anthropic (more restrictive):** On 2026-01-09, Anthropic began blocking Max OAuth tokens in third-party clients. The 2026-02-19 policy update made it explicit: *"Using OAuth tokens obtained through Claude Free, Pro, or Max accounts in any other product, tool, or service — including the Agent SDK — is not permitted."* OAuth is permitted only via Claude Code and Claude.ai itself. OpenClaw, OpenCode, and others had tokens revoked; at least one developer reports an account ban ([The Register, 2026-02-20](https://www.theregister.com/2026/02/20/anthropic_clarifies_ban_third_party_claude_access/); [Winbuzzer, 2026-02-19](https://winbuzzer.com/2026/02/19/anthropic-bans-claude-subscription-oauth-in-third-party-apps-xcxwbn/); [StackSweep analysis](https://www.engineerscodex.com/anthropic-claude-subscription-switcharoo/)). Our Sprint 5 Claude Max plan therefore sits in a gray zone — *self-host-only verbatim-header passthrough* is not explicitly addressed, and Anthropic engineer Thariq Shihipar's only public clarification distinguishes "personal experimentation" (OK) from "revenue-generating products" (not OK). The boundary is undefined.

**OpenAI (more permissive — for now):** OpenAI has not banned third-party use of Codex OAuth tokens. Romain Huet publicly endorsed third-party usage *("…wherever they like — in the app, in the terminal, but also in JetBrains, Xcode, OpenCode, Pi, and now Claude Code")*. Codex CLI itself is Apache 2.0 and the auth flow is intentionally local. **Caveats:**
- OpenAI's account-sharing policy still prohibits sharing credentials between users — meaning **multi-tenant pooling is clearly off-limits**, same as Anthropic.
- Codex docs *recommend API key for programmatic workflows* — implicit signal that ChatGPT-auth reuse is tolerated but not endorsed for automation.
- The `chatgpt.com/backend-api/codex/responses` endpoint is undocumented and can change without notice. No SLA.
- OpenAI's services agreement still prohibits circumventing rate limits and reverse-engineering protected APIs — a hosted account-pool would clearly violate this, but a self-hosted local forwarder is consistent with the public Huet quote.

**Net comparison:** A self-host-only, headers-preserved, single-user-per-instance Codex passthrough is **lower ToS risk** than the analogous Claude Max passthrough — OpenAI has a public endorsement we can point to; Anthropic has an explicit prohibition. Both are unsafe to host as a multi-tenant service.

---

## 5. Recommendation

**Pursue Sprint 5.5 — but only after Sprint 5 ships and proves the pattern, and gate it on the same self-host-only posture.**

**Rough scope:**
- New adapter: `OpenAIChatGPTAdapter` (or extend `OpenAIAdapter` with auth-mode detection).
- Detect bearer-OAuth vs `sk-…` in incoming auth header. OAuth path → forward to `https://chatgpt.com/backend-api/codex/responses` preserving all headers verbatim. API-key path → existing `api.openai.com/v1/responses` (or `/v1/chat/completions`) behavior.
- Customer points their `OPENAI_BASE_URL` (or Codex CLI config equivalent) at the self-hosted gateway; gateway forwards. No persisted token state on our infra — same passthrough rule as Claude Max.
- Document a one-page setup guide pointing to Simon Willison's plugin as prior-art evidence.

**ToS posture (mirror Sprint 5):**
- Self-host only; explicitly excluded from Hosted Tier.
- Documented as power-user opt-in, not headline marketing.
- README disclaimer: undocumented endpoint, no SLA, single-user only, no pooling, can break without notice.
- Cite the Huet endorsement quote in our docs for posterity.

**No concrete blocker** — but two pre-flight items:
1. **Confirm the endpoint surface accepts the `responses` API shape** our `OpenAIAdapter` already speaks. Skim `simonw/llm-openai-via-codex` source for the exact request envelope (notably the `"instructions"` field Willison mentions) — likely a small adapter tweak rather than a parallel implementation.
2. **Monitor OpenAI ToS drift.** Anthropic's stance flipped in ~6 months. We should land Sprint 5.5 with a kill-switch flag (`ENABLE_CHATGPT_OAUTH_PASSTHROUGH=false` default) so we can disable it cleanly if OpenAI follows Anthropic's lead.

---

## Sources

- https://developers.openai.com/codex/auth
- https://developers.openai.com/codex/cli
- https://developers.openai.com/codex/cli/reference
- https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan
- https://help.openai.com/en/articles/11381614
- https://github.com/openai/codex
- https://github.com/openai/codex/issues/3820
- https://github.com/openai/codex/issues/2733
- https://github.com/openai/codex/issues/3286
- https://github.com/simonw/llm-openai-via-codex
- https://simonwillison.net/2026/Apr/23/gpt-5-5/
- https://simonwillison.net/2025/Nov/9/gpt-5-codex-mini/
- https://codex.danielvaughan.com/2026/04/24/codex-subscription-api-programmatic-access-gpt-5-5-chatgpt-plan/
- https://www.theregister.com/2026/02/20/anthropic_clarifies_ban_third_party_claude_access/
- https://winbuzzer.com/2026/02/19/anthropic-bans-claude-subscription-oauth-in-third-party-apps-xcxwbn/
- https://www.engineerscodex.com/anthropic-claude-subscription-switcharoo/
- https://help.openai.com/en/articles/10471989-openai-account-sharing-policy
- https://aionx.co/chatgpt-reviews/chatgpt-plus-api-access/

## Unanswerable from public sources

- OpenAI's verbatim ToS clause about non-public/backend APIs (`openai.com/policies/*` returned 403 to WebFetch; would need direct browser fetch to quote).
- Whether OpenAI has any private guidance to enterprise customers about programmatic Codex-backend reuse beyond the Huet tweet.
- Whether Anthropic has *explicitly* enforced against pure passthrough gateways (vs full token-extracting clients like OpenClaw). Public coverage doesn't distinguish.
