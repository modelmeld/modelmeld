# Roadmap

This is the **public-facing** roadmap for `modelmeld`. The internal
detail (sprint-level task lists, sequencing decisions, etc.) lives in
the project plan and is not committed to this repository.

We commit to **themes**, not dates. The "next few releases" section is
the most concrete; longer-range items are directional.

## Current focus

- **Public OSS launch readiness.** Polish the open-source surface to
  the point where a stranger landing on the repo can install, run a
  routed completion, and read what they need to subscribe to the
  registry feed in under five minutes.
- **Real-world dogfooding.** Continuous routing of the project team's
  own AI-assisted coding traffic through the gateway, with the data
  feeding the next round of default-weight tuning.

## Next few releases

Themes likely to land within the next 2–3 minor versions. Subject to
revision based on contributor feedback + dogfooding signal.

- **Postgres-backed memory store.** OSS users currently get the
  in-process `InMemoryMemoryStore`; a Postgres implementation lands
  next, behind an optional install extra so the default install stays
  light.
- **Google Gemini adapter.** Capability registry already lists Gemini
  models. Only the wire-format adapter is missing.
- **Streaming-failover refinements.** Better handling when an upstream
  fails *mid-stream* (vs at connection time). Today's behavior is
  correct but not graceful; tokens already delivered to the client
  can't be unsent, so the failover behavior on streams is constrained.
- **Improved task-category classifier.** The current regex-based
  classifier has known coverage gaps (e.g. `"reason through"` vs
  `"reason about"`). A lighter-weight statistical classifier is the
  most likely successor.
- **Richer integration guides.** Cursor / Claude Code / Aider / Cline
  guides to join the existing AutoGen / CrewAI / LangGraph / OpenClaw
  set.

## Future themes (directional)

- **Multi-region routing.** Route to the closest healthy backend when
  latency matters (e.g. interactive coding flows vs batch
  summarization).
- **Speculative execution.** Issue requests to multiple backends in
  parallel for low-latency paths, take whichever returns first.
  Bounded by configured per-request cost ceiling.
- **Quality regression detection.** Continuous monitoring that flags
  when a backend's outputs on a given task category drift below the
  capability registry's expectation. Closes the loop on the curation
  pipeline.
- **Open framework integrations.** Beyond agent-OS frameworks, work
  with the maintainers of editor extensions, browser AI assistants,
  and CI/CD bots to add native gateway awareness.

## Explicitly NOT on the roadmap

These are out of scope and not under reconsideration. The full
rationale is in the project plan; short version:

- **Generic cloud-to-cloud proxy.** Our focus is routing intelligence
  for coding-tool deployments, not generic provider-to-provider
  forwarding.
- **Autonomous agent framework.** We integrate with the major
  frameworks; we don't ship one. Building an agent runtime would
  fragment focus.
- **Model fine-tuning UX.** Out of scope. We route between models;
  others train them.
- **Markup-resale of upstream inference providers.** We host
  open-weights directly via vLLM where hosting is appropriate; cloud
  calls go to the customer's own upstream account with their own key.
- **FedRAMP / HIPAA / classified deployments.** Different sales motion
  with different compliance requirements; not where we focus today.
- **Custom embeddings / RAG infrastructure.** We're a chat-completions
  gateway. RAG layers above us; embeddings infrastructure below.

## How to influence the roadmap

- **Open a GitHub Discussion** before opening an issue for a larger
  proposal. Discussions are the right venue for "should we do X?"
  questions; issues are for "here's a concrete bug or
  well-scoped feature."
- **Architectural proposals** go through the RFC tier described in
  `GOVERNANCE.md` — a written design doc reviewed before any code
  lands.
- **Contributions land features faster.** A well-scoped PR with tests
  reaches users in a release; a feature request waits in a queue.
  See `CONTRIBUTING.md`.

## Versioning + stability

See [`docs/api-stability.md`](../docs/api-stability.md) for which
symbols carry SemVer compatibility commitments. Pre-1.0 we may break
things; we'll call it out in the changelog when we do.
