# Contributing to ModelMeld

Thanks for considering a contribution. This document covers everything from
"how do I set up my dev environment" to "how do I get my PR reviewed quickly."

## TL;DR

1. Fork the repo, branch off `main`
2. `pip install -e ".[dev,openai,anthropic,tokenizer]"`
3. Run `ruff format && ruff check && pyright src/modelmeld` before pushing
4. Make your change
5. Add or update tests (`pytest` must pass; coverage must not drop)
6. Commit with a `Signed-off-by:` trailer (`git commit -s`) — required (see DCO below)
7. Open a PR using the template; fill in the test plan

We aim to first-respond on every PR within 7 days. Triage labels (`bug`,
`enhancement`, `good-first-issue`, `needs-design`) are applied as part of
that response.

## Dev environment

Reproducible toolchain (pin these or our CI matrix may not match yours):

| Tool | Pinned version | Why |
|---|---|---|
| Python | 3.10, 3.11, or 3.12 | CI matrix |
| `pip` | latest | — |
| `uv` (optional) | latest | Used for hashed lockfile generation; not required for daily dev |

Setup:

```bash
git clone https://github.com/modelmeld/modelmeld.git
cd modelmeld
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev,openai,anthropic,tokenizer]"
```

Run the test suite:

```bash
pytest                              # full suite
pytest tests/test_router_tiered.py  # one file
pytest -k "anthropic"               # one substring
pytest --cov=modelmeld           # with coverage
```

The full suite takes ~30 seconds on a modern laptop. There are no
external-service dependencies for unit tests; integration tests against
real providers are gated by env vars (see `scripts/smoke_anthropic.py`
for the pattern).

## Code style

Enforced by CI:

- **Formatting:** `ruff format` (drop-in compatible with `black`)
- **Lint:** `ruff check`
- **Types:** `pyright` on `src/modelmeld/` (test code is type-checked best-effort)

Run all three locally before opening a PR:

```bash
ruff format src tests
ruff check src tests
pyright src/modelmeld
```

We do not currently enforce a per-line-length limit beyond what `ruff
format` decides. Use prose-rather-than-comments for the *why*; let
well-named identifiers handle the *what*.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>(<scope>): <short summary>

<body — one or more paragraphs explaining the WHY>

<footer — e.g. Fixes #123>
```

Types we use: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`,
`ci`, `build`.

The body should explain *why* you made the change — the *what* is in
the diff. If you're fixing a bug, name the root cause; if you're adding a
feature, name the motivating use case.

## Developer Certificate of Origin (DCO)

We use [DCO](https://developercertificate.org) instead of a Contributor
License Agreement (CLA). Practically, this means every commit needs a
`Signed-off-by:` trailer:

```
Signed-off-by: Your Name <your.email@example.com>
```

Add it automatically with `git commit -s` (or `git commit --signoff`).
CI rejects PRs whose commits are missing the trailer.

The DCO is a developer's lightweight assertion that they wrote (or have
the right to contribute) the code. It does not assign copyright; you keep
your contributions under your name. Read the full text at
<https://developercertificate.org>.

If a commit lacks the trailer, fix it with:

```bash
git commit --amend --signoff               # last commit only
git rebase HEAD~N --signoff                # last N commits
```

### Dual-licensing implication

ModelMeld offers a commercial license alongside AGPL-3.0-or-later for
customers who can't comply with AGPL's network-copyleft terms (typically
companies embedding the gateway in closed-source products or offering
managed services with proprietary modifications). The DCO grants us the
right to relicense your contributions under that commercial license
alongside the AGPL.

In practice this means: if you contribute, your code stays attributed
to you under the AGPL, but the project's maintainers may also offer it
to commercial customers under different terms. This is standard
practice for dual-licensed projects (Sentry, Grafana, MongoDB Community
Server use the same pattern with DCO + dual-licensing). It does **not**
give us copyright assignment — you keep your copyright.

If you're not comfortable with this, please don't contribute; or open
a GitHub Discussion to talk through alternatives. See `NOTICE` for the
canonical statement, or email `hello@modelmeld.ai` with questions.

## Pull requests

Use the PR template (`.github/PULL_REQUEST_TEMPLATE.md`). The minimum
viable PR:

- [ ] Tests added or updated (and they actually exercise the change)
- [ ] Docs updated if user-facing behavior changed
- [ ] `ruff format && ruff check && pyright src/modelmeld` pass
- [ ] `pytest` passes
- [ ] All commits have DCO signoff
- [ ] PR description explains the *why*, not just the *what*

**Small PRs get reviewed faster.** If your change is large, consider
splitting it across multiple PRs that each leave the codebase in a
working state.

### Review SLA

- **First response:** within 7 days of PR open (triage label + initial
  feedback or "looks good, will review properly this week")
- **Merge decision:** within 14 days of last PR update, assuming all
  CI is green and feedback is addressed
- These are *targets*, not guarantees. Maintainers are humans with day
  jobs. If a PR has stalled for over 14 days without activity, please
  ping in the PR or open a thread in GitHub Discussions.

## Finding something to work on

Look for issues labeled [`good first issue`](https://github.com/ModelMeld/modelmeld/issues?q=is%3Aopen+label%3A%22good+first+issue%22)
— these are intentionally scoped to be self-contained and well-defined.

If you have an idea that isn't already an issue, open a GitHub Discussion
first to gauge interest before sinking a weekend into it. Some changes
(new adapters, new memory tiers, framework integrations) are easy yeses;
others (major refactors of the routing layer, changes to the open-core
boundary, registry data file edits) need more discussion.

## What we *don't* accept PRs for

The bundled snapshot data files in `src/modelmeld/scout/data/` are
curated centrally. If you see a model score that looks wrong, file an
issue and we'll evaluate the adjustment for the next feed release. The
reason: routing decisions are global, and uncoordinated edits across
contributors would produce inconsistent behavior across forks.

## Reporting bugs

See the issue templates in `.github/ISSUE_TEMPLATE/`. For security-
sensitive issues, **do not file a public issue** — see
[`SECURITY.md`](SECURITY.md).

## AI-assisted contributions

Use whatever tools help you produce good work — including AI coding
assistants. The quality bar is the same regardless:

- Every line you submit is yours to understand, test, and defend in
  review.
- Reviewers will ask substantive questions about non-obvious changes.
  If you can't explain a design choice, the PR isn't ready.
- PRs that read as raw model output — unexplained refactors,
  hallucinated APIs, test changes the author can't justify — will be
  sent back for human judgment.

We don't require attribution trailers for AI-assisted work. We care
whether the change is correct, well-tested, and consistent with the
codebase, not how it was produced.

## Code of Conduct

We follow the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). Report
violations to `hello@modelmeld.ai`. The maintainer team handles
reports confidentially.

## Questions?

GitHub Discussions for anything that isn't a bug or a feature request.
For commercial / enterprise inquiries, contact `hello@modelmeld.ai`.

Thanks for contributing.
