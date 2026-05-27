# Governance

This document describes who makes decisions in the `modelmeld`
project and how. It exists to answer the "who do I email if X" question
that enterprise adopters ask before depending on a project, and to
make our process inspectable rather than informal.

The project is intentionally **lightweight on process** at this early
stage. As the contributor base grows, this document will evolve to
match. Significant governance changes are themselves subject to the
process described here.

---

## Decision-making model

`modelmeld` is currently operated under a **BDFL** (benevolent
dictator for life) model. The project owner makes final calls on:

- Roadmap direction + sprint scope
- Acceptance of PRs that touch architecture or public API
- Maintainer additions + removals
- Release timing + version numbers
- Code of Conduct enforcement
- Trademark + brand-use policy

This is appropriate for the project's current scale (small contributor
set, single primary maintainer). The stated intent is to **migrate to a
maintainers-by-consensus model** once we have ≥3 active maintainers
with merge rights and ≥6 months of joint activity. Migration triggers
include: substantial external contribution velocity, multiple
organizations depending on the project commercially, or the founder's
explicit step-back.

The BDFL stance is **not a hostility to disagreement**. Open a GitHub
Discussion or a comment on the relevant PR; well-argued dissent
routinely changes outcomes. The model exists to keep decisions
*timely*, not to suppress debate.

## Maintainer set

See [`MAINTAINERS.md`](MAINTAINERS.md) for the current list. Maintainers:

- Can merge PRs after the review SLA in `CONTRIBUTING.md`
- Triage incoming issues + apply labels
- Cut releases (per the release process below)
- Are responsible for the code areas they own in `CODEOWNERS`

Becoming a maintainer requires:

- A sustained track record of high-quality PRs in the project (~5+
  merged PRs over 3+ months, not a hard rule)
- Demonstrated good judgment in code review + community interactions
- Invitation by an existing maintainer
- Acceptance of [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) responsibilities

Maintainers can be removed by majority vote of other maintainers, or by
their own request. Inactivity (no project activity for 6 months without
prior notice) is also grounds for removal to emeritus status; emeritus
maintainers can be reinstated by request.

## Release process

Releases follow **semantic versioning** as documented in
[`docs/api-stability.md`](../docs/api-stability.md). Cadence is
**release-as-ready** — we do not commit to a fixed schedule. Typical
intervals during active development are 2-4 weeks between minor
versions; security fixes ship as patch versions whenever ready.

### Cutting a release

1. Open a PR titled `chore(release): vX.Y.Z` from the latest `main`
2. PR body lists the user-visible changes (release-please autogenerates
   this from Conventional Commits since the last release)
3. After review + merge, tag the merge commit `vX.Y.Z`
4. The release workflow (`.github/workflows/release.yml`) takes over:
   - Builds wheel + sdist
   - Signs artifacts via Sigstore
   - Generates CycloneDX SBOM
   - Publishes to PyPI via Trusted Publishing
   - Creates GitHub Release with signed artifacts + SBOM + signature
   - Publishes Docker image to GHCR
5. Maintainer announces the release in GitHub Discussions

### Hotfix release flow

For security or critical-bug fixes that can't wait for the next minor:

1. Branch from the last release tag (`vX.Y.Z` → `vX.Y.(Z+1)-rc`)
2. Cherry-pick the fix
3. Tag `vX.Y.(Z+1)`
4. Same release workflow as above
5. Announcement includes the CVE id (if applicable) + upgrade urgency

## How proposals work

Three escalating venues, by scope:

| Scope | Where |
|---|---|
| Minor change (bug fix, small feature, doc) | PR with description |
| Larger change (new adapter, new memory tier, API addition) | GitHub Discussion → PR |
| Architectural change (open-core boundary edit, registry feed mechanics, breaking API) | GitHub Discussion → RFC PR in `docs/rfcs/` → implementation PR |

The "RFC" tier is a written design doc reviewed before any code lands.
For pre-1.0 we don't formally require RFCs but they accelerate
architectural decisions by giving everyone the same artifact to read.

## Conflicts of interest

Maintainers disclose financial interests in dependencies, adjacent
products, or organizations whose business is affected by modelmeld
roadmap. Disclosure is informal at current scale — list them in
[`MAINTAINERS.md`](MAINTAINERS.md). When a decision touches a
maintainer's disclosed interest, they recuse from the decision.

## Trademark

The "ModelMeld" name + logo are trademarks of the owning entity. See
[`TRADEMARK.md`](TRADEMARK.md) for permitted + restricted uses. The
AGPL-3.0-or-later license on the code does not grant trademark rights.

## Changing this document

Changes to `GOVERNANCE.md` go through the same process as architectural
changes: GitHub Discussion → RFC PR → implementation PR. The BDFL has
final approval but is expected to publicly engage with substantive
objections before merging.
