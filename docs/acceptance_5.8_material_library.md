# 5.8 Material Library Acceptance Record

Date: 2026-05-09

## Baseline

- Root repo `LabEmbodied` merged PR #1 into `main`.
  - Merge commit: `30de15f`
  - Handoff archive commit: `128f84d`
- Nested repo `LabEmbodied` merged PR #1 into `main`.
  - Merge commit: `b3ee419`
  - Follow-up validated commits: `5460f43`, `2153ebd`
- Baseline tag name for both repos: `v5.8-material-library-accepted`

## Validation

- Root `python -m pytest -q`: `224 passed`
- Root `python -m compileall -q src LabSOPGuard\backend tests`: passed
- Root `scripts/check_project_scope.ps1`: passed
- LabSOPGuard `python -m pytest -q`: `206 passed`
- LabSOPGuard `python -m compileall -q backend src`: passed
- Frontend `npm run build`: passed
- Frontend `npm test -- --run`: `14 passed`
- Material retrieval golden set strict mode:
  - `python scripts/evaluate_material_retrieval_59.py --fixture-json tests/fixtures/material_library_58_acceptance.json --top-k 5 --strict --min-canonical-hit-rate 0.95 --min-top-k-hit-rate 0.95 --min-top1-hit-rate 0.75`
  - Result: passed (`canonical_hit_rate=1.0`, `top_k_hit_rate=1.0`, `top1_hit_rate=0.9667`)
- Frontend 5.8 smoke on mock data:
  - `cd frontend; npm run smoke:materials58:mock`
  - Result: passed
- LabSOPGuard latest `main` GitHub smoke CI: passed (backend fixture + frontend Playwright smoke)

## Runtime Acceptance

- Formal material library opened at `/materials`.
- Candidate review opened at `/materials/review` through the experiment review route.
- Formal key material count: `74`.
- Canonical action groups present:
  - `hand-paper`
  - `hand-bottle`
  - `hand-spatula`
  - `hand-balance`
  - `hand-container`
- Professional PDF artifacts are kept under `专业报告`.
- Professional PDF artifacts do not appear in the formal key material grid.
- Default candidate review queue excludes rejected, deferred, and not-selected candidates.
- Current 5.8 run showed `pending_total: 0`.

## Stash Review

- Root stash list (`git stash list`) is currently empty.
- Nested `LabSOPGuard` stash list (`git stash list`) is currently empty.
- Both repos already have release rollback tags:
  - `v5.8-material-library-accepted`
