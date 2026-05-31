# Worker C Taskbook Acceptance Runbook

Date: 2026-05-20

Owner lane: Worker C, validation scripts and handoff records only.

Write scope: `D:\LabCapability\scripts`, `D:\LabCapability\docs`, and `D:\LabCapability\docs\project_plans`. This lane does not modify `src`, `tests`, `LabSOPGuard\backend`, or `LabSOPGuard\frontend-app`.

## Purpose

This runbook turns the implementation taskbook acceptance commands into a replayable PowerShell workflow:

- Python compilation: `python -m compileall -q src`
- Python regression suite: `python -m pytest -q`
- frontend build: `npm run build` under `LabSOPGuard\frontend-app`
- optional evidence package validation: `evidence-package-validate --package-root <package> [--strict]`
- optional evidence package evaluation: `evidence-package-eval --package-root <package> --queries <queries> [--output <output>] [--limit <n>]`

The script preserves the project boundary from `AGENTS.md`: LabCapability owns long dual-view evidence extraction, time alignment, physical evidence, evidence packages, retrieval, and eval. It does not productize PTZ, DeepStream deployment, or multi-camera SLA validation.

## Script

Main entrypoint:

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
.\scripts\run_taskbook_acceptance.ps1 -DryRun
```

With evidence package checks:

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
.\scripts\run_taskbook_acceptance.ps1 `
  -PackageRoot 'D:\LabCapability\data\sessions\<experiment>\material_references' `
  -EvidencePackageStrict `
  -EvalQueries 'D:\LabCapability\data\sessions\<experiment>\reports\evidence_package_eval_queries.json' `
  -EvalOutput 'D:\LabCapability\data\sessions\<experiment>\reports\evidence_package_eval.json'
```

Useful options:

| Option | Use |
| --- | --- |
| `-DryRun` | Print commands and record planned steps without executing them. |
| `-RecordPath <json>` | Write a machine-readable execution record. |
| `-PythonExe <path>` | Use a specific Python, for example the LabSOPGuard conda env. |
| `-CompileTargets <paths>` | Override compile targets; default is `src`. |
| `-PytestArgs <args>` | Override pytest args; default is `-q`. |
| `-FrontendDir <path>` | Override frontend directory; default is `LabSOPGuard\frontend-app`. |
| `-PackageRoot <path>` | Enables evidence-package validation and, with `-EvalQueries`, eval. |
| `-EvidencePackageStrict` | Adds `--strict` to `evidence-package-validate`. |
| `-EvalQueries <json>` | Query fixture for `evidence-package-eval`. |
| `-EvalOutput <json>` | Optional eval output path. |
| `-EvalLimit <n>` | Optional eval query limit. |
| `-SkipCompileAll`, `-SkipPytest`, `-SkipFrontendBuild` | Skip expensive baseline steps. |
| `-SkipEvidencePackageValidate`, `-SkipEvidencePackageEval` | Skip package-specific checks. |
| `-ContinueOnError` | Continue collecting later step results after a failure. |

Dry-run with an execution JSON:

```powershell
.\scripts\run_taskbook_acceptance.ps1 `
  -DryRun `
  -PackageRoot 'D:\LabCapability\data\sessions\<experiment>\material_references' `
  -EvidencePackageStrict `
  -EvalQueries 'D:\LabCapability\data\sessions\<experiment>\reports\evidence_package_eval_queries.json' `
  -EvalOutput 'D:\LabCapability\data\sessions\<experiment>\reports\evidence_package_eval.json' `
  -RecordPath 'docs\project_plans\worker_c_acceptance_record.<date>.json'
```

## Execution Record Template

Copy this section into the concrete run note and replace placeholders.

| Field | Value |
| --- | --- |
| Date/time | `<YYYY-MM-DD HH:mm timezone>` |
| Operator/thread | `Worker C` |
| Repo root | `D:\LabCapability` |
| Git revision | `<commit or dirty-worktree note>` |
| Python | `<python --version and path>` |
| Node/npm | `<node --version / npm --version>` |
| Package root | `<path or not run>` |
| Eval queries | `<path or not run>` |
| External services/GPU | `<available / unavailable / not required>` |

| Step | Command | Status | Evidence/output | Notes |
| --- | --- | --- | --- | --- |
| compileall | `python -m compileall -q src` | `<passed/failed/skipped>` | `<stdout/stderr or record path>` |  |
| pytest | `python -m pytest -q` | `<passed/failed/skipped>` | `<passed count or first failure>` |  |
| frontend build | `npm run build` | `<passed/failed/skipped>` | `<build summary>` |  |
| evidence-package-validate | `python -m key_action_indexer.cli evidence-package-validate --package-root <package> --strict` | `<passed/failed/skipped>` | `<errors/warnings count>` |  |
| evidence-package-eval | `python -m key_action_indexer.cli evidence-package-eval --package-root <package> --queries <queries> --output <output>` | `<passed/failed/skipped>` | `<query_count/rates>` |  |

## Real Issue List Template

Use concrete evidence. Keep productization and external acceptance gaps visible but separate from LabCapability code regressions.

| ID | Symptom | Evidence | Likely cause | Fix status | Next action | Scope |
| --- | --- | --- | --- | --- | --- | --- |
| RI-001 | `<what failed or looked wrong>` | `<command output, file path, timestamp, segment ID, eval result, or screenshot>` | `<technical hypothesis>` | `<fixed/mitigated/needs data/needs GPU/needs product decision/out of scope>` | `<one concrete step>` | `<LabCapability / productization / external acceptance>` |
| RI-002 | `<example: evidence-package eval misses target object>` | `<query id and top results>` | `<retrieval or evidence gap>` | `needs data` | `<add or review targeted evidence>` | `LabCapability` |
| RI-EXT-PTZ | PTZ acceptance is not part of LabCapability core validation. | `AGENTS.md` points PTZ work to `D:\PtzTracker`. | Separate product surface and hardware/control stack. | `out of scope` | Validate PTZ in the PTZ tracker project. | `external acceptance` |
| RI-EXT-DS | DeepStream runtime/SLA acceptance is not part of this taskbook script. | No DeepStream deployment contract is owned by this repository lane. | Productized GPU/video service integration needs deployment-specific checks. | `needs product decision` | Define DeepStream service contract, hardware, sample streams, and pass/fail SLA outside this lane. | `productization` |
| RI-EXT-MCAM | Multi-camera SLA/five-camera orchestration is outside LabCapability core validation. | `AGENTS.md` points multi-camera monitoring to `D:\MultiCameraMonitor`. | Separate camera orchestration and monitoring product. | `out of scope` | Validate multi-camera SLA in the monitor/product acceptance plan. | `external acceptance` |

## Scope Notes

- LabCapability acceptance can fail on evidence package schema, time alignment, physical change log, retrieval, or eval quality.
- PTZ belongs to `D:\PtzTracker`.
- DeepStream deployment, throughput, latency, and hardware SLA belong to productization or external deployment acceptance.
- Multi-camera SLA, five-camera orchestration, wireless video SDKs, camera proxy/streaming, and monitor recording belong to `D:\MultiCameraMonitor` or external acceptance.
- A frontend `npm run build` check here is regression coverage only; it is not a request to expand complex frontend scope.

## Completed Worker C Checks

- Confirmed CLI help for `evidence-package-validate`, `evidence-package-eval`, and `acceptance-pipeline`.
- Added `scripts\run_taskbook_acceptance.ps1` with `-DryRun`, skip flags, evidence package parameters, and optional JSON record output.
- This runbook provides the execution record and real issue list template for future concrete runs.
