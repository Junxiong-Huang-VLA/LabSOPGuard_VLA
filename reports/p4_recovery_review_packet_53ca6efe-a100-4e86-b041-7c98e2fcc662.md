# Missing-Step Review Packet: 53ca6efe-a100-4e86-b041-7c98e2fcc662

- Generated: `2026-05-08T09:03:54.097361+00:00`
- Recovery plan: `D:\LabCapability\reports\p4_missing_step_recovery_53ca6efe-a100-4e86-b041-7c98e2fcc662.json`
- Session dir: `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index`
- Target steps: `2`
- Candidate totals: video `16`, transcript `0`, assets `16`

## Review Rules

- Do not auto-approve this packet.
- Approve only when the listed clip/keyframe evidence visually supports the SOP step.
- Use transcript or text-only evidence as support, not as strong visual confirmation.
- Keep segment-level retrieval backfill as retrieval-only unless a reviewer confirms real visual process evidence.

## Step Summary

| # | Step | Action | Status | Confidence | Window | Best Video | Best Asset | Suggested |
|---|---|---|---|---:|---|---:|---:|---|
| 1 | Sample Handling | sample_handling | not_observed | 0.1 | 2026-05-07T23:21:25.056408+00:00 to 2026-05-07T23:21:55.056408+00:00 | 0.5765 | 0.49 | review_for_approval |
| 2 | Recording | recording | not_observed | 0.1 | 2026-05-07T23:21:25.056408+00:00 to 2026-05-07T23:21:55.056408+00:00 | 0.554 | 0.49 | review_for_approval |

## 1. Sample Handling

- Step ID: `step_003`
- Expected action: `sample_handling`
- Current status: `not_observed`
- Confidence: `0.1`
- Recovery reason: step is not observed in the current process artifact
- Window: `2026-05-07T23:21:25.056408+00:00 to 2026-05-07T23:21:55.056408+00:00`
- Candidate counts: video `8`, transcript `0`, assets `8`
- Reviewer hint: `review_for_approval`; video and asset candidates overlap the recovery window or expected action

### Candidate Video Events

| # | Score | Conf | Event | Object | Segment | Micro | Time | Key Clips/Frames |
|---|---:|---:|---|---|---|---|---|---|
| 1 | 0.5765 | 0.7326 | object_movement_detected | sample_bottle | seg_000003 | seg_000003_micro_001 | 2026-05-07T23:21:24.790408+00:00 to 2026-05-07T23:21:26.723408+00:00 |  |
| 2 | 0.5433 | 0.8666 | liquid_flow_candidate_visual | beaker | seg_000003 | seg_000003_micro_001 | 2026-05-07T23:21:25.056741+00:00 to 2026-05-07T23:21:29.390074+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_002\contact.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_002\peak.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_002\release.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000003_micro_002_first_person.mp4` |
| 3 | 0.5296 | 0.7979 | object_movement_detected | reagent_bottle | seg_000003 | seg_000003_micro_001 | 2026-05-07T23:21:24.790408+00:00 to 2026-05-07T23:21:26.723408+00:00 |  |
| 4 | 0.5242 | 0.4712 | liquid_flow_candidate_visual | reagent_bottle | seg_000003 | seg_000003_micro_002 | 2026-05-07T23:21:26.790075+00:00 to 2026-05-07T23:21:27.723408+00:00 |  |
| 5 | 0.4994 | 0.647 | liquid_flow_candidate_visual | container | seg_000003 | seg_000003_micro_001 | 2026-05-07T23:21:24.790408+00:00 to 2026-05-07T23:21:29.990075+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_001\contact.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_001\peak.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_001\release.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000003_micro_001_first_person.mp4` |

### Candidate Assets

| # | Score | Type | Source | Segment | Micro | Time | Path |
|---|---:|---|---|---|---|---|---|
| 1 | 0.49 | video_clip | micro_clip | seg_000003 | seg_000003_micro_001 | 2026-05-07T23:21:24.790408+00:00 to 2026-05-07T23:21:26.723408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000003_micro_001_third_person.mp4` |
| 2 | 0.49 | video_clip | micro_clip | seg_000003 | seg_000003_micro_001 | 2026-05-07T23:21:24.790408+00:00 to 2026-05-07T23:21:26.723408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000003_micro_001_first_person.mp4` |
| 3 | 0.49 | keyframe | micro_keyframe | seg_000003 | seg_000003_micro_001 | 2026-05-07T23:21:24.790408+00:00 to 2026-05-07T23:21:26.723408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_001\contact.jpg` |
| 4 | 0.49 | keyframe | micro_keyframe | seg_000003 | seg_000003_micro_001 | 2026-05-07T23:21:24.790408+00:00 to 2026-05-07T23:21:26.723408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_001\peak.jpg` |
| 5 | 0.49 | keyframe | micro_keyframe | seg_000003 | seg_000003_micro_001 | 2026-05-07T23:21:24.790408+00:00 to 2026-05-07T23:21:26.723408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_001\release.jpg` |

### Decision Template

```json
{
  "confirmation_id": "53ca6efe-a100-4e86-b041-7c98e2fcc662:step_003",
  "step_id": "step_003",
  "step_name": "Sample Handling",
  "decision": "",
  "reviewer": "",
  "note": "step_003: reviewed recovery candidates; visual_match=; transcript_support=; decision=",
  "visual_match": "",
  "transcript_support": "",
  "chosen_evidence_ids": [],
  "best_video_match_score": 0.5765,
  "best_asset_match_score": 0.49,
  "transcript_candidate_count": 0
}
```

## 2. Recording

- Step ID: `step_004`
- Expected action: `recording`
- Current status: `not_observed`
- Confidence: `0.1`
- Recovery reason: step is not observed in the current process artifact
- Window: `2026-05-07T23:21:25.056408+00:00 to 2026-05-07T23:21:55.056408+00:00`
- Candidate counts: video `8`, transcript `0`, assets `8`
- Reviewer hint: `review_for_approval`; video and asset candidates overlap the recovery window or expected action

### Candidate Video Events

| # | Score | Conf | Event | Object | Segment | Micro | Time | Key Clips/Frames |
|---|---:|---:|---|---|---|---|---|---|
| 1 | 0.554 | 0.62 | object_movement_candidate | balance | seg_000004 | seg_000004_micro_001 | 2026-05-07T23:21:41.890075+00:00 to 2026-05-07T23:21:44.456408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000004_micro_001_third_person.mp4`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000004_micro_001_first_person.mp4`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\contact.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\peak.jpg` |
| 2 | 0.5138 | 0.4191 | equipment_panel_operation_candidate | balance | seg_000003 | seg_000003_micro_004 | 2026-05-07T23:21:25.856741+00:00 to 2026-05-07T23:21:29.990075+00:00 |  |
| 3 | 0.4933 | 0.9167 | experiment_action_classification | balance | seg_000004 | seg_000004_micro_001 | 2026-05-07T23:21:41.890075+00:00 to 2026-05-07T23:21:44.456408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\contact.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\peak.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\release.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000004_micro_001_first_person.mp4` |
| 4 | 0.4933 | 0.9167 | hand_object_contact | balance | seg_000004 | seg_000004_micro_001 | 2026-05-07T23:21:41.890075+00:00 to 2026-05-07T23:21:44.456408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\contact.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\peak.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\release.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000004_micro_001_first_person.mp4` |
| 5 | 0.4933 | 0.9167 | object_state_change | balance | seg_000004 | seg_000004_micro_001 | 2026-05-07T23:21:41.890075+00:00 to 2026-05-07T23:21:44.456408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\contact.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\peak.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000004_micro_001\release.jpg`<br>`D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000004_micro_001_first_person.mp4` |

### Candidate Assets

| # | Score | Type | Source | Segment | Micro | Time | Path |
|---|---:|---|---|---|---|---|---|
| 1 | 0.49 | video_clip | micro_clip | seg_000003 | seg_000003_micro_004 | 2026-05-07T23:21:29.223408+00:00 to 2026-05-07T23:21:30.056408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000003_micro_004_third_person.mp4` |
| 2 | 0.49 | video_clip | micro_clip | seg_000003 | seg_000003_micro_004 | 2026-05-07T23:21:29.223408+00:00 to 2026-05-07T23:21:30.056408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\clips\micro\seg_000003_micro_004_first_person.mp4` |
| 3 | 0.49 | keyframe | micro_keyframe | seg_000003 | seg_000003_micro_004 | 2026-05-07T23:21:29.223408+00:00 to 2026-05-07T23:21:30.056408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_004\contact.jpg` |
| 4 | 0.49 | keyframe | micro_keyframe | seg_000003 | seg_000003_micro_004 | 2026-05-07T23:21:29.223408+00:00 to 2026-05-07T23:21:30.056408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_004\peak.jpg` |
| 5 | 0.49 | keyframe | micro_keyframe | seg_000003 | seg_000003_micro_004 | 2026-05-07T23:21:29.223408+00:00 to 2026-05-07T23:21:30.056408+00:00 | `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index\keyframes\micro\seg_000003_micro_004\release.jpg` |

### Decision Template

```json
{
  "confirmation_id": "53ca6efe-a100-4e86-b041-7c98e2fcc662:step_004",
  "step_id": "step_004",
  "step_name": "Recording",
  "decision": "",
  "reviewer": "",
  "note": "step_004: reviewed recovery candidates; visual_match=; transcript_support=; decision=",
  "visual_match": "",
  "transcript_support": "",
  "chosen_evidence_ids": [],
  "best_video_match_score": 0.554,
  "best_asset_match_score": 0.49,
  "transcript_candidate_count": 0
}
```
