# alignment_health.json Schema

**Location:** `{output_dir}/metadata/alignment_health.json`

## Description

Sliding window drift estimation output with EMA-smoothed offsets, drift windows, alerts, and health summary. Computed by `estimate_sliding_window_drift()`.

## Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `smoothed_offsets` | array | Per-anchor EMA-smoothed offset history |
| `drift_windows` | array | Sliding window drift measurements |
| `alerts` | array | Drift threshold violation alerts |
| `stable_window` | object\|null | Window with lowest jitter (best local alignment) |
| `summary` | object | Aggregate health metrics |

## summary Fields

| Field | Type | Description |
|-------|------|-------------|
| `mean_offset_ms` | float | Mean time offset in milliseconds |
| `jitter_ms` | float | Mean inter-offset variation in ms |
| `drift_events` | int | Number of windows exceeding threshold |
| `max_drift_sec` | float | Maximum single-window drift |
| `status` | string | `"healthy"`, `"warning"`, `"drift_alert"`, or `"no_data"` |
| `window_size` | int | Sliding window size used |
| `smoothing_alpha` | float | EMA smoothing factor used |
| `alert_threshold_sec` | float | Drift alert threshold used |

## Status Values

- **`healthy`**: All windows below threshold, max_drift < 70% of threshold
- **`warning`**: No alerts but max_drift >= 70% of threshold
- **`drift_alert`**: At least one window exceeds threshold
- **`no_data`**: Empty offset_history input

## Alert Format

```json
{
  "type": "drift_threshold_exceeded",
  "window_start": 3,
  "window_end": 7,
  "drift_sec": 1.82,
  "threshold_sec": 1.5
}
```

## Degradation Behavior

When `status == "drift_alert"`, the pipeline applies confidence degradation:
- `segment.final_score *= alignment_degradation_factor` (default 0.85)
- `segment.alignment_report.degraded = true`
- Original timestamps are preserved (no correction applied)
