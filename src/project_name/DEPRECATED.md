# DEPRECATED

This module (`src/project_name/`) is deprecated and will be removed in future versions.

## Migration Path

All functionality has been migrated to `src/experiment/`.

### Module Mapping

| Old Module | New Module | Status |
|------------|------------|--------|
| `project_name.detection.*` | `experiment.vlm_client` | ✅ Migrated |
| `project_name.monitoring.*` | Removed (not part of core product) | ⚠️ Deprecated |
| `project_name.video.*` | `experiment.service` (VideoFrameExtractor) | ✅ Migrated |
| `project_name.report.*` | Removed (not part of core product) | ⚠️ Deprecated |

## Remaining Dependencies

`backend/main.py` still imports:
- `project_name.detection.multi_level_detector` (lines 44-46)
- `project_name.monitoring.sop_engine` (lines 71-73)
- `project_name.video.capture` (lines 88-90)
- `project_name.report.pdf_report` (lines 99-101)

These will be removed in the next refactor phase.

## Timeline

- **Current**: Marked as deprecated
- **Next release**: Remove all imports from `backend/main.py`
- **Future**: Delete entire `src/project_name/` directory

---

**Do not add new code to this module.**
