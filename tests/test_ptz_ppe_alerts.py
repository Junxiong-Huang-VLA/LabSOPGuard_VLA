import pytest

pytest.skip("PTZ tracker tests moved to D:\\PtzTracker scope.", allow_module_level=True)

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.ptz_tracker_streaming import PtzTrackerService
from ptz_tracker.detector import ComplianceResult


def test_ptz_ppe_alert_detects_missing_lab_coat_and_gloves_with_cooldown():
    service = PtzTrackerService()
    comp = ComplianceResult(
        has_lab_coat=False,
        has_gloves=False,
        lab_coat_conf=0.12,
        gloves_conf=0.08,
    )

    assert service._update_alert_state(comp, 1000.0) == []
    assert service._update_alert_state(comp, 1002.0) == []

    first = service._update_alert_state(comp, 1004.0)
    assert len(first) == 1
    assert first[0].types == ["no_lab_coat", "no_gloves"]

    assert service._update_alert_state(comp, 1030.0) == []

    second = service._update_alert_state(comp, 1065.0)
    assert len(second) == 1
    assert second[0].types == ["no_lab_coat", "no_gloves"]


def test_ptz_ppe_alert_detects_missing_gloves_only():
    service = PtzTrackerService()
    comp = ComplianceResult(
        has_lab_coat=True,
        has_gloves=False,
        lab_coat_conf=0.91,
        gloves_conf=0.05,
    )

    service._update_alert_state(comp, 2000.0)
    service._update_alert_state(comp, 2002.0)
    events = service._update_alert_state(comp, 2004.0)

    assert len(events) == 1
    assert events[0].types == ["no_gloves"]
