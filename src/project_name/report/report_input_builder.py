from __future__ import annotations

from typing import Any, Dict, List


class ReportInputBuilder:
    """Aggregate structured events into report-ready input payload."""

    def build(self, session_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        violations = [e for e in events if bool(e.get("violation_flag", False))]
        det_count = len([e for e in events if not bool(e.get("violation_flag", False))])
        sev_counter: Dict[str, int] = {}
        for v in violations:
            sev = str(v.get("severity_level", "unknown"))
            sev_counter[sev] = sev_counter.get(sev, 0) + 1

        timeline = [
            {
                "timestamp": e.get("timestamp"),
                "event_type": e.get("event_type"),
                "sop_step": e.get("sop_step"),
                "violation_flag": e.get("violation_flag"),
                "severity_level": e.get("severity_level"),
            }
            for e in events
        ]

        compliance_ratio = 1.0
        if events:
            compliance_ratio = max(0.0, 1.0 - len(violations) / len(events))

        return {
            "title": "Lab SOP Compliance Report",
            "session_id": session_id,
            "summary": {
                "total_events": len(events),
                "detection_events": det_count,
                "violation_events": len(violations),
                "severity_distribution": sev_counter,
                "compliance_ratio": compliance_ratio,
            },
            "violations": violations,
            "timeline": timeline,
        }
