from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_record(record: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
    errors = []
    for key in required_fields:
        if key not in record:
            errors.append(f"missing field: {key}")

    if "instruction" in record and isinstance(record["instruction"], str):
        if len(record["instruction"].strip()) < 3:
            errors.append("instruction too short")

    if "action_sequence" in record and not isinstance(record["action_sequence"], list):
        errors.append("action_sequence must be list")

    return (len(errors) == 0, errors)
