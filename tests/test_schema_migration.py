"""Tests for schema migration."""
from labsopguard.schema_migration import auto_migrate, check_version_compatibility, CURRENT_VERSIONS


class TestSchemaMigration:
    def test_migrate_events_v3_to_v4(self):
        data = {
            "schema_version": "physical_events.v3",
            "events": [{"event_id": "evt_1", "event_type": "object_move"}],
        }
        result = auto_migrate(data)
        assert result["schema_version"] == "physical_events.v4"
        assert result["events"][0]["related_detection_classes"] == []
        assert result["events"][0]["notes"] == ""

    def test_migrate_preprocessing_v3_to_v4(self):
        data = {
            "schema_version": "preprocessing.v3",
            "event_preprocessing": {"detection_frame_count": 100},
        }
        result = auto_migrate(data)
        assert result["schema_version"] == "preprocessing.v4"
        assert result["event_preprocessing"]["tracked_object_count"] == 0

    def test_no_migration_needed(self):
        data = {"schema_version": "physical_events.v4", "events": []}
        result = auto_migrate(data)
        assert result == data

    def test_unknown_version(self):
        data = {"schema_version": "unknown.v99", "data": []}
        result = auto_migrate(data)
        assert result == data

    def test_empty_data(self):
        assert auto_migrate({}) == {}
        assert auto_migrate(None) is None

    def test_check_version_compatibility(self):
        assert check_version_compatibility(
            {"schema_version": "physical_events.v4"}, "physical_events"
        )
        assert not check_version_compatibility(
            {"schema_version": "physical_events.v3"}, "physical_events"
        )
