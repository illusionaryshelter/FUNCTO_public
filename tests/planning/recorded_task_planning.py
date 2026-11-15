"""Recorded-planning test harness uses the production passive recorder."""

# BEGIN one recorder implementation serves production and recorded checks.
from manipulation.planning_records import (
    MAX_CURVE_METADATA_BYTES, _json_value, _diagnostic_value,
    _write_json_value, _write_diagnostic_json, save_curve, plan_and_record,
)
# END this harness introduces no planning or archive implementation.
