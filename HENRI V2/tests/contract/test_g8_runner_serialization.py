"""Contract: G8 thermo serialization reaches the runner JSONL record.

Guards the G8 diagnostic-hole fix (carrier/g8-thermo-timescale):
  * the runner tele.emit record MUST serialize chosen["thermo_ratios"]
    (None when the planner is default-OFF / ratio path absent) and the
    Gibbs-branch marker thermo_gibbs;
  * both keys live inside the same telemetry record as the existing
    "thermo_shadow" sidecar key;
  * planner-level default-OFF/Gibbs semantics are covered by
    test_g8_wiring.py; this test guards the runner record line only.

Static source-assert pattern (same family as test_arc_thermostat_shadow.py).
"""
import sys
from pathlib import Path

HENRI2 = Path(__file__).resolve().parents[1]  # <wt>/HENRI V2
sys.path.insert(0, str(HENRI2))

SRC = (HENRI2 / "production_arc_run.py").read_text(encoding="utf-8")


def test_runner_serializes_thermo_ratios_key():
    assert '"thermo_ratios": chosen.get("thermo_ratios")' in SRC


def test_runner_serializes_gibbs_marker():
    assert '"thermo_gibbs": bool(chosen.get("thermo_gibbs", False))' in SRC


def test_keys_live_in_same_record_as_shadow():
    a = SRC.index('"thermo_ratios": chosen.get("thermo_ratios")')
    b = SRC.index('"thermo_shadow": thermo_shadow_info')
    assert abs(a - b) < 200, (a, b)


def test_default_off_path_none_safe():
    # chosen.get(...) with the key absent returns None -> JSON null.
    # No indexing, no KeyError on the default-OFF planner record.
    assert 'chosen.get("thermo_ratios")' in SRC
