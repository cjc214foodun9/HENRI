"""Import-purity contract for the production ARC runner (Tranche 1, approved 2026-08-21).

Guards the causal module graph of `production_arc_run.py`:

1. The FALSIFIED O-VSA ingress tokenizer must NOT appear as a top-level import.
   (O-VSA_INGRESS_FALSIFIED sealed 65aef1b; top-level symbol had no live consumer.)
2. The gated local `DynamicActionSpaceTransducer` import (action-fiber block,
   default-OFF `HENRI_ARC_ACTION_FIBER`) MUST remain available to its consumer.
3. Protected live components MUST remain imported:
   - HenriSwarmOrchestrator (production_arc_run.py:555 `orch = HenriSwarmOrchestrator(...)`)
   - SagnacMCTSPlanner (gated instantiation + `compute_rt_information_gain` under
     HENRI_ARC_TARGET_GROUNDING / HENRI_ARC_RT_MCTS)
   - AdaptiveViscoelasticThermostat (module-level in henri_api_bridge.py + gated runner path)
"""
import re
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[2] / "production_arc_run.py"

TOP_LEVEL_O_VSA = re.compile(r"^from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer", re.MULTILINE)
GATED_O_VSA = re.compile(r"^\s+from o_vsa_ingress_tokenizer import DynamicActionSpaceTransducer", re.MULTILINE)

PROTECTED = {
    "HenriSwarmOrchestrator": re.compile(r"^from darwinian_phase_swarm import HenriSwarmOrchestrator", re.MULTILINE),
    "SagnacMCTSPlanner": re.compile(r"^from sagnac_mcts_planner import SagnacMCTSPlanner", re.MULTILINE),
    "AdaptiveViscoelasticThermostat": re.compile(
        r"^from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat", re.MULTILINE
    ),
}


def _runner_source() -> str:
    assert RUNNER.is_file(), f"runner not found: {RUNNER}"
    return RUNNER.read_text(encoding="utf-8")


def test_runner_exists():
    assert RUNNER.is_file()


def test_no_top_level_o_vsa_import():
    assert not TOP_LEVEL_O_VSA.search(_runner_source())


def test_gated_local_o_vsa_import_remains():
    assert GATED_O_VSA.search(_runner_source())


def test_protected_imports_remain():
    src = _runner_source()
    for name, pattern in PROTECTED.items():
        assert pattern.search(src), f"protected import missing: {name}"
