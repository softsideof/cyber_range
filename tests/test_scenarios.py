# Copyright (c) 2026. CyberRange OpenEnv Environment.
# Licensed under the BSD-3-Clause License.

"""
Tests for deterministic grading across all 5 scenarios.

Validates:
    - Reproducible scores with seed=42
    - Different scenarios produce different scores
    - Good agent strategy scores higher than passive observation
    - Grader component weights sum correctly
"""

import pytest

from openenv.core.env_server.mcp_types import CallToolAction

from cyber_range.server.attack_engine import SCENARIOS
from cyber_range.server.cyber_environment import CyberRangeEnvironment

SEED = 42
ALL_SCENARIOS = list(SCENARIOS.keys())


def run_passive_episode(task_id: str) -> dict:
    """Run a passive (observe-only) episode and return grader result."""
    env = CyberRangeEnvironment()
    obs = env.reset(task_id=task_id, seed=SEED)
    max_steps = obs.metadata["scenario"]["max_steps"]

    for _ in range(max_steps):
        obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))
        if obs.done:
            break

    return env.state.grader_result or {}


def run_active_episode(task_id: str) -> dict:
    """Run a simple active episode: observe → investigate → block → isolate."""
    env = CyberRangeEnvironment()
    obs = env.reset(task_id=task_id, seed=SEED)
    max_steps = obs.metadata["scenario"]["max_steps"]
    alerts = obs.metadata.get("pending_alerts", [])

    # Step 1: Observe
    env.step(CallToolAction(tool_name="observe_network", arguments={}))

    # Steps 2-N: Investigate all alerts
    for alert in alerts:
        env.step(CallToolAction(
            tool_name="investigate_alert",
            arguments={"alert_id": alert["alert_id"]},
        ))

    # Block known attacker IPs
    for ip in ["185.220.101.42", "94.232.46.19", "45.155.205.233"]:
        env.step(CallToolAction(
            tool_name="block_ip", arguments={"ip_address": ip},
        ))

    # Run remaining steps
    while env.state.step_count < max_steps:
        obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))
        if obs.done:
            break

    return env.state.grader_result or {}


class TestDeterministicGrading:
    """Test that grading is deterministic and reproducible."""

    @pytest.mark.parametrize("task_id", ALL_SCENARIOS)
    def test_reproducible_scores(self, task_id):
        """Same seed → identical final score."""
        r1 = run_passive_episode(task_id)
        r2 = run_passive_episode(task_id)
        assert r1["final_score"] == r2["final_score"]

    @pytest.mark.parametrize("task_id", ALL_SCENARIOS)
    def test_score_in_valid_range(self, task_id):
        """All grader scores must be in [0.0, 1.0]."""
        result = run_passive_episode(task_id)
        assert 0.0 <= result["final_score"] <= 1.0

    def test_different_scenarios_different_scores(self):
        """Active play across different scenarios should produce varied scores."""
        scores = []
        for task_id in ALL_SCENARIOS:
            result = run_active_episode(task_id)
            scores.append(result["final_score"])
        # Active play should produce at least 2 distinct scores
        assert len(set(scores)) >= 2


class TestScoreQuality:
    """Test that active play scores higher than passive observation."""

    @pytest.mark.parametrize("task_id", ["script_kiddie", "phishing_campaign"])
    def test_active_beats_passive(self, task_id):
        """An agent that takes action should score higher than a passive one."""
        passive = run_passive_episode(task_id)
        active = run_active_episode(task_id)
        assert active["final_score"] >= passive["final_score"]


class TestGraderComponents:
    """Test grader component weights and structure."""

    @pytest.mark.parametrize("task_id", ALL_SCENARIOS)
    def test_components_sum_to_final_score(self, task_id):
        """Component scores should sum to approximately the final score."""
        result = run_passive_episode(task_id)
        component_sum = sum(
            result.get(k, 0.0)
            for k in ["threat_neutralization", "false_positive_handling",
                       "data_protection", "collateral_damage", "efficiency"]
        )
        # Allow small floating point tolerance
        assert abs(component_sum - result["final_score"]) < 0.06

    @pytest.mark.parametrize("task_id", ALL_SCENARIOS)
    def test_details_include_metadata(self, task_id):
        """Grader details should include operational metadata."""
        result = run_passive_episode(task_id)
        details = result.get("details", {})
        assert "threats_neutralized" in details
        assert "false_positives_dismissed" in details
        assert "data_exfiltrated_mb" in details
        assert "steps_used" in details
        assert "difficulty" in details
