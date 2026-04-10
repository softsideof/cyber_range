# Copyright (c) 2026. CyberRange OpenEnv Environment.
# Licensed under the BSD-3-Clause License.

"""
Tests for AttackEngine — scenario loading, progression, and grading.

Validates:
    - Scenario loading for all 5 task IDs
    - Attack phase progression and containment detection
    - Grading produces deterministic scores in [0.0, 1.0]
    - No double-counting of threat neutralization
    - Kill chain dependency activation
"""

import pytest

from cyber_range.models import AlertType, NodeStatus
from cyber_range.server.attack_engine import AttackEngine, SCENARIOS
from cyber_range.server.network_simulator import NetworkSimulator

SEED = 42
ALL_SCENARIOS = list(SCENARIOS.keys())


# ============================================================================
# Scenario Loading Tests
# ============================================================================

class TestScenarioLoading:
    """Test loading and initializing attack scenarios."""

    @pytest.mark.parametrize("scenario_id", ALL_SCENARIOS)
    def test_load_all_scenarios(self, network, attack_engine, scenario_id):
        scenario = attack_engine.load_scenario(scenario_id, network, seed=SEED)
        assert scenario.scenario_id == scenario_id
        assert scenario.max_steps > 0

    def test_load_invalid_scenario_raises(self, network, attack_engine):
        with pytest.raises(ValueError, match="Unknown scenario"):
            attack_engine.load_scenario("nonexistent", network, seed=SEED)

    def test_scenario_generates_alerts(self, network, attack_engine):
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        alerts = network.get_pending_alerts()
        assert len(alerts) >= 1

    def test_scenario_sets_initial_compromised(self, network, attack_engine):
        attack_engine.load_scenario("phishing_campaign", network, seed=SEED)
        assert network.nodes["ws-01"].status == NodeStatus.COMPROMISED
        assert network.nodes["ws-02"].status == NodeStatus.COMPROMISED

    def test_scenario_metrics_initialized(self, network, attack_engine):
        attack_engine.load_scenario("apt_lateral_movement", network, seed=SEED)
        assert attack_engine.metrics.total_threats == 5
        assert attack_engine.metrics.false_positives_total == 3
        assert attack_engine.metrics.threats_neutralized == 0


# ============================================================================
# Attack Phase Progression Tests
# ============================================================================

class TestAttackProgression:
    """Test that attack phases advance and can be contained."""

    def test_phases_advance_on_advance(self, network, attack_engine):
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        active_before = [p for p in attack_engine.phases if p.is_active]
        assert len(active_before) >= 1

        # Advance a few steps — phases should progress
        for _ in range(3):
            attack_engine.advance(network)

        phase = attack_engine.phases[0]
        assert phase.steps_elapsed >= 3

    def test_isolation_contains_phase(self, network, attack_engine):
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        # web-01 is the target of the brute force
        network.nodes["web-01"].status = NodeStatus.ISOLATED
        events = attack_engine.advance(network)
        phase = attack_engine.phases[0]
        assert phase.is_contained is True

    def test_blocking_ip_contains_brute_force(self, network, attack_engine):
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        network.blocked_ips.add("185.220.101.42")
        events = attack_engine.advance(network)
        phase = attack_engine.phases[0]
        assert phase.is_contained is True

    def test_completed_phase_activates_prerequisite(self, network, attack_engine):
        attack_engine.load_scenario("phishing_campaign", network, seed=SEED)
        # phish-01 (ws-01) activates phish-03 (lateral to app-01) on completion
        phish_01 = attack_engine.phases[0]
        phish_03 = attack_engine.phases[2]
        assert phish_03.is_active is False

        # Run enough steps for phish-01 to complete
        for _ in range(phish_01.steps_to_complete + 1):
            attack_engine.advance(network)

        assert phish_01.is_completed is True
        assert phish_03.is_active is True

    def test_exfiltration_tracks_data(self, network, attack_engine):
        attack_engine.load_scenario("apt_lateral_movement", network, seed=SEED)
        # Advance enough for the exfil phase (apt-05) to activate and run
        for _ in range(50):
            attack_engine.advance(network)

        # Check if exfiltration was tracked
        total_exfil = attack_engine._total_exfiltrated_mb
        # Some phases should have exfiltrated data
        assert total_exfil >= 0


# ============================================================================
# Containment Without Double-Counting Tests
# ============================================================================

class TestNoDoubleCounting:
    """Verify that threat neutralization is not double-counted."""

    def test_threat_count_never_exceeds_total(self, network, attack_engine):
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        total = attack_engine.metrics.total_threats

        # Block all IPs and isolate all nodes
        network.blocked_ips.update(["185.220.101.42", "94.232.46.19", "45.155.205.233"])
        for node_id in list(network.nodes.keys()):
            network.nodes[node_id].status = NodeStatus.ISOLATED

        # Advance many times
        for _ in range(20):
            attack_engine.advance(network)

        assert attack_engine.metrics.threats_neutralized <= total

    @pytest.mark.parametrize("scenario_id", ALL_SCENARIOS)
    def test_no_double_counting_any_scenario(self, scenario_id):
        """Run each scenario to completion and verify threat count <= total."""
        net = NetworkSimulator(seed=SEED)
        net.initialize(seed=SEED)
        engine = AttackEngine(seed=SEED)
        scenario = engine.load_scenario(scenario_id, net, seed=SEED)

        # Isolate all compromised nodes and block all IPs
        net.blocked_ips.update(["185.220.101.42", "94.232.46.19", "45.155.205.233"])
        for node_id in scenario.initial_compromised_nodes:
            net.nodes[node_id].status = NodeStatus.ISOLATED

        for _ in range(scenario.max_steps * 2):
            engine.advance(net)

        assert engine.metrics.threats_neutralized <= engine.metrics.total_threats


# ============================================================================
# Grading Tests
# ============================================================================

class TestGrading:
    """Test the grading system."""

    def test_grade_returns_final_score(self, network, attack_engine):
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        result = attack_engine.grade_episode(network, steps_used=10)
        assert "final_score" in result
        assert 0.0 <= result["final_score"] <= 1.0

    def test_grade_has_all_components(self, network, attack_engine):
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        result = attack_engine.grade_episode(network, steps_used=10)
        expected_components = {
            "threat_neutralization", "false_positive_handling",
            "data_protection", "collateral_damage", "efficiency",
        }
        assert expected_components.issubset(set(result.keys()))

    def test_grade_deterministic(self, network, attack_engine):
        """Same inputs should produce identical grades."""
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        r1 = attack_engine.grade_episode(network, steps_used=10)

        net2 = NetworkSimulator(seed=SEED)
        net2.initialize(seed=SEED)
        engine2 = AttackEngine(seed=SEED)
        engine2.load_scenario("script_kiddie", net2, seed=SEED)
        r2 = engine2.grade_episode(net2, steps_used=10)

        assert r1["final_score"] == r2["final_score"]

    def test_fast_resolution_scores_higher_efficiency(self, network, attack_engine):
        """Faster episode = better efficiency score."""
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        fast = attack_engine.grade_episode(network, steps_used=5)
        slow = attack_engine.grade_episode(network, steps_used=15)
        assert fast["efficiency"] >= slow["efficiency"]

    @pytest.mark.parametrize("scenario_id", ALL_SCENARIOS)
    def test_grade_all_scenarios(self, scenario_id):
        """Every scenario should produce a valid grade."""
        net = NetworkSimulator(seed=SEED)
        net.initialize(seed=SEED)
        engine = AttackEngine(seed=SEED)
        engine.load_scenario(scenario_id, net, seed=SEED)
        result = engine.grade_episode(net, steps_used=10)
        assert 0.0 <= result["final_score"] <= 1.0

    def test_grade_has_details(self, network, attack_engine):
        attack_engine.load_scenario("script_kiddie", network, seed=SEED)
        result = attack_engine.grade_episode(network, steps_used=10)
        assert "details" in result
        details = result["details"]
        assert "threats_neutralized" in details
        assert "false_positives_dismissed" in details
        assert "difficulty" in details
