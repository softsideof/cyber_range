# Copyright (c) 2026. CyberRange OpenEnv Environment.
# Licensed under the BSD-3-Clause License.

"""
Tests for RewardCalculator — the multi-objective reward function.

Validates:
    - Positive rewards: threat neutralization, FP dismissal, intel, exfiltration prevention
    - Negative rewards: false negatives, business disruption, resource waste
    - Shaped rewards: health delta, attack chain resolution
    - Cumulative tracking
"""

import pytest

from cyber_range.models import ActionResult
from cyber_range.server.reward_calculator import RewardCalculator


@pytest.fixture
def calc():
    return RewardCalculator()


# ============================================================================
# Positive Reward Tests
# ============================================================================

class TestPositiveRewards:
    """Test that good SOC decisions are rewarded."""

    def test_threat_neutralized_positive(self, calc):
        result = ActionResult(
            action_type="isolate_host", success=True,
            description="Host isolated",
            threat_neutralized=True,
            threat_severity_multiplier=1.0,
        )
        reward = calc.calculate(result)
        assert reward > 0

    def test_high_severity_threat_bigger_reward(self, calc):
        low = ActionResult(
            action_type="block_ip", success=True,
            description="Blocked",
            threat_neutralized=True,
            threat_severity_multiplier=1.0,
        )
        high = ActionResult(
            action_type="isolate_host", success=True,
            description="Isolated",
            threat_neutralized=True,
            threat_severity_multiplier=2.0,
        )
        # Reset calc for each
        calc1 = RewardCalculator()
        calc2 = RewardCalculator()
        r_low = calc1.calculate(low)
        r_high = calc2.calculate(high)
        assert r_high > r_low

    def test_fp_correctly_dismissed_positive(self, calc):
        result = ActionResult(
            action_type="dismiss_alert", success=True,
            description="Dismissed FP",
            false_positive_correctly_dismissed=True,
        )
        reward = calc.calculate(result)
        assert reward > 0

    def test_intel_gathered_positive(self, calc):
        result = ActionResult(
            action_type="deploy_honeypot", success=True,
            description="Honeypot",
            intel_gathered=1.0,
            resource_cost=4.0,
        )
        reward = calc.calculate(result)
        # Intel reward (+2.0) minus resource cost (-2.0) — net should be zero or positive
        # +2.0 * 1.0 - 0.5 * 4.0 = 2.0 - 2.0 = 0.0
        assert reward >= 0.0

    def test_exfiltration_prevented_positive(self, calc):
        result = ActionResult(
            action_type="isolate_host", success=True,
            description="Prevented exfil",
            exfiltration_prevented_mb=10.0,
        )
        reward = calc.calculate(result)
        assert reward > 0


# ============================================================================
# Negative Reward Tests
# ============================================================================

class TestNegativeRewards:
    """Test that bad SOC decisions are penalized."""

    def test_healthy_host_isolated_negative(self, calc):
        result = ActionResult(
            action_type="isolate_host", success=True,
            description="Isolated healthy host",
            healthy_host_isolated=True,
            resource_cost=3.0,
        )
        reward = calc.calculate(result)
        assert reward < 0

    def test_real_threat_ignored_severe_penalty(self, calc):
        result = ActionResult(
            action_type="dismiss_alert", success=True,
            description="Dismissed real threat",
            real_threat_ignored=True,
        )
        reward = calc.calculate(result)
        assert reward < -10  # Should be heavily penalized

    def test_critical_services_disrupted_severe_penalty(self, calc):
        result = ActionResult(
            action_type="isolate_host", success=True,
            description="Disrupted critical",
            critical_services_disrupted=True,
            healthy_host_isolated=True,
            resource_cost=3.0,
        )
        reward = calc.calculate(result)
        assert reward < -20  # -20 for critical + -8 for healthy + cost

    def test_resource_cost_penalized(self, calc):
        expensive = ActionResult(
            action_type="run_forensics", success=True,
            description="Expensive scan",
            resource_cost=5.0,
        )
        cheap = ActionResult(
            action_type="investigate_alert", success=True,
            description="Cheap scan",
            resource_cost=1.0,
        )
        calc1 = RewardCalculator()
        calc2 = RewardCalculator()
        r_expensive = calc1.calculate(expensive)
        r_cheap = calc2.calculate(cheap)
        assert r_expensive < r_cheap

    def test_failed_action_small_penalty(self, calc):
        result = ActionResult(
            action_type="isolate_host", success=False,
            description="Node not found",
        )
        reward = calc.calculate(result)
        assert reward < 0


# ============================================================================
# Shaped Reward Tests
# ============================================================================

class TestShapedRewards:
    """Test shaped rewards for learning guidance."""

    def test_health_improvement_rewarded(self, calc):
        result = ActionResult(
            action_type="restore_backup", success=True,
            description="Restored",
            threat_neutralized=True,
            health_delta=0.08,
            resource_cost=8.0,
        )
        # +10 (threat) + 2*0.08 (health) - 0.5*8 (cost) = 10 + 0.16 - 4 = 6.16
        reward = calc.calculate(result)
        assert reward > 0

    def test_attack_chain_resolved_bonus(self, calc):
        result = ActionResult(
            action_type="isolate_host", success=True,
            description="Chain resolved",
            attack_chain_resolved=True,
        )
        reward = calc.calculate(result)
        assert reward >= 25.0  # Big bonus for full chain resolution


# ============================================================================
# Cumulative Tracking Tests
# ============================================================================

class TestCumulativeReward:
    """Test cumulative reward tracking."""

    def test_starts_at_zero(self, calc):
        assert calc.cumulative_reward == 0.0

    def test_accumulates_across_actions(self, calc):
        r1 = ActionResult(
            action_type="dismiss_alert", success=True,
            description="", false_positive_correctly_dismissed=True,
        )
        r2 = ActionResult(
            action_type="dismiss_alert", success=True,
            description="", false_positive_correctly_dismissed=True,
        )
        calc.calculate(r1)
        calc.calculate(r2)
        assert calc.cumulative_reward == 6.0  # 3.0 + 3.0

    def test_reset_clears_cumulative(self, calc):
        r = ActionResult(
            action_type="dismiss_alert", success=True,
            description="", false_positive_correctly_dismissed=True,
        )
        calc.calculate(r)
        assert calc.cumulative_reward != 0
        calc.reset()
        assert calc.cumulative_reward == 0.0
