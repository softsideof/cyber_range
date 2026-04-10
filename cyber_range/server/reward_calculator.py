"""Multi-objective reward function for SOC analyst actions."""

try:
    from ..models import ActionResult
except ImportError:
    from cyber_range.models import ActionResult


class RewardCalculator:
    """Computes step-level reward from an ActionResult."""


    def __init__(self) -> None:
        self._cumulative_reward: float = 0.0

    def reset(self) -> None:
        """Reset cumulative tracking."""
        self._cumulative_reward = 0.0

    @property
    def cumulative_reward(self) -> float:
        """Total reward accumulated this episode."""
        return round(self._cumulative_reward, 2)

    def calculate(self, result: ActionResult) -> float:
        """
        Calculate reward for a single action.

        Args:
            result: The ActionResult from the agent's defensive action.

        Returns:
            Float reward value (positive = good, negative = bad).
        """
        reward = 0.0

        # === POSITIVE REWARDS ===

        # Correctly identifying and neutralizing a real threat
        if result.threat_neutralized:
            reward += 10.0 * result.threat_severity_multiplier

        # Correctly dismissing a false positive (shows intelligence)
        if result.false_positive_correctly_dismissed:
            reward += 3.0

        # Preventing data exfiltration
        if result.exfiltration_prevented_mb > 0:
            reward += 5.0 * result.exfiltration_prevented_mb

        # Gathering intelligence (honeypot, forensics)
        if result.intel_gathered > 0:
            reward += 2.0 * result.intel_gathered

        # === NEGATIVE REWARDS (PENALTIES) ===

        # Isolating a healthy host (business disruption!)
        if result.healthy_host_isolated:
            reward -= 8.0

        # Missing a real threat (false negative is catastrophic)
        if result.real_threat_ignored:
            reward -= 15.0

        # Resource waste (budget is finite, like real SOCs)
        if result.resource_cost > 0:
            reward -= 0.5 * result.resource_cost

        # Network went down due to over-aggressive response
        if result.critical_services_disrupted:
            reward -= 20.0

        # === SHAPED REWARDS (guide learning) ===

        # Reward for improving network health score
        if result.health_delta != 0:
            reward += 2.0 * result.health_delta

        # Bonus for resolving entire attack chains (not just symptoms)
        if result.attack_chain_resolved:
            reward += 25.0

        # Small penalty for no-op or failed actions (encourage decisiveness)
        if not result.success and result.action_type != "observe_network":
            reward -= 0.5

        self._cumulative_reward += reward
        return round(reward, 2)
