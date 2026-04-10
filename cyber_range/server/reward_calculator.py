"""Multi-objective reward function for SOC analyst actions.

Reward design philosophy (learned from winner Kube SRE Gym):
- Episode needs HIGH VARIANCE for GRPO to compute meaningful advantages
- Successful episodes should score +3 to +8+, failed ones -2.0
- Repeat commands must be penalized — LLMs will spam observe_network otherwise
- Phase-order bonus teaches the correct SOC workflow
- Efficiency bonus rewards fast resolution
"""

from __future__ import annotations

try:
    from ..models import ActionResult
except ImportError:
    from cyber_range.models import ActionResult


# ─────────────────────────────────────────────────────────────
# Reward constants (tuned for GRPO signal variance)
# ─────────────────────────────────────────────────────────────

# Per-step action rewards
THREAT_NEUTRALIZED_BASE   =  10.0   # × threat_severity_multiplier
FP_DISMISSED_REWARD       =   3.0   # Correctly ignoring a false positive
EXFIL_PREVENTED_BASE      =   5.0   # × MB prevented
INTEL_GATHERED_BASE       =   2.0   # × intel_gathered
HEALTH_DELTA_BASE         =   2.0   # × health improvement

# Penalties
HEALTHY_HOST_ISOLATED     =  -8.0   # Business disruption
REAL_THREAT_IGNORED       = -15.0   # Catastrophic miss
RESOURCE_COST_BASE        =  -0.5   # × resource_cost
SERVICES_DISRUPTED        = -20.0   # Network outage
FAILED_ACTION_PENALTY     =  -0.5   # Non-observe failed action

# Episode-level bonuses (credit-assignment shaping)
ATTACK_CHAIN_RESOLVED     =  25.0   # Full kill-chain contained
TIMEOUT_PENALTY           =  -2.0   # Failed episode (applied after max_steps with 0 threats neutralized)
EFFICIENCY_BONUS_MAX      =   5.0   # Bonus for resolving in < 50% of max_steps
EFFICIENCY_BONUS_FLOOR    =   1.0   # Minimum bonus on any successful completion

# GRPO training-critical rewards
REPEAT_COMMAND_PENALTY    =  -0.15  # Per repeated action type in an episode (fights reward hacking)
PHASE_ORDER_BONUS         =   0.20  # For correct triage → investigate → contain workflow order
PHASE_ORDER_PENALTY       =  -0.30  # For skipping phases (jumping to block/isolate without investigating)


class RewardCalculator:
    """
    Computes step-level and episode-level rewards from ActionResults.

    Key design decisions:
    1. Repeat penalty teaches exploration, prevents observe_network spam
    2. Phase-order bonus teaches the SOC workflow (not just outcomes)
    3. High-variance episode rewards enable GRPO to compute clean advantages
    """

    # SOC workflow phase order
    _PHASE_ORDER = ["observe_network", "investigate_alert", "run_forensics",
                    "block_ip", "isolate_host", "dismiss_alert"]
    _CONTAINMENT_ACTIONS = {"block_ip", "isolate_host", "restore_backup"}
    _INVESTIGATION_ACTIONS = {"investigate_alert", "run_forensics", "deploy_honeypot"}

    def __init__(self) -> None:
        self._cumulative_reward: float = 0.0
        self._action_history: list[str] = []        # For repeat penalty
        self._phase_history: list[str] = []         # For phase-order bonus
        self._investigated_before_contain: bool = False

    def reset(self) -> None:
        """Reset all tracking for a new episode."""
        self._cumulative_reward = 0.0
        self._action_history = []
        self._phase_history = []
        self._investigated_before_contain = False

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
        action_type = result.action_type

        # ── REPEAT COMMAND PENALTY (GRPO-critical) ──────────────────────
        # Penalize re-using the same action type in the same episode.
        # Exemptions: investigate_alert and run_forensics (legitimately repeated).
        if (action_type in self._action_history
                and action_type not in ("investigate_alert", "run_forensics", "dismiss_alert")):
            repeat_count = self._action_history.count(action_type)
            reward += REPEAT_COMMAND_PENALTY * min(repeat_count, 3)  # Cap at 3x penalty

        self._action_history.append(action_type)

        # ── PHASE-ORDER BONUS ──────────────────────────────────────────
        # Reward agents that investigate before containing.
        # Penalize those that jump straight to block/isolate.
        if action_type in self._CONTAINMENT_ACTIONS:
            if self._investigated_before_contain:
                reward += PHASE_ORDER_BONUS
                self._phase_history.append("contain")
            else:
                reward += PHASE_ORDER_PENALTY   # Blocked without investigating
        elif action_type in self._INVESTIGATION_ACTIONS:
            self._investigated_before_contain = True
            self._phase_history.append("investigate")

        # ── POSITIVE REWARDS ───────────────────────────────────────────

        if result.threat_neutralized:
            reward += THREAT_NEUTRALIZED_BASE * result.threat_severity_multiplier

        if result.false_positive_correctly_dismissed:
            reward += FP_DISMISSED_REWARD

        if result.exfiltration_prevented_mb > 0:
            reward += EXFIL_PREVENTED_BASE * result.exfiltration_prevented_mb

        if result.intel_gathered > 0:
            reward += INTEL_GATHERED_BASE * result.intel_gathered

        if result.health_delta != 0:
            reward += HEALTH_DELTA_BASE * result.health_delta

        if result.attack_chain_resolved:
            reward += ATTACK_CHAIN_RESOLVED

        # ── NEGATIVE REWARDS ───────────────────────────────────────────

        if result.healthy_host_isolated:
            reward += HEALTHY_HOST_ISOLATED

        if result.real_threat_ignored:
            reward += REAL_THREAT_IGNORED

        if result.resource_cost > 0:
            reward += RESOURCE_COST_BASE * result.resource_cost

        if result.critical_services_disrupted:
            reward += SERVICES_DISRUPTED

        if not result.success and action_type != "observe_network":
            reward += FAILED_ACTION_PENALTY

        self._cumulative_reward += reward
        return round(reward, 2)

    def apply_episode_end_bonuses(
        self,
        threats_neutralized: int,
        total_threats: int,
        steps_used: int,
        max_steps: int,
    ) -> float:
        """
        Apply efficiency bonus or timeout penalty at episode end.

        Call this once when the episode terminates to add the final
        GRPO signal that separates successful from failed episodes.

        Args:
            threats_neutralized: Number of threats fully contained
            total_threats:       Total threats in the scenario
            steps_used:          Steps consumed
            max_steps:           Maximum steps allowed

        Returns:
            Episode-end bonus/penalty (also added to cumulative_reward)
        """
        bonus = 0.0

        if total_threats > 0 and threats_neutralized == 0:
            # Complete failure — apply hard timeout penalty
            bonus += TIMEOUT_PENALTY
        elif threats_neutralized > 0:
            # Efficiency bonus — scales with how fast the resolution was
            step_fraction = steps_used / max(max_steps, 1)
            if step_fraction < 0.5:
                # Fast resolution: full efficiency bonus, scaled by fraction neutralized
                effectiveness = threats_neutralized / total_threats
                bonus += EFFICIENCY_BONUS_MAX * effectiveness
            else:
                # Slow but successful — minimum completion bonus
                bonus += EFFICIENCY_BONUS_FLOOR

        self._cumulative_reward += bonus
        return round(bonus, 2)
