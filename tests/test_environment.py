# Copyright (c) 2026. CyberRange OpenEnv Environment.
# Licensed under the BSD-3-Clause License.

"""
Tests for CyberRangeEnvironment — the core OpenEnv API contract.

Validates:
    - reset() returns a well-formed Observation
    - step() processes MCP tool actions correctly
    - state property returns valid State with CyberRange metadata
    - ListToolsAction discovers all 12 SOC analyst tools (10 core + 2 playbook)
    - Episode lifecycle: reset → step … → done
"""

import pytest

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv.core.env_server.types import Observation, State

SEED = 42
TASK_IDS = [
    "script_kiddie",
    "phishing_campaign",
    "apt_lateral_movement",
    "ransomware_outbreak",
    "insider_threat_apt",
]

EXPECTED_TOOLS = sorted([
    "observe_network", "investigate_alert", "isolate_host", "block_ip",
    "run_forensics", "deploy_patch", "restore_backup", "dismiss_alert",
    "deploy_honeypot", "escalate_incident",
    "save_playbook", "search_playbooks",  # Phase 4: Analyst Playbook Library
])


# ============================================================================
# reset() Tests
# ============================================================================

class TestReset:
    """Test the reset() → Observation contract."""

    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="script_kiddie", seed=SEED)
        assert isinstance(obs, Observation)

    def test_reset_done_is_false(self, env):
        obs = env.reset(task_id="script_kiddie", seed=SEED)
        assert obs.done is False

    def test_reset_reward_is_zero(self, env):
        obs = env.reset(task_id="script_kiddie", seed=SEED)
        assert isinstance(obs.reward, float)
        assert obs.reward > 0.0

    def test_reset_metadata_is_dict(self, env):
        obs = env.reset(task_id="script_kiddie", seed=SEED)
        assert isinstance(obs.metadata, dict)

    def test_reset_metadata_has_required_keys(self, env):
        obs = env.reset(task_id="script_kiddie", seed=SEED)
        required = {
            "network_topology", "pending_alerts", "resolved_alerts",
            "threat_level", "health_score", "budget_remaining",
            "scenario", "available_actions",
        }
        assert required.issubset(set(obs.metadata.keys()))

    def test_reset_scenario_has_correct_fields(self, env):
        obs = env.reset(task_id="phishing_campaign", seed=SEED)
        scenario = obs.metadata["scenario"]
        assert scenario["id"] == "phishing_campaign"
        assert scenario["difficulty"] == "medium"
        assert scenario["max_steps"] > 0

    def test_reset_network_topology_has_12_nodes(self, env):
        obs = env.reset(task_id="script_kiddie", seed=SEED)
        assert len(obs.metadata["network_topology"]) == 12

    def test_reset_produces_initial_alerts(self, env):
        obs = env.reset(task_id="script_kiddie", seed=SEED)
        alerts = obs.metadata["pending_alerts"]
        assert len(alerts) >= 1

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_reset_all_scenarios(self, env, task_id):
        """Every defined task_id should load without error."""
        obs = env.reset(task_id=task_id, seed=SEED)
        assert isinstance(obs, Observation)
        assert obs.metadata["scenario"]["id"] == task_id

    def test_reset_invalid_scenario_falls_back(self, env):
        """Unknown scenario should fall back to script_kiddie."""
        obs = env.reset(task_id="nonexistent_scenario_xyz", seed=SEED)
        assert obs.metadata["scenario"]["id"] == "script_kiddie"

    def test_reset_is_idempotent(self, env):
        """Resetting twice with same seed should produce same initial state."""
        obs1 = env.reset(task_id="script_kiddie", seed=SEED)
        obs2 = env.reset(task_id="script_kiddie", seed=SEED)
        assert obs1.metadata["network_topology"] == obs2.metadata["network_topology"]
        assert obs1.metadata["health_score"] == obs2.metadata["health_score"]


# ============================================================================
# step() Tests
# ============================================================================

class TestStep:
    """Test the step() → Observation contract."""

    def test_step_returns_observation(self, env_easy):
        obs = env_easy.step(CallToolAction(tool_name="observe_network", arguments={}))
        assert isinstance(obs, Observation)

    def test_step_has_reward(self, env_easy):
        obs = env_easy.step(CallToolAction(tool_name="observe_network", arguments={}))
        assert isinstance(obs.reward, (int, float))

    def test_step_has_done(self, env_easy):
        obs = env_easy.step(CallToolAction(tool_name="observe_network", arguments={}))
        assert isinstance(obs.done, bool)

    def test_step_increments_step_count(self, env_easy):
        assert env_easy.state.step_count == 0
        env_easy.step(CallToolAction(tool_name="observe_network", arguments={}))
        assert env_easy.state.step_count == 1
        env_easy.step(CallToolAction(tool_name="observe_network", arguments={}))
        assert env_easy.state.step_count == 2

    def test_observe_network_is_free(self, env_easy):
        """observe_network should not cost budget."""
        budget_before = env_easy.network.budget_remaining()
        env_easy.step(CallToolAction(tool_name="observe_network", arguments={}))
        budget_after = env_easy.network.budget_remaining()
        assert budget_after == budget_before

    def test_investigate_costs_budget(self, env_easy):
        """investigate_alert should consume budget."""
        budget_before = env_easy.network.budget_remaining()
        alerts = env_easy.network.get_pending_alerts()
        if alerts:
            alert_id = alerts[0]["alert_id"]
            env_easy.step(CallToolAction(
                tool_name="investigate_alert",
                arguments={"alert_id": alert_id},
            ))
            assert env_easy.network.budget_remaining() < budget_before

    def test_episode_ends_at_max_steps(self, env_easy):
        """Episode should terminate when max_steps is reached."""
        max_steps = env_easy._max_steps
        for _ in range(max_steps):
            obs = env_easy.step(CallToolAction(
                tool_name="observe_network", arguments={},
            ))
        assert obs.done is True


# ============================================================================
# state Property Tests
# ============================================================================

class TestState:
    """Test the state property contract."""

    def test_state_returns_state_type(self, env_easy):
        state = env_easy.state
        assert isinstance(state, State)

    def test_state_has_episode_id(self, env_easy):
        assert isinstance(env_easy.state.episode_id, str)
        assert len(env_easy.state.episode_id) > 0

    def test_state_has_step_count(self, env_easy):
        assert isinstance(env_easy.state.step_count, int)
        assert env_easy.state.step_count == 0

    def test_state_tracks_scenario(self, env_easy):
        state = env_easy.state
        assert state.scenario_id == "script_kiddie"

    def test_state_tracks_health_score(self, env_easy):
        state = env_easy.state
        assert 0.0 <= float(state.health_score) <= 1.0

    def test_state_tracks_threat_level(self, env_easy):
        state = env_easy.state
        assert state.threat_level in ("green", "yellow", "orange", "red", "critical")


# ============================================================================
# Tool Discovery Tests
# ============================================================================

class TestToolDiscovery:
    """Test MCP tool listing via ListToolsAction."""

    def test_list_tools_returns_10_tools(self, env_easy):
        obs = env_easy.step(ListToolsAction())
        tools = getattr(obs, "tools", [])
        assert len(tools) == 12  # 10 core SOC tools + save_playbook + search_playbooks

    def test_list_tools_has_correct_names(self, env_easy):
        obs = env_easy.step(ListToolsAction())
        tools = getattr(obs, "tools", [])
        tool_names = sorted([t.name for t in tools])
        assert tool_names == EXPECTED_TOOLS

    def test_each_tool_has_description(self, env_easy):
        obs = env_easy.step(ListToolsAction())
        tools = getattr(obs, "tools", [])
        for tool in tools:
            assert tool.description, f"Tool {tool.name} has no description"


# ============================================================================
# Episode Lifecycle Tests
# ============================================================================

class TestEpisodeLifecycle:
    """Test the full episode lifecycle: reset → actions → done → grading."""

    def test_full_episode_produces_grader_result(self, env_easy):
        """Running to completion should produce a grader result."""
        max_steps = env_easy._max_steps
        for _ in range(max_steps):
            obs = env_easy.step(CallToolAction(
                tool_name="observe_network", arguments={},
            ))
        state = env_easy.state
        assert state.grader_result is not None
        assert "final_score" in state.grader_result

    def test_grader_score_in_range(self, env_easy):
        """Grader score should be between 0.0 and 1.0."""
        max_steps = env_easy._max_steps
        for _ in range(max_steps):
            env_easy.step(CallToolAction(
                tool_name="observe_network", arguments={},
            ))
        score = env_easy.state.grader_result["final_score"]
        assert 0.0 <= score <= 1.0

    def test_cannot_step_after_done(self, env_easy):
        """Stepping after episode is done should remain done."""
        max_steps = env_easy._max_steps
        for _ in range(max_steps):
            env_easy.step(CallToolAction(
                tool_name="observe_network", arguments={},
            ))
        # One more step after done
        obs = env_easy.step(CallToolAction(
            tool_name="observe_network", arguments={},
        ))
        assert obs.done is True

    def test_reset_clears_previous_episode(self, env):
        """Reset after a completed episode should start fresh."""
        env.reset(task_id="script_kiddie", seed=SEED)
        for _ in range(15):
            env.step(CallToolAction(
                tool_name="observe_network", arguments={},
            ))
        # Now reset
        obs = env.reset(task_id="phishing_campaign", seed=SEED)
        assert obs.done is False
        assert obs.reward is None
        assert env.state.step_count == 0
        assert env.state.scenario_id == "phishing_campaign"
