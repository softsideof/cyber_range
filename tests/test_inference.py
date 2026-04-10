# Copyright (c) 2026. CyberRange OpenEnv Environment.
# Licensed under the BSD-3-Clause License.

"""
Tests for inference.py utility functions.

Validates:
    - parse_tool_call() correctly extracts tool names and arguments
    - format_observation() produces valid LLM-ready prompts
    - HeuristicAgent decision-making logic
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference import HeuristicAgent, format_observation, parse_tool_call


# ============================================================================
# parse_tool_call Tests
# ============================================================================

class TestParseToolCall:
    """Test LLM response parsing into tool calls."""

    def test_parse_simple_tool(self):
        text = 'TOOL: observe_network\nARGS: {}'
        name, args = parse_tool_call(text)
        assert name == "observe_network"
        assert args == {}

    def test_parse_tool_with_args(self):
        text = 'TOOL: investigate_alert\nARGS: {"alert_id": "ALT-0001"}'
        name, args = parse_tool_call(text)
        assert name == "investigate_alert"
        assert args == {"alert_id": "ALT-0001"}

    def test_parse_tool_with_ip_arg(self):
        text = 'TOOL: block_ip\nARGS: {"ip_address": "185.220.101.42"}'
        name, args = parse_tool_call(text)
        assert name == "block_ip"
        assert args == {"ip_address": "185.220.101.42"}

    def test_parse_empty_defaults_to_observe(self):
        name, args = parse_tool_call("")
        assert name == "observe_network"
        assert args == {}

    def test_parse_none_defaults_to_observe(self):
        name, args = parse_tool_call("")
        assert name == "observe_network"

    def test_parse_malformed_args_defaults_empty(self):
        text = 'TOOL: block_ip\nARGS: {broken json}'
        name, args = parse_tool_call(text)
        assert name == "block_ip"
        assert args == {}

    def test_parse_with_surrounding_text(self):
        text = (
            "I should investigate the first alert.\n"
            "TOOL: investigate_alert\n"
            'ARGS: {"alert_id": "ALT-0002"}\n'
            "This will reveal if it's a real threat."
        )
        name, args = parse_tool_call(text)
        assert name == "investigate_alert"
        assert args["alert_id"] == "ALT-0002"

    def test_parse_case_insensitive_tool(self):
        text = 'tool: BLOCK_IP\nargs: {"ip_address": "1.2.3.4"}'
        name, args = parse_tool_call(text)
        assert name.lower() == "block_ip"


# ============================================================================
# format_observation Tests
# ============================================================================

class TestFormatObservation:
    """Test observation formatting for LLM context."""

    def test_format_dict_observation(self):
        obs = {"threat_level": "yellow", "health_score": 0.9}
        result = format_observation(obs, step=1, max_steps=15)
        assert "Step 1/15" in result
        assert "yellow" in result

    def test_format_truncates_large_topology(self):
        obs = {"network_topology": [{"id": f"node-{i}"} for i in range(12)]}
        result = format_observation(obs, step=1, max_steps=15)
        # Should truncate to 6 + note
        assert "more nodes" in result

    def test_format_string_observation(self):
        result = format_observation("raw text result", step=5, max_steps=20)
        assert "Step 5/20" in result
        assert "raw text" in result

    def test_format_includes_action_prompt(self):
        result = format_observation({}, step=1, max_steps=10)
        assert "next action" in result.lower()


# ============================================================================
# HeuristicAgent Tests
# ============================================================================

class TestHeuristicAgent:
    """Test the rule-based fallback agent."""

    def test_first_action_is_observe(self):
        agent = HeuristicAgent(initial_alerts=[], initial_topology=[])
        tool, args = agent.decide(None, [])
        assert tool == "observe_network"

    def test_investigates_before_acting(self):
        alerts = [
            {"alert_id": "ALT-0001", "confidence": 0.9},
            {"alert_id": "ALT-0002", "confidence": 0.3},
        ]
        agent = HeuristicAgent(initial_alerts=alerts, initial_topology=[])
        # First: observe
        agent.decide(None, alerts)
        # Next: should investigate an alert
        tool, args = agent.decide(None, alerts)
        assert tool == "investigate_alert"

    def test_blocks_discovered_ips(self):
        agent = HeuristicAgent(initial_alerts=[], initial_topology=[])
        agent._step = 10  # Skip early phases
        agent._investigated_alerts = {"ALT-0001"}
        agent._ips_to_block = ["203.0.113.5"]

        tool, args = agent.decide(None, [])
        assert tool == "block_ip"
        assert args["ip_address"] == "203.0.113.5"

    def test_dismisses_fp_candidates(self):
        alerts = [
            {"alert_id": "ALT-FP1", "confidence": 0.3},
        ]
        agent = HeuristicAgent(initial_alerts=alerts, initial_topology=[])
        agent._step = 20  # Past investigation/blocking phases
        agent._investigated_alerts = {"ALT-FP1"}
        agent._ips_to_block = []
        agent._blocked_ips = {"185.220.101.42", "94.232.46.19", "45.155.205.233"}
        agent._confirmed_fps = ["ALT-FP1"]

        tool, args = agent.decide(None, [])
        assert tool == "dismiss_alert"

    def test_processes_investigation_results(self):
        alerts = [
            {"alert_id": "ALT-0001", "confidence": 0.9},
        ]
        agent = HeuristicAgent(initial_alerts=alerts, initial_topology=[])

        # Step 1: observe (returns observe_network)
        agent.decide(None, alerts)

        # Step 2: investigate ALT-0001
        agent.decide(None, alerts)

        # Step 3: Pass investigation result. The agent should process it,
        # discover the attacker IP and compromised node, then immediately
        # try to block the attacker IP.
        investigation_result = {
            "details": {
                "forensic_evidence": "Analysis confirms malicious activity! Unauthorized access.",
                "alert_id": "ALT-0001",
                "source_ip": "203.0.113.99",
                "related_node_id": "ws-01",
            }
        }
        tool, args = agent.decide(investigation_result, alerts)
        # Agent processes the result and immediately returns a block_ip action
        # for the discovered external IP
        assert tool == "block_ip"
        assert args["ip_address"] == "203.0.113.99"
        # The compromised node should be queued for later isolation
        assert "ws-01" in agent._nodes_to_isolate
