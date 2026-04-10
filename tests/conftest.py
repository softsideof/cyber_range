# Copyright (c) 2026. CyberRange OpenEnv Environment.
# Licensed under the BSD-3-Clause License.

"""
Shared test fixtures for CyberRange test suite.
"""

import os
import sys

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cyber_range.server.attack_engine import AttackEngine
from cyber_range.server.cyber_environment import CyberRangeEnvironment
from cyber_range.server.network_simulator import NetworkSimulator
from cyber_range.server.reward_calculator import RewardCalculator

SEED = 42


@pytest.fixture
def network():
    """Initialized NetworkSimulator with seed=42."""
    net = NetworkSimulator(seed=SEED)
    net.initialize(seed=SEED)
    return net


@pytest.fixture
def attack_engine():
    """Fresh AttackEngine with seed=42."""
    return AttackEngine(seed=SEED)


@pytest.fixture
def reward_calc():
    """Fresh RewardCalculator."""
    return RewardCalculator()


@pytest.fixture
def env():
    """Fresh CyberRangeEnvironment (in-process, no server)."""
    return CyberRangeEnvironment()


@pytest.fixture
def env_easy(env):
    """Environment reset to the easy (script_kiddie) scenario."""
    env.reset(task_id="script_kiddie", seed=SEED)
    return env


@pytest.fixture
def env_medium(env):
    """Environment reset to the medium (phishing_campaign) scenario."""
    env.reset(task_id="phishing_campaign", seed=SEED)
    return env


@pytest.fixture
def env_hard(env):
    """Environment reset to the hard (apt_lateral_movement) scenario."""
    env.reset(task_id="apt_lateral_movement", seed=SEED)
    return env
