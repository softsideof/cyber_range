"""
CyberRange Benchmark Suite

Validates the environment and benchmarks agent performance.

Usage:
    python examples/benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from openenv.core.env_server.mcp_types import CallToolAction
from cyber_range.server.cyber_environment import CyberRangeEnvironment
from cyber_range.server.attack_engine import SCENARIOS


def run_validation():
    """Run all validation checks."""
    results = []

    # Test 1: Environment creation
    try:
        env = CyberRangeEnvironment()
        results.append(("Environment creation", True, "CyberRangeEnvironment instantiated"))
    except Exception as e:
        results.append(("Environment creation", False, str(e)))
        return results

    # Test 2-6: Scenario loading
    for scenario_id in SCENARIOS:
        try:
            obs = env.reset(task_id=scenario_id, seed=42)
            assert obs.metadata is not None
            assert "scenario" in obs.metadata
            results.append((f"Load {scenario_id}", True, f"max_steps={obs.metadata['scenario']['max_steps']}"))
        except Exception as e:
            results.append((f"Load {scenario_id}", False, str(e)))

    # Test 7: Tool execution
    try:
        obs = env.reset(task_id="script_kiddie", seed=42)
        obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))
        assert obs.reward is not None
        results.append(("Tool: observe_network", True, f"reward={obs.reward}"))
    except Exception as e:
        results.append(("Tool: observe_network", False, str(e)))

    # Test 8: Grading
    try:
        env2 = CyberRangeEnvironment()
        obs = env2.reset(task_id="script_kiddie", seed=42)
        for _ in range(15):
            obs = env2.step(CallToolAction(tool_name="observe_network", arguments={}))
            if obs.done:
                break
        state = env2.state
        grader = getattr(state, "grader_result", {})
        score = grader.get("final_score", -1)
        assert 0.0 <= score <= 1.0
        results.append(("Deterministic grading", True, f"score={score:.3f}"))
    except Exception as e:
        results.append(("Deterministic grading", False, str(e)))

    # Test 9: Seed reproducibility
    try:
        scores = []
        for _ in range(3):
            env3 = CyberRangeEnvironment()
            obs = env3.reset(task_id="script_kiddie", seed=42)
            for _ in range(15):
                obs = env3.step(CallToolAction(tool_name="observe_network", arguments={}))
                if obs.done:
                    break
            state = env3.state
            grader = getattr(state, "grader_result", {})
            scores.append(grader.get("final_score", -1))
        assert len(set(scores)) == 1
        results.append(("Seed reproducibility", True, f"3 runs = {scores[0]:.3f}"))
    except Exception as e:
        results.append(("Seed reproducibility", False, str(e)))

    # Test 10: MITRE coverage
    try:
        all_techs = set()
        for sid, cfg in SCENARIOS.items():
            for phase in cfg.attack_phases:
                if phase.mitre_technique_id:
                    all_techs.add(phase.mitre_technique_id)
        assert len(all_techs) >= 10
        results.append(("MITRE ATT&CK coverage", True, f"{len(all_techs)} techniques mapped"))
    except Exception as e:
        results.append(("MITRE ATT&CK coverage", False, str(e)))

    # Test 11: Adaptive adversary
    try:
        adaptive_scenarios = [sid for sid, cfg in SCENARIOS.items() if cfg.adversary_behavior != "static"]
        assert len(adaptive_scenarios) >= 3
        results.append(("Adaptive adversary", True, f"{len(adaptive_scenarios)} scenarios have adaptive adversaries"))
    except Exception as e:
        results.append(("Adaptive adversary", False, str(e)))

    return results


def main():
    print("=" * 60)
    print("  CyberRange Benchmark Suite")
    print("=" * 60)
    print()

    results = run_validation()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    for name, ok, detail in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        print(f"         {detail}")

    print()
    print(f"  Results: {passed}/{total} tests passed")
    print()

    if passed == total:
        print("  🎉 All tests passed! Environment is ready for training.")
    else:
        print("  ⚠️  Some tests failed. Review the output above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
