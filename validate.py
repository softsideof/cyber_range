"""
Pre-submission manual validation script.
Mirrors what `openenv validate` checks.
"""
import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log = open("validation_results.log", "w", encoding="utf-8")
def p(msg):
    print(msg)
    log.write(msg + "\n")
    log.flush()

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status))
    p(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

p("=" * 60)
p("  CyberRange Pre-Submission Validation")
p("=" * 60)

# 1. STRUCTURE CHECKS
p("\n--- Structure ---")

check("openenv.yaml exists", 
      os.path.isfile("openenv.yaml"))

check("models.py exists", 
      os.path.isfile("cyber_range/models.py"))

check("client.py exists", 
      os.path.isfile("cyber_range/client.py"))

check("__init__.py exists", 
      os.path.isfile("cyber_range/__init__.py"))

check("server/app.py exists", 
      os.path.isfile("cyber_range/server/app.py"))

check("server/cyber_environment.py exists",
      os.path.isfile("cyber_range/server/cyber_environment.py"))

check("server/Dockerfile exists",
      os.path.isfile("cyber_range/server/Dockerfile"))

check("inference.py in root",
      os.path.isfile("inference.py"))

check("README.md exists",
      os.path.isfile("README.md"))

check("pyproject.toml exists",
      os.path.isfile("pyproject.toml"))

# 2. OPENENV.YAML VALIDATION
p("\n--- openenv.yaml ---")
import yaml
with open("openenv.yaml") as f:
    config = yaml.safe_load(f)

check("spec_version present", "spec_version" in config, str(config.get("spec_version")))
check("name present", "name" in config, config.get("name"))
check("type present", "type" in config, config.get("type"))
check("runtime present", "runtime" in config, config.get("runtime"))
check("app present", "app" in config, config.get("app"))
check("port present", "port" in config, str(config.get("port")))

# 3. IMPORT CHECKS
p("\n--- Imports ---")

try:
    from cyber_range.server.cyber_environment import CyberRangeEnvironment
    check("CyberRangeEnvironment imports", True)
except Exception as e:
    check("CyberRangeEnvironment imports", False, str(e))

try:
    from cyber_range.client import CyberRangeEnv
    check("CyberRangeEnv (client) imports", True)
except Exception as e:
    check("CyberRangeEnv (client) imports", False, str(e))

try:
    from cyber_range import CyberRangeEnv, CallToolAction, ListToolsAction
    check("Package exports correct", True)
except Exception as e:
    check("Package exports correct", False, str(e))

try:
    from cyber_range.server.app import app
    check("FastAPI app imports", True)
except Exception as e:
    check("FastAPI app imports", False, str(e))

# 4. TYPED MODELS
p("\n--- Typed Models ---")

try:
    from openenv.core.env_server.types import Action, Observation, State
    check("Action/Observation/State imports", True)
except Exception as e:
    check("Action/Observation/State imports", False, str(e))

try:
    env = CyberRangeEnvironment()
    obs = env.reset(task_id="script_kiddie", seed=42)
    check("reset() returns Observation", isinstance(obs, Observation))
    check("Observation.done is bool", isinstance(obs.done, bool))
    check("Observation.reward is float", isinstance(obs.reward, (int, float)))
    check("Observation.metadata is dict", isinstance(obs.metadata, dict))
except Exception as e:
    check("reset() works", False, str(e))

try:
    from openenv.core.env_server.mcp_types import CallToolAction
    obs2 = env.step(CallToolAction(tool_name="observe_network", arguments={}))
    check("step() returns Observation", isinstance(obs2, Observation))
except Exception as e:
    check("step() works", False, str(e))

try:
    state = env.state
    check("state property returns State", isinstance(state, State))
    check("state.episode_id is str", isinstance(state.episode_id, str))
    check("state.step_count is int", isinstance(state.step_count, int))
except Exception as e:
    check("state works", False, str(e))

# 5. TOOL DISCOVERY
p("\n--- Tool Discovery ---")

try:
    tool_obs = env.step(ListToolsAction())
    tools = getattr(tool_obs, "tools", [])
    tool_names = sorted([t.name for t in tools])
    check("ListToolsAction returns tools", len(tools) >= 10, f"found {len(tools)}")
    expected_core = sorted([
        "observe_network", "investigate_alert", "isolate_host", "block_ip",
        "run_forensics", "deploy_patch", "restore_backup", "dismiss_alert",
        "deploy_honeypot", "escalate_incident",
    ])
    check("All 10 core tools present", all(t in tool_names for t in expected_core), str(tool_names))
except Exception as e:
    check("Tool discovery", False, str(e))

# 6. 3+ TASKS WITH GRADERS
p("\n--- Tasks & Graders ---")

TASKS = ["script_kiddie", "phishing_campaign", "apt_lateral_movement"]
check("3+ tasks defined", len(TASKS) >= 3, f"{len(TASKS)} tasks")

all_scores = []
for task_id in TASKS:
    env_t = CyberRangeEnvironment()
    obs_t = env_t.reset(task_id=task_id, seed=42)
    max_steps = obs_t.metadata.get("scenario", {}).get("max_steps", 15)

    # Run a few actions
    env_t.step(CallToolAction(tool_name="observe_network", arguments={}))
    env_t.step(CallToolAction(tool_name="block_ip", arguments={"ip_address": "185.220.101.42"}))

    # Run to completion
    for _ in range(max_steps):
        if env_t.state.step_count >= max_steps: break
        r = env_t.step(CallToolAction(tool_name="observe_network", arguments={}))
        if r.done: break

    grader = getattr(env_t.state, "grader_result", {})
    score = grader.get("final_score", -1.0)
    all_scores.append(score)
    check(f"Task '{task_id}' grader score in [0,1]", 0.0 <= score <= 1.0, f"score={score}")

check("Graders produce varied scores", len(set(all_scores)) > 1,
      f"scores={all_scores}")

# 7. INFERENCE SCRIPT
p("\n--- Inference Script ---")

check("inference.py uses OpenAI client",
      "from openai import OpenAI" in open("inference.py").read())

check("inference.py reads API_BASE_URL",
      "API_BASE_URL" in open("inference.py").read())

check("inference.py reads MODEL_NAME",
      "MODEL_NAME" in open("inference.py").read())

check("inference.py reads HF_TOKEN",
      "HF_TOKEN" in open("inference.py").read())

# 8. DOCKERFILE
p("\n--- Dockerfile ---")

dockerfile_content = open("cyber_range/server/Dockerfile").read()
check("Dockerfile has EXPOSE", "EXPOSE" in dockerfile_content)
check("Dockerfile has HEALTHCHECK", "HEALTHCHECK" in dockerfile_content)
check("Dockerfile has CMD", "CMD" in dockerfile_content)

# SUMMARY
p(f"\n{'='*60}")
passed = sum(1 for _, s in results if s == PASS)
failed = sum(1 for _, s in results if s == FAIL)
p(f"  TOTAL: {passed} passed, {failed} failed out of {len(results)} checks")
if failed == 0:
    p("  STATUS: ALL CHECKS PASSED!")
else:
    p("  STATUS: SOME CHECKS FAILED")
    for name, status in results:
        if status == FAIL:
            p(f"    FAILED: {name}")
p(f"{'='*60}")

log.close()
