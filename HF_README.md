---
title: CyberRange Environment Server
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# 🛡️ CyberRange — SOC Analyst RL Environment

**A self-improving cybersecurity training environment for reinforcement learning agents.**

CyberRange simulates a 12-node enterprise network under active cyber attack. An AI agent (SOC analyst) must detect threats, triage false positives, contain attackers, and eradicate persistent malware — all under budget and time constraints.

## Key Features

| Feature | Details |
|---------|---------|
| **6 Scenarios** | Script Kiddie → Supply Chain → Insider+APT (easy→nightmare) |
| **12 SOC Tools** | observe, investigate, isolate, block, forensics, patch, restore, dismiss, honeypot, escalate, save_playbook, search_playbooks |
| **Adversarial Designer** | LLM generates novel attack scenarios targeting agent weaknesses |
| **3-Persona LLM Judge** | Junior Analyst + Senior Analyst + Incident Commander evaluation |
| **MITRE ATT&CK** | 16 techniques across 8 tactics |
| **GRPO Training** | Full training pipeline with TRL GRPOTrainer integration |
| **Persistent Skill Library** | SQLite-backed playbook store for cross-episode learning |

## API

```python
from cyber_range.server.cyber_environment import CyberRangeEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

env = CyberRangeEnvironment()
obs = env.reset(task_id="script_kiddie", seed=42)
obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))
```

## Links

- **GitHub**: [softsideof/cyber_range](https://github.com/softsideof/cyber_range)
