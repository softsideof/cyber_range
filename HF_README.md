---
title: CyberRange Environment Server
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ CyberRange — SOC Analyst RL Environment

**An adaptive cybersecurity training environment for reinforcement learning agents.**

Built for the **Meta PyTorch × Scaler OpenEnv Hackathon** using the OpenEnv 0.2.2 framework.

## What This Does

CyberRange simulates a 12-node enterprise network under active cyber attack. An AI agent (SOC analyst) must:

- **Detect** threats by investigating SIEM alerts
- **Triage** false positives from real attacks using forensic evidence
- **Contain** threats by blocking attacker IPs and isolating compromised hosts
- **Eradicate** persistent malware via backup restoration

## Key Features

| Feature | Details |
|---------|---------|
| **6 Scenarios** | Script Kiddie → Supply Chain → Insider+APT (easy→nightmare) |
| **10 SOC Tools** | observe, investigate, isolate, block, forensics, patch, restore, dismiss, honeypot, escalate |
| **Adaptive Adversary** | 4 behavior modes: static, evasive, persistent, adaptive |
| **MITRE ATT&CK** | 16 techniques across 8 tactics |
| **Realistic Forensics** | Process trees, SHA-256 hashes, VirusTotal scores, Cobalt Strike signatures |
| **Multi-Objective Grading** | Threats (35%) + FP handling (20%) + Data protection (20%) + Collateral (15%) + Efficiency (10%) |
| **Gymnasium Wrapper** | Compatible with TRL, RLlib, Stable Baselines 3 |

## API Usage

```python
from cyber_range import CyberRangeEnv
from openenv.core.env_server.mcp_types import CallToolAction

with CyberRangeEnv.from_env("keshav-005/cyber_range") as env:
    result = await env.step(CallToolAction(tool_name="observe_network", arguments={}))
```

## Links

- **GitHub**: [keshav-005/cyber_range](https://github.com/keshav-005/cyber_range)
- **Inference**: Heuristic + LLM (Llama 3.3 70B) agent with evidence-driven decision making
- **Seed**: 42 (fully reproducible)
