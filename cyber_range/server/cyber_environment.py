"""Core MCPEnvironment implementation — 10 SOC analyst tools."""


from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

try:
    from .network_simulator import NetworkSimulator
    from .attack_engine import AttackEngine
    from .reward_calculator import RewardCalculator
    from .cyber_judge import CyberJudge, EpisodeLogger
except ImportError:
    from cyber_range.server.network_simulator import NetworkSimulator
    from cyber_range.server.attack_engine import AttackEngine
    from cyber_range.server.reward_calculator import RewardCalculator
    from cyber_range.server.cyber_judge import CyberJudge, EpisodeLogger


class CyberRangeEnvironment(MCPEnvironment):
    """
    A simulated SOC environment where an AI agent learns
    incident detection, investigation, and response.

    The agent interacts via 10 MCP tools that model real SOC analyst actions.
    Each tool call advances the simulation: the attacker progresses,
    new alerts may fire, and the agent receives a reward signal.

    Supports 6 task scenarios:
        - script_kiddie (easy): Single brute-force attack
        - phishing_campaign (medium): Multi-host phishing with false positives
        - apt_lateral_movement (hard): Full APT kill chain
        - ransomware_outbreak (hard): Time-critical ransomware lateral spread
        - supply_chain_compromise (hard): Trojaned software update with C2
        - insider_threat_apt (nightmare): Dual simultaneous threat

    Each scenario has a deterministic grader producing scores from 0.0 to 1.0.
    """

    def __init__(self) -> None:
        """Initialize the CyberRange environment with MCP tools."""
        self.network = NetworkSimulator()
        self.attack_engine = AttackEngine()
        self.reward_calc = RewardCalculator()
        self.judge = CyberJudge()
        self.episode_logger = EpisodeLogger()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario_id: str = "script_kiddie"
        self._max_steps: int = 15
        self._episode_done: bool = False
        self._episode_events: list[str] = []
        self._last_reward: float = 0.0
        self._grader_result: Optional[dict] = None

        # Create MCP server and register all 10 tools
        mcp = FastMCP("cyber_range")


        @mcp.tool
        def observe_network() -> dict:
            """
            Get the current state of the enterprise network.

            Returns the full network topology, pending alerts, threat level,
            health score, budget remaining, and episode progress. Call this
            first to understand the situation before taking action.

            Returns:
                Dictionary with network_topology, pending_alerts, threat_level,
                health_score, budget_remaining, step, max_steps, scenario,
                active_incidents, and episode_score.
            """
            from cyber_range.models import ActionResult
            self._last_action_result = ActionResult(
                action_type="observe_network", success=True,
                description="Network observation complete.",
                resource_cost=0.0,
            )
            return self._build_full_observation()


        @mcp.tool
        def investigate_alert(alert_id: str) -> dict:
            """
            Perform a deep investigation into a specific security alert.

            This reveals whether the alert is a real threat or a false positive,
            along with detailed evidence and recommendations. Costs some time
            and budget but provides critical intelligence.

            Args:
                alert_id: The ID of the alert to investigate (e.g., 'ALT-0001').

            Returns:
                Investigation results including is_false_positive, evidence,
                and recommended next actions.
            """
            result = self.network.investigate_alert(alert_id)
            self._last_action_result = result
            self._advance_simulation()
            return {
                "action": "investigate_alert",
                "result": result.description,
                "success": result.success,
                "details": result.details,
                "reward": self._last_reward,
                "network_summary": self._build_status_summary(),
            }


        @mcp.tool
        def isolate_host(node_id: str) -> dict:
            """
            Quarantine a host from the network to contain a threat.

            The host will be disconnected from all network segments. Use this
            when you've confirmed a host is compromised. WARNING: Isolating a
            healthy critical host causes business disruption (negative reward).

            Args:
                node_id: The ID of the host to isolate (e.g., 'web-01', 'ws-02').

            Returns:
                Isolation result, whether a threat was neutralized, and
                whether business disruption occurred.
            """
            result = self.network.isolate_host(node_id)
            if result.success and result.threat_neutralized:
                self.network.mark_alerts_resolved_for_node(node_id)
            self._last_action_result = result
            self._advance_simulation()
            return {
                "action": "isolate_host",
                "result": result.description,
                "success": result.success,
                "threat_neutralized": result.threat_neutralized,
                "business_disruption": result.healthy_host_isolated,
                "reward": self._last_reward,
                "network_summary": self._build_status_summary(),
            }


        @mcp.tool
        def block_ip(ip_address: str) -> dict:
            """
            Block an IP address at the perimeter firewall.

            Use this to block external attacker IPs. Be careful not to block
            internal IPs unless absolutely necessary, as this disrupts services.

            Args:
                ip_address: The IP address to block (e.g., '185.220.101.42').

            Returns:
                Block result and whether the IP was an attacker or internal host.
            """
            result = self.network.block_ip(ip_address)
            if result.success and result.threat_neutralized:
                self.network.mark_alerts_resolved_for_ip(ip_address)
            self._last_action_result = result
            self._advance_simulation()
            return {
                "action": "block_ip",
                "result": result.description,
                "success": result.success,
                "threat_neutralized": result.threat_neutralized,
                "reward": self._last_reward,
                "network_summary": self._build_status_summary(),
            }


        @mcp.tool
        def run_forensics(node_id: str) -> dict:
            """
            Run memory and disk forensics on a host.

            Performs deep analysis to find malware, suspicious processes,
            anomalous connections, and evidence of credential theft. Expensive
            in time and budget but reveals critical intelligence.

            Args:
                node_id: The ID of the host to analyze (e.g., 'dc-01').

            Returns:
                Forensic findings including malware, suspicious processes,
                anomalous connections, and recommendations.
            """
            result = self.network.run_forensics(node_id)
            self._last_action_result = result
            self._advance_simulation()
            return {
                "action": "run_forensics",
                "result": result.description,
                "success": result.success,
                "details": result.details,
                "reward": self._last_reward,
                "network_summary": self._build_status_summary(),
            }


        @mcp.tool
        def deploy_patch(node_id: str) -> dict:
            """
            Push a security patch to a vulnerable host.

            Patches known vulnerabilities on the specified host. Cannot be
            applied to isolated, offline, or encrypted hosts.

            Args:
                node_id: The ID of the host to patch (e.g., 'web-01').

            Returns:
                Patch result and list of vulnerabilities patched.
            """
            result = self.network.deploy_patch(node_id)
            self._last_action_result = result
            self._advance_simulation()
            return {
                "action": "deploy_patch",
                "result": result.description,
                "success": result.success,
                "details": result.details,
                "reward": self._last_reward,
                "network_summary": self._build_status_summary(),
            }


        @mcp.tool
        def restore_backup(node_id: str) -> dict:
            """
            Restore a compromised or encrypted host from its backup.

            This is expensive but fully removes malware and restores the host
            to a clean state. Only works on compromised/encrypted/isolated hosts.

            Args:
                node_id: The ID of the host to restore (e.g., 'ws-01').

            Returns:
                Restoration result and whether the threat was neutralized.
            """
            result = self.network.restore_backup(node_id)
            if result.success and result.threat_neutralized:
                self.network.mark_alerts_resolved_for_node(node_id)
            self._last_action_result = result
            self._advance_simulation()
            return {
                "action": "restore_backup",
                "result": result.description,
                "success": result.success,
                "threat_neutralized": result.threat_neutralized,
                "reward": self._last_reward,
                "network_summary": self._build_status_summary(),
            }


        @mcp.tool
        def dismiss_alert(alert_id: str) -> dict:
            """
            Dismiss an alert as a false positive.

            Use this when your investigation determines an alert is benign.
            Correctly dismissing false positives is rewarded. WARNING: Dismissing
            a real threat is severely penalized.

            Args:
                alert_id: The ID of the alert to dismiss (e.g., 'ALT-0004').

            Returns:
                Dismissal result. If the alert was indeed a false positive,
                you receive a positive reward.
            """
            result = self.network.dismiss_alert(alert_id)
            self._last_action_result = result
            self._advance_simulation()
            return {
                "action": "dismiss_alert",
                "result": result.description,
                "success": result.success,
                "correctly_dismissed": result.false_positive_correctly_dismissed,
                "real_threat_missed": result.real_threat_ignored,
                "reward": self._last_reward,
                "network_summary": self._build_status_summary(),
            }


        @mcp.tool
        def deploy_honeypot() -> dict:
            """
            Deploy a honeypot server to attract and observe attacker activity.

            The honeypot gathers intelligence about the attacker's tools and
            techniques. Can only be deployed once per episode.

            Returns:
                Deployment result and initial intelligence gathered.
            """
            result = self.network.deploy_honeypot()
            self._last_action_result = result
            self._advance_simulation()
            return {
                "action": "deploy_honeypot",
                "result": result.description,
                "success": result.success,
                "intel_gathered": result.intel_gathered,
                "reward": self._last_reward,
                "network_summary": self._build_status_summary(),
            }


        @mcp.tool
        def escalate_incident(description: str) -> dict:
            """
            Escalate an incident to a senior analyst for review.

            This is a safe fallback when you're uncertain, but it costs
            resources and time. Provide a clear description of what you've
            found and why you're escalating.

            Args:
                description: A description of the incident and reason for escalation.

            Returns:
                Escalation confirmation and resource cost.
            """
            result = self.network.escalate_incident(description)
            self._last_action_result = result
            self._advance_simulation()
            return {
                "action": "escalate_incident",
                "result": result.description,
                "success": result.success,
                "reward": self._last_reward,
                "network_summary": self._build_status_summary(),
            }


        @mcp.tool
        def save_playbook(name: str, description: str, steps: list) -> dict:
            """
            Save a successful investigation sequence as a reusable playbook.

            Use this AFTER successfully resolving an incident to capture
            the effective strategy for future similar incidents. Playbooks
            help you respond faster to recurring attack patterns.

            Args:
                name:        Short identifier (e.g., 'ransomware_triage', 'apt_fp_first')
                description: When to use this playbook and what scenario type it fits
                steps:       Ordered list of tool calls that worked (e.g.,
                             ['investigate_alert ALT-0001', 'dismiss_alert ALT-0002', 'block_ip 185.x.x.x'])

            Returns:
                Confirmation the playbook was saved and its ID
            """
            from cyber_range.server.playbook_store import PlaybookStore
            store = PlaybookStore.get_instance()
            playbook_id = store.save(
                name=name,
                description=description,
                steps=steps,
                scenario_id=self._scenario_id,
            )
            return {
                "action": "save_playbook",
                "playbook_id": playbook_id,
                "success": True,
                "message": f"Playbook '{name}' saved successfully (ID: {playbook_id})",
                "reward": 0.0,
            }

        @mcp.tool
        def search_playbooks(query: str) -> dict:
            """
            Search your saved playbooks for strategies matching the current situation.

            Call this at the START of an episode to see if you have a proven
            strategy for this type of incident. If a matching playbook is found,
            follow its steps in order.

            Args:
                query: Natural language description of the current situation
                       (e.g., 'ransomware lateral movement with false positives',
                              'APT exfiltration database')

            Returns:
                List of matching playbooks with their steps and success rates
            """
            from cyber_range.server.playbook_store import PlaybookStore
            store = PlaybookStore.get_instance()
            matches = store.search(query, top_k=3)
            return {
                "action": "search_playbooks",
                "matches_found": len(matches),
                "playbooks": matches,
                "tip": "Follow the steps of a matching playbook in order for best results.",
                "reward": 0.0,
            }

        # Initialize MCPEnvironment with our tool server
        super().__init__(mcp)
        self._last_action_result = None


    # ========================================================================
    # OpenEnv API: reset / step / state
    # ========================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Start a new incident response episode.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Optional custom episode ID.
            **kwargs: Additional parameters. Supports:
                task_id (str): Scenario to load ('script_kiddie', 'phishing_campaign',
                               'apt_lateral_movement'). Defaults to 'script_kiddie'.

        Returns:
            Initial Observation with network state, alerts, and scenario description.
        """
        task_id = kwargs.get("task_id", "script_kiddie")
        self._scenario_id = task_id

        # Initialize network
        self.network.initialize(seed=seed)

        # Load attack scenario
        try:
            scenario = self.attack_engine.load_scenario(task_id, self.network, seed=seed)
        except ValueError:
            scenario = self.attack_engine.load_scenario("script_kiddie", self.network, seed=seed)
            self._scenario_id = "script_kiddie"

        self._max_steps = scenario.max_steps

        # Reset reward tracking
        self.reward_calc.reset()
        self.episode_logger.reset()
        self._episode_done = False
        self._episode_events = []
        self._last_reward = 0.0
        self._grader_result = None
        self._last_action_result = None

        # Reset state
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Build initial observation
        initial_obs = self._build_full_observation()
        initial_obs["scenario"] = {
            "id": scenario.scenario_id,
            "name": scenario.name,
            "description": scenario.description,
            "difficulty": scenario.difficulty.value,
            "max_steps": scenario.max_steps,
        }
        initial_obs["available_actions"] = [
            "observe_network", "investigate_alert", "isolate_host", "block_ip",
            "run_forensics", "deploy_patch", "restore_backup", "dismiss_alert",
            "deploy_honeypot", "escalate_incident",
        ]

        return Observation(
            done=False,
            reward=0.0,
            metadata=initial_obs,
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Increments the step counter, delegates MCP tool execution to the
        base class, then applies stored reward and done status.

        Note: Reward computation and metric updates happen inside the tool
        functions via _advance_simulation(). This method just plumbs those
        values onto the observation returned by the MCP framework.
        """
        self._state.step_count += 1
        self.network.increment_step()
        self._last_action_result = None
        self._last_reward = 0.0  # Reset so tools with no cost report 0

        # Execute the tool via MCPEnvironment
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Apply the reward computed by the tool function's _advance_simulation()
        if self._last_action_result is not None:
            obs.reward = self._last_reward

        # Check termination conditions
        done = self._check_done()
        obs.done = done

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step used by the WebSocket handler."""
        self._state.step_count += 1
        self.network.increment_step()
        self._last_action_result = None
        self._last_reward = 0.0

        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        # Apply the reward computed by the tool function's _advance_simulation()
        if self._last_action_result is not None:
            obs.reward = self._last_reward

        # Check termination conditions
        done = self._check_done()
        obs.done = done

        return obs

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (return error)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions. "
                "Available tools: observe_network, investigate_alert, isolate_host, "
                "block_ip, run_forensics, deploy_patch, restore_backup, dismiss_alert, "
                "deploy_honeypot, escalate_incident.",
            },
        )

    @property
    def state(self) -> State:
        """Return current episode state with CyberRange-specific metadata."""
        # State has extra="allow" so we can add custom fields
        state_data = {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "scenario_id": self._scenario_id,
            "max_steps": self._max_steps,
            "episode_done": self._episode_done,
            "cumulative_reward": self.reward_calc.cumulative_reward,
            "threat_level": self.network.calculate_threat_level(),
            "health_score": self.network.health_score(),
            "grader_result": self._grader_result,
        }
        return State(**state_data)

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    def _advance_simulation(self) -> None:
        """Advance the attack engine and compute reward."""
        events = self.attack_engine.advance(self.network)
        self._episode_events.extend(events)

        # Compute reward for the last action
        if self._last_action_result is not None:
            reward = self.reward_calc.calculate(self._last_action_result)
            self._last_reward = reward
            self.attack_engine.update_metrics(self._last_action_result)

            # Log to episode logger for LLM judge
            self.episode_logger.record(
                step=self._state.step_count,
                action=self._last_action_result.action_type,
                args=getattr(self._last_action_result, "action_args", {}),
                reward=reward,
            )

    def _check_done(self) -> bool:
        """Check episode termination conditions."""
        if self._episode_done:
            return True

        done = (
            self._state.step_count >= self._max_steps
            or self.attack_engine.is_fully_contained()
            or self.network.is_catastrophic_breach()
            or self.network.budget_remaining() <= 0
        )

        if done:
            self._episode_done = True
            if self._grader_result is None:
                self._grader_result = self.attack_engine.grade_episode(
                    self.network, self._state.step_count
                )
                # Apply episode-end efficiency bonus / timeout penalty
                metrics = self.attack_engine.metrics
                end_bonus = self.reward_calc.apply_episode_end_bonuses(
                    threats_neutralized=metrics.threats_neutralized,
                    total_threats=metrics.total_threats,
                    steps_used=self._state.step_count,
                    max_steps=self._max_steps,
                )
                self._grader_result["episode_end_bonus"] = end_bonus
                self._grader_result["total_episode_reward"] = self.reward_calc.cumulative_reward

                # Run LLM multi-persona judge and merge scores
                det_score = self._grader_result.get("final_score", 0.0)
                scenario_meta = {
                    **self._grader_result.get("details", {}),
                    "scenario_name": getattr(self.attack_engine.scenario, "name", ""),
                    "difficulty": getattr(self.attack_engine.scenario, "difficulty", ""),
                    "adversary_behavior": getattr(self.attack_engine.scenario, "adversary_behavior", ""),
                    "threat_count": getattr(self.attack_engine.scenario, "threat_count", 0),
                    "false_positive_count": getattr(self.attack_engine.scenario, "false_positive_count", 0),
                    "max_steps": self._max_steps,
                }
                judge_result = self.judge.evaluate(
                    self.episode_logger.to_judge_format(),
                    scenario_meta,
                    det_score,
                )
                self._grader_result["judge"] = judge_result
                # If judge is enabled, use combined score as the primary score
                if judge_result["judge_enabled"]:
                    self._grader_result["final_score"] = judge_result["combined_score"]
                    self._grader_result["deterministic_score"] = det_score
                self._log_episode_results()

        return done

    def _log_episode_results(self) -> None:
        """Export the final episode results to outputs/evals/."""
        import json
        import time
        from pathlib import Path
        
        base_dir = Path(__file__).parent.parent.resolve()
        eval_dir = base_dir / "outputs" / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        if self._grader_result:
            log_data = {
                "episode_id": self._state.episode_id,
                "timestamp": time.time(),
                "scenario_id": self._scenario_id,
                "steps": self._state.step_count,
                "cumulative_reward": self.reward_calc.cumulative_reward,
                "grader_result": self._grader_result
            }
            
            try:
                with open(eval_dir / f"eval_{self._state.episode_id}.json", "w") as f:
                    json.dump(log_data, f, indent=2)
            except Exception:
                pass

    def _build_full_observation(self) -> dict:
        """Build the complete agent observation."""
        return {
            "network_topology": self.network.get_visible_topology(),
            "pending_alerts": self.network.get_pending_alerts(),
            "resolved_alerts": self.network.get_resolved_alert_ids(),
            "active_incidents": self.attack_engine.get_active_incidents(),
            "threat_level": self.network.calculate_threat_level(),
            "health_score": self.network.health_score(),
            "budget_remaining": self.network.budget_remaining(),
            "step": self._state.step_count,
            "max_steps": self._max_steps,
            "cumulative_reward": self.reward_calc.cumulative_reward,
            "episode_done": self._episode_done,
            "recent_events": self._episode_events[-5:] if self._episode_events else [],
            "honeypot_intel": self.network.honeypot_intel[-3:] if self.network.honeypot_intel else [],
            "grader_result": self._grader_result,
        }

    def _build_status_summary(self) -> dict:
        """Build a compact status summary for tool responses."""
        return {
            "threat_level": self.network.calculate_threat_level(),
            "health_score": self.network.health_score(),
            "budget_remaining": self.network.budget_remaining(),
            "step": self._state.step_count,
            "max_steps": self._max_steps,
            "pending_alerts_count": len(self.network.get_pending_alerts()),
            "compromised_hosts": self.network.compromised_count(),
            "active_incidents": len(self.attack_engine.get_active_incidents()),
            "episode_done": self._episode_done,
            "cumulative_reward": self.reward_calc.cumulative_reward,
        }
