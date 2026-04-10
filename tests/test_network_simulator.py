# Copyright (c) 2026. CyberRange OpenEnv Environment.
# Licensed under the BSD-3-Clause License.

"""
Tests for NetworkSimulator — the 12-node enterprise network simulation.

Validates:
    - Network initialization and topology
    - Each defensive action (isolate, block, forensics, patch, restore, dismiss)
    - Edge cases (double isolation, invalid nodes, budget depletion)
    - Health score and threat level calculations
"""

import pytest

from cyber_range.models import ActionResult, NodeStatus, NodeType

SEED = 42


# ============================================================================
# Initialization Tests
# ============================================================================

class TestNetworkInit:
    """Test network initialization and topology."""

    def test_initializes_12_nodes(self, network):
        assert len(network.nodes) == 12

    def test_all_nodes_start_healthy(self, network):
        for node in network.nodes.values():
            assert node.status == NodeStatus.HEALTHY

    def test_has_critical_nodes(self, network):
        critical = [n for n in network.nodes.values() if n.is_critical]
        assert len(critical) >= 4  # fw, dc, web, mail, db, backup

    def test_has_correct_node_types(self, network):
        types = {n.node_type for n in network.nodes.values()}
        assert NodeType.FIREWALL in types
        assert NodeType.DOMAIN_CONTROLLER in types
        assert NodeType.WEB_SERVER in types
        assert NodeType.DATABASE in types
        assert NodeType.WORKSTATION in types
        assert NodeType.HONEYPOT in types

    def test_initial_budget_is_100(self, network):
        assert network.budget_remaining() == 100.0

    def test_health_score_starts_at_1(self, network):
        assert network.health_score() == 1.0

    def test_no_initial_alerts(self, network):
        assert len(network.get_pending_alerts()) == 0

    def test_no_blocked_ips(self, network):
        assert len(network.blocked_ips) == 0


# ============================================================================
# Investigate Alert Tests
# ============================================================================

class TestInvestigateAlert:
    """Test the investigate_alert action."""

    def test_nonexistent_alert_fails(self, network):
        result = network.investigate_alert("ALT-9999")
        assert not result.success
        assert result.resource_cost > 0

    def test_investigate_marks_alert_investigated(self, network):
        from cyber_range.models import AlertSeverity, AlertType, NetworkAlert
        network.add_alert(NetworkAlert(
            alert_id="ALT-TEST", timestamp=0.0,
            severity=AlertSeverity.HIGH,
            source_ip="1.2.3.4", destination_ip="10.0.2.1",
            alert_type=AlertType.BRUTE_FORCE,
            description="Test alert", confidence=0.9,
            raw_log="test", is_false_positive=False,
            related_node_id="web-01",
        ))
        result = network.investigate_alert("ALT-TEST")
        assert result.success
        assert network.alerts["ALT-TEST"].investigated

    def test_investigate_reveals_false_positive_status(self, network):
        from cyber_range.models import AlertSeverity, AlertType, NetworkAlert
        network.add_alert(NetworkAlert(
            alert_id="ALT-FP", timestamp=0.0,
            severity=AlertSeverity.LOW,
            source_ip="10.0.2.2", destination_ip="10.0.0.1",
            alert_type=AlertType.ANOMALOUS_TRAFFIC,
            description="Benign traffic", confidence=0.3,
            raw_log="test", is_false_positive=True,
            related_node_id="mail-01",
        ))
        result = network.investigate_alert("ALT-FP")
        assert "forensic_evidence" in result.details
        assert "benign" in result.details["forensic_evidence"].lower()

    def test_investigate_costs_budget(self, network):
        from cyber_range.models import AlertSeverity, AlertType, NetworkAlert
        network.add_alert(NetworkAlert(
            alert_id="ALT-B", timestamp=0.0,
            severity=AlertSeverity.HIGH,
            source_ip="1.2.3.4", destination_ip="10.0.2.1",
            alert_type=AlertType.BRUTE_FORCE,
            description="Test", confidence=0.9, raw_log="test",
            related_node_id="web-01",
        ))
        budget_before = network.budget_remaining()
        network.investigate_alert("ALT-B")
        assert network.budget_remaining() < budget_before


# ============================================================================
# Isolate Host Tests
# ============================================================================

class TestIsolateHost:
    """Test the isolate_host action."""

    def test_isolate_valid_node(self, network):
        result = network.isolate_host("ws-01")
        assert result.success
        assert network.nodes["ws-01"].status == NodeStatus.ISOLATED

    def test_isolate_healthy_penalizes(self, network):
        result = network.isolate_host("ws-01")
        assert result.healthy_host_isolated is True

    def test_isolate_compromised_neutralizes(self, network):
        network.compromise_node("ws-01", step=0)
        result = network.isolate_host("ws-01")
        assert result.threat_neutralized is True
        assert result.healthy_host_isolated is False

    def test_isolate_critical_disrupts(self, network):
        result = network.isolate_host("dc-01")
        assert result.critical_services_disrupted is True

    def test_double_isolate_fails(self, network):
        network.isolate_host("ws-01")
        result = network.isolate_host("ws-01")
        assert not result.success

    def test_isolate_nonexistent_fails(self, network):
        result = network.isolate_host("nonexistent-99")
        assert not result.success

    def test_isolate_costs_budget(self, network):
        budget_before = network.budget_remaining()
        network.isolate_host("ws-01")
        assert network.budget_remaining() < budget_before


# ============================================================================
# Block IP Tests
# ============================================================================

class TestBlockIP:
    """Test the block_ip action."""

    def test_block_external_ip(self, network):
        result = network.block_ip("185.220.101.42")
        assert result.success
        assert result.threat_neutralized is True
        assert "185.220.101.42" in network.blocked_ips

    def test_block_internal_ip_bad(self, network):
        result = network.block_ip("10.0.2.1")
        assert result.healthy_host_isolated is True

    def test_block_critical_internal_disrupts(self, network):
        result = network.block_ip("10.0.1.1")  # DC IP
        assert result.critical_services_disrupted is True

    def test_double_block_fails(self, network):
        network.block_ip("185.220.101.42")
        result = network.block_ip("185.220.101.42")
        assert not result.success

    def test_block_costs_budget(self, network):
        budget_before = network.budget_remaining()
        network.block_ip("1.2.3.4")
        assert network.budget_remaining() < budget_before


# ============================================================================
# Forensics Tests
# ============================================================================

class TestRunForensics:
    """Test the run_forensics action."""

    def test_forensics_healthy_node(self, network):
        result = network.run_forensics("ws-01")
        assert result.success
        assert result.details["malware_found"] is False

    def test_forensics_compromised_node(self, network):
        network.compromise_node("ws-01", step=0)
        result = network.run_forensics("ws-01")
        assert result.success
        assert result.details["malware_found"] is True
        assert len(result.details["process_tree"]) > 0

    def test_forensics_nonexistent_fails(self, network):
        result = network.run_forensics("fake-99")
        assert not result.success

    def test_forensics_is_expensive(self, network):
        budget_before = network.budget_remaining()
        network.run_forensics("ws-01")
        assert budget_before - network.budget_remaining() >= 5.0


# ============================================================================
# Deploy Patch Tests
# ============================================================================

class TestDeployPatch:
    """Test the deploy_patch action."""

    def test_patch_vulnerable_node(self, network):
        # web-01 has CVE-2024-1234-nginx
        assert len(network.nodes["web-01"].vulnerabilities) > 0
        result = network.deploy_patch("web-01")
        assert result.success
        assert len(network.nodes["web-01"].vulnerabilities) == 0

    def test_patch_no_vulns(self, network):
        result = network.deploy_patch("ws-01")
        assert result.success  # Succeeds but no vulns to patch

    def test_patch_isolated_fails(self, network):
        network.isolate_host("web-01")
        result = network.deploy_patch("web-01")
        assert not result.success

    def test_patch_nonexistent_fails(self, network):
        result = network.deploy_patch("fake-99")
        assert not result.success


# ============================================================================
# Restore Backup Tests
# ============================================================================

class TestRestoreBackup:
    """Test the restore_backup action."""

    def test_restore_compromised_node(self, network):
        network.compromise_node("ws-01", step=0)
        result = network.restore_backup("ws-01")
        assert result.success
        assert result.threat_neutralized is True
        assert network.nodes["ws-01"].status == NodeStatus.HEALTHY

    def test_restore_healthy_fails(self, network):
        result = network.restore_backup("ws-01")
        assert not result.success

    def test_restore_is_very_expensive(self, network):
        network.compromise_node("ws-01", step=0)
        budget_before = network.budget_remaining()
        network.restore_backup("ws-01")
        assert budget_before - network.budget_remaining() >= 8.0


# ============================================================================
# Dismiss Alert Tests
# ============================================================================

class TestDismissAlert:
    """Test the dismiss_alert action."""

    def test_dismiss_nonexistent_fails(self, network):
        result = network.dismiss_alert("ALT-NOPE")
        assert not result.success

    def test_dismiss_false_positive_correct(self, network):
        from cyber_range.models import AlertSeverity, AlertType, NetworkAlert
        network.add_alert(NetworkAlert(
            alert_id="ALT-FP", timestamp=0.0,
            severity=AlertSeverity.LOW,
            source_ip="10.0.6.1", destination_ip="10.0.0.1",
            alert_type=AlertType.ANOMALOUS_TRAFFIC,
            description="Backup spike", confidence=0.3,
            raw_log="test", is_false_positive=True,
            related_node_id="backup-01",
        ))
        result = network.dismiss_alert("ALT-FP")
        assert result.success
        assert result.false_positive_correctly_dismissed is True

    def test_dismiss_real_threat_bad(self, network):
        from cyber_range.models import AlertSeverity, AlertType, NetworkAlert
        network.add_alert(NetworkAlert(
            alert_id="ALT-REAL", timestamp=0.0,
            severity=AlertSeverity.CRITICAL,
            source_ip="1.2.3.4", destination_ip="10.0.2.1",
            alert_type=AlertType.INTRUSION,
            description="Real intrusion", confidence=0.95,
            raw_log="test", is_false_positive=False,
            related_node_id="web-01",
        ))
        result = network.dismiss_alert("ALT-REAL")
        assert result.real_threat_ignored is True

    def test_dismiss_already_dismissed_fails(self, network):
        from cyber_range.models import AlertSeverity, AlertType, NetworkAlert
        network.add_alert(NetworkAlert(
            alert_id="ALT-DUP", timestamp=0.0,
            severity=AlertSeverity.LOW,
            source_ip="10.0.6.1", destination_ip="10.0.0.1",
            alert_type=AlertType.ANOMALOUS_TRAFFIC,
            description="Test", confidence=0.3,
            raw_log="test", is_false_positive=True,
            related_node_id="backup-01",
        ))
        network.dismiss_alert("ALT-DUP")
        result = network.dismiss_alert("ALT-DUP")
        assert not result.success


# ============================================================================
# Honeypot Tests
# ============================================================================

class TestDeployHoneypot:
    """Test the deploy_honeypot action."""

    def test_deploy_honeypot(self, network):
        result = network.deploy_honeypot()
        assert result.success
        assert network.honeypot_deployed is True
        assert result.intel_gathered > 0

    def test_double_deploy_fails(self, network):
        network.deploy_honeypot()
        result = network.deploy_honeypot()
        assert not result.success


# ============================================================================
# Health Score & Threat Level Tests
# ============================================================================

class TestHealthAndThreat:
    """Test health score and threat level calculations."""

    def test_health_decreases_on_compromise(self, network):
        assert network.health_score() == 1.0
        network.compromise_node("ws-01", step=0)
        assert network.health_score() < 1.0

    def test_threat_level_escalates(self, network):
        assert network.calculate_threat_level() == "green"
        network.compromise_node("ws-01", step=0)
        assert network.calculate_threat_level() != "green"

    def test_catastrophic_breach(self, network):
        assert not network.is_catastrophic_breach()
        network.compromise_node("dc-01", step=0)
        network.compromise_node("web-01", step=0)
        network.compromise_node("db-01", step=0)
        assert network.is_catastrophic_breach()

    def test_compromised_count(self, network):
        assert network.compromised_count() == 0
        network.compromise_node("ws-01", step=0)
        network.compromise_node("ws-02", step=0)
        assert network.compromised_count() == 2
