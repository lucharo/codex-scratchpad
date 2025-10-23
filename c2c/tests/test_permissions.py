"""Tests for permission management system."""

import pytest

from c2c.permissions import (
    PermissionAction,
    PermissionDecision,
    PermissionManager,
    PermissionPolicy,
    PermissionRequest,
    RiskLevel,
)


def test_permission_request_creation():
    """Test creating a permission request."""
    request = PermissionRequest(
        request_id="test-req-1",
        session_id="test-session",
        action=PermissionAction.EXECUTE_COMMAND,
        description="Run tests",
        details={"command": "pytest"},
    )

    assert request.request_id == "test-req-1"
    assert request.session_id == "test-session"
    assert request.action == PermissionAction.EXECUTE_COMMAND
    assert request.decision == PermissionDecision.PENDING


def test_permission_policy_matching():
    """Test policy matching logic."""
    policy = PermissionPolicy(
        name="safe-reads",
        action=PermissionAction.READ_FILE,
        decision=PermissionDecision.APPROVED,
        max_risk=RiskLevel.LOW,
    )

    # Should match
    request1 = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.READ_FILE,
        description="Read README",
        risk_level=RiskLevel.LOW,
    )
    assert policy.matches(request1)

    # Should not match - wrong action
    request2 = PermissionRequest(
        request_id="req2",
        session_id="sess1",
        action=PermissionAction.WRITE_FILE,
        description="Write file",
        risk_level=RiskLevel.LOW,
    )
    assert not policy.matches(request2)

    # Should not match - risk too high
    request3 = PermissionRequest(
        request_id="req3",
        session_id="sess1",
        action=PermissionAction.READ_FILE,
        description="Read sensitive file",
        risk_level=RiskLevel.HIGH,
    )
    assert not policy.matches(request3)


def test_permission_policy_pattern_matching():
    """Test policy pattern matching."""
    policy = PermissionPolicy(
        name="safe-git",
        action=PermissionAction.GIT_OPERATION,
        pattern=r"git\s+status",
        decision=PermissionDecision.APPROVED,
    )

    # Should match
    request1 = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.GIT_OPERATION,
        description="Check status",
        details={"command": "git status"},
    )
    assert policy.matches(request1)

    # Should not match - pattern doesn't match
    request2 = PermissionRequest(
        request_id="req2",
        session_id="sess1",
        action=PermissionAction.GIT_OPERATION,
        description="Push changes",
        details={"command": "git push"},
    )
    assert not policy.matches(request2)


def test_permission_manager_init():
    """Test permission manager initialization."""
    manager = PermissionManager()

    assert len(manager.policies) > 0
    assert len(manager.requests) == 0


def test_permission_manager_risk_assessment():
    """Test risk assessment logic."""
    manager = PermissionManager()

    # Low risk - read operation
    request1 = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.READ_FILE,
        description="Read file",
        details={"path": "/tmp/test.txt"},
    )
    assert manager.assess_risk(request1) == RiskLevel.LOW

    # High risk - sensitive file read
    request2 = PermissionRequest(
        request_id="req2",
        session_id="sess1",
        action=PermissionAction.READ_FILE,
        description="Read credentials",
        details={"path": "/home/user/.ssh/id_rsa"},
    )
    assert manager.assess_risk(request2) == RiskLevel.HIGH

    # Critical risk - dangerous command
    request3 = PermissionRequest(
        request_id="req3",
        session_id="sess1",
        action=PermissionAction.EXECUTE_COMMAND,
        description="Delete all",
        details={"command": "rm -rf /"},
    )
    assert manager.assess_risk(request3) == RiskLevel.CRITICAL

    # High risk - file deletion
    request4 = PermissionRequest(
        request_id="req4",
        session_id="sess1",
        action=PermissionAction.DELETE_FILE,
        description="Delete file",
    )
    assert manager.assess_risk(request4) == RiskLevel.HIGH


def test_permission_manager_auto_approve():
    """Test automatic approval of safe actions."""
    manager = PermissionManager()

    request = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.READ_FILE,
        description="Read README",
        risk_level=RiskLevel.LOW,
    )

    result = manager.request_permission(request)

    assert result.decision == PermissionDecision.APPROVED
    assert result.decided_by is not None
    assert "policy" in result.decided_by


def test_permission_manager_auto_deny():
    """Test automatic denial of dangerous actions."""
    manager = PermissionManager()

    request = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.GIT_OPERATION,
        description="Force push",
        details={"command": "git push --force"},
        risk_level=RiskLevel.CRITICAL,
    )

    result = manager.request_permission(request)

    assert result.decision == PermissionDecision.DENIED
    assert result.denial_reason is not None


def test_permission_manager_escalate():
    """Test escalation of medium-risk actions."""
    manager = PermissionManager()

    request = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.DELETE_FILE,
        description="Delete old logs",
        details={"path": "/tmp/old.log"},
    )

    result = manager.request_permission(request)

    assert result.decision == PermissionDecision.ESCALATED


def test_permission_manager_manual_decision():
    """Test making manual decisions on requests."""
    manager = PermissionManager()

    request = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.DELETE_FILE,
        description="Delete file",
    )

    # Initially escalated
    result = manager.request_permission(request)
    assert result.decision == PermissionDecision.ESCALATED

    # User approves
    updated = manager.make_decision(
        "req1", PermissionDecision.APPROVED, decided_by="user"
    )

    assert updated.decision == PermissionDecision.APPROVED
    assert updated.decided_by == "user"
    assert updated.decided_at is not None


def test_permission_manager_deny_with_reason():
    """Test denying with a custom reason."""
    manager = PermissionManager()

    request = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.NETWORK_REQUEST,
        description="Make API call",
    )

    manager.request_permission(request)

    # User denies with reason
    updated = manager.make_decision(
        "req1",
        PermissionDecision.DENIED,
        decided_by="user",
        reason="External API not allowed",
    )

    assert updated.decision == PermissionDecision.DENIED
    assert updated.denial_reason == "External API not allowed"


def test_permission_manager_get_pending():
    """Test getting pending requests."""
    manager = PermissionManager()

    # Create multiple requests
    req1 = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.DELETE_FILE,
        description="Delete file 1",
    )
    req2 = PermissionRequest(
        request_id="req2",
        session_id="sess2",
        action=PermissionAction.NETWORK_REQUEST,
        description="API call",
    )
    req3 = PermissionRequest(
        request_id="req3",
        session_id="sess1",
        action=PermissionAction.READ_FILE,
        description="Read file",
        risk_level=RiskLevel.LOW,
    )

    manager.request_permission(req1)  # escalated
    manager.request_permission(req2)  # escalated
    manager.request_permission(req3)  # approved

    # Get all pending
    pending = manager.get_pending_requests()
    assert len(pending) == 2

    # Get pending for specific session
    sess1_pending = manager.get_pending_requests(session_id="sess1")
    assert len(sess1_pending) == 1
    assert sess1_pending[0].session_id == "sess1"


def test_permission_manager_get_request():
    """Test retrieving a specific request."""
    manager = PermissionManager()

    request = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.WRITE_FILE,
        description="Write config",
    )

    manager.request_permission(request)

    retrieved = manager.get_request("req1")
    assert retrieved is not None
    assert retrieved.request_id == "req1"

    # Non-existent request
    not_found = manager.get_request("nonexistent")
    assert not_found is None


def test_permission_manager_clear_session():
    """Test clearing requests for a session."""
    manager = PermissionManager()

    req1 = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.DELETE_FILE,
        description="Delete file",
    )
    req2 = PermissionRequest(
        request_id="req2",
        session_id="sess2",
        action=PermissionAction.DELETE_FILE,
        description="Delete file",
    )

    manager.request_permission(req1)
    manager.request_permission(req2)

    assert len(manager.requests) == 2

    manager.clear_session_requests("sess1")

    assert len(manager.requests) == 1
    assert "req2" in manager.requests
    assert "req1" not in manager.requests


def test_permission_manager_add_custom_policy():
    """Test adding a custom policy."""
    manager = PermissionManager()

    custom_policy = PermissionPolicy(
        name="allow-test-commands",
        action=PermissionAction.EXECUTE_COMMAND,
        pattern=r"pytest|nose|unittest",
        decision=PermissionDecision.APPROVED,
        max_risk=RiskLevel.LOW,
    )

    manager.add_policy(custom_policy)

    request = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.EXECUTE_COMMAND,
        description="Run tests",
        details={"command": "pytest tests/"},
        risk_level=RiskLevel.LOW,
    )

    result = manager.request_permission(request)

    assert result.decision == PermissionDecision.APPROVED
    assert "allow-test-commands" in result.decided_by


def test_safe_package_install_approved():
    """Test that safe package installs are auto-approved."""
    manager = PermissionManager()

    request = PermissionRequest(
        request_id="req1",
        session_id="sess1",
        action=PermissionAction.INSTALL_PACKAGE,
        description="Install pytest",
        details={"command": "pip install pytest"},
        risk_level=RiskLevel.LOW,
    )

    result = manager.request_permission(request)

    assert result.decision == PermissionDecision.APPROVED
