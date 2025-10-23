"""Permission management system for controlling sub-agent actions.

This module implements a trust-based permission system where sub-agents must
request approval for actions, and the main agent can auto-approve, auto-deny,
or escalate to the user based on configurable policies.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PermissionAction(str, Enum):
    """Types of actions that require permission."""

    EXECUTE_COMMAND = "execute_command"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"
    NETWORK_REQUEST = "network_request"
    GIT_OPERATION = "git_operation"
    INSTALL_PACKAGE = "install_package"
    CUSTOM = "custom"


class PermissionDecision(str, Enum):
    """Decision on a permission request."""

    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"
    ESCALATED = "escalated"  # Escalated to user


class RiskLevel(str, Enum):
    """Risk level of an action."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PermissionRequest(BaseModel):
    """A permission request from a sub-agent."""

    request_id: str = Field(..., description="Unique request identifier")
    session_id: str = Field(..., description="Session making the request")
    action: PermissionAction = Field(..., description="Type of action")
    description: str = Field(..., description="Human-readable description")
    details: dict = Field(
        default_factory=dict,
        description="Action-specific details (e.g., command, file path)",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.MEDIUM, description="Assessed risk level"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Request timestamp"
    )
    decision: PermissionDecision = Field(
        default=PermissionDecision.PENDING, description="Current decision"
    )
    decided_at: Optional[datetime] = Field(
        None, description="Decision timestamp"
    )
    decided_by: Optional[str] = Field(
        None, description="Who made the decision (user/policy)"
    )
    denial_reason: Optional[str] = Field(None, description="Reason if denied")

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        }
    )


class PermissionPolicy(BaseModel):
    """Policy for automatically handling permission requests."""

    name: str = Field(..., description="Policy name")
    action: PermissionAction = Field(..., description="Action this applies to")
    pattern: Optional[str] = Field(
        None, description="Regex pattern to match against details"
    )
    decision: PermissionDecision = Field(
        ..., description="Auto-decision (approved/denied/escalated)"
    )
    max_risk: RiskLevel = Field(
        default=RiskLevel.CRITICAL,
        description="Maximum risk level this policy applies to",
    )
    reason: Optional[str] = Field(
        None, description="Reason for this policy decision"
    )

    def matches(self, request: PermissionRequest) -> bool:
        """Check if this policy matches a request.

        Args:
            request: Permission request to check

        Returns:
            True if policy applies to this request
        """
        # Check action type
        if self.action != request.action:
            return False

        # Check risk level
        risk_levels = [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]
        if risk_levels.index(request.risk_level) > risk_levels.index(
            self.max_risk
        ):
            return False

        # Check pattern if specified
        if self.pattern:
            # Convert details to string for pattern matching
            details_str = str(request.details)
            if not re.search(self.pattern, details_str, re.IGNORECASE):
                return False

        return True


class PermissionManager:
    """Manages permission requests and applies policies."""

    def __init__(self):
        """Initialize the permission manager."""
        self.requests: dict[str, PermissionRequest] = {}
        self.policies: list[PermissionPolicy] = []
        self._load_default_policies()

    def _load_default_policies(self) -> None:
        """Load default safe policies."""
        # Auto-approve safe read operations
        self.policies.append(
            PermissionPolicy(
                name="safe-reads",
                action=PermissionAction.READ_FILE,
                decision=PermissionDecision.APPROVED,
                max_risk=RiskLevel.LOW,
                reason="Read operations are generally safe",
            )
        )

        # Auto-approve safe git operations
        self.policies.append(
            PermissionPolicy(
                name="safe-git-status",
                action=PermissionAction.GIT_OPERATION,
                pattern=r"git\s+(status|log|diff|show|branch\s+--list)",
                decision=PermissionDecision.APPROVED,
                max_risk=RiskLevel.LOW,
                reason="Read-only git operations are safe",
            )
        )

        # Auto-deny destructive operations without explicit approval
        self.policies.append(
            PermissionPolicy(
                name="deny-destructive-git",
                action=PermissionAction.GIT_OPERATION,
                pattern=r"git\s+(reset\s+--hard|push\s+--force|clean\s+-.*[fd])",
                decision=PermissionDecision.DENIED,
                max_risk=RiskLevel.CRITICAL,
                reason="Destructive git operations require explicit approval",
            )
        )

        # Escalate file deletions to user
        self.policies.append(
            PermissionPolicy(
                name="escalate-deletions",
                action=PermissionAction.DELETE_FILE,
                decision=PermissionDecision.ESCALATED,
                max_risk=RiskLevel.HIGH,
                reason="File deletions should be reviewed",
            )
        )

        # Escalate network requests
        self.policies.append(
            PermissionPolicy(
                name="escalate-network",
                action=PermissionAction.NETWORK_REQUEST,
                decision=PermissionDecision.ESCALATED,
                max_risk=RiskLevel.MEDIUM,
                reason="Network requests should be reviewed",
            )
        )

        # Auto-approve low-risk package installs from known repos
        self.policies.append(
            PermissionPolicy(
                name="safe-package-install",
                action=PermissionAction.INSTALL_PACKAGE,
                pattern=r"(pip|uv|npm)\s+install\s+[a-zA-Z0-9\-_]+",
                decision=PermissionDecision.APPROVED,
                max_risk=RiskLevel.LOW,
                reason="Standard package installs from known repos",
            )
        )

    def add_policy(self, policy: PermissionPolicy) -> None:
        """Add a custom policy.

        Args:
            policy: Policy to add (inserted at the beginning for priority)
        """
        self.policies.insert(0, policy)

    def assess_risk(self, request: PermissionRequest) -> RiskLevel:
        """Assess the risk level of a request.

        Args:
            request: Permission request to assess

        Returns:
            Assessed risk level
        """
        # This is a simple heuristic - could be much more sophisticated
        if request.action == PermissionAction.DELETE_FILE:
            return RiskLevel.HIGH
        elif request.action == PermissionAction.EXECUTE_COMMAND:
            # Check for dangerous commands
            cmd = request.details.get("command", "")
            dangerous_patterns = [
                r"rm\s+-rf",
                r"dd\s+",
                r"mkfs",
                r":(){ :|:& };:",  # fork bomb
                r">\s*/dev/sd",
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, cmd, re.IGNORECASE):
                    return RiskLevel.CRITICAL
            return RiskLevel.MEDIUM
        elif request.action == PermissionAction.NETWORK_REQUEST:
            return RiskLevel.MEDIUM
        elif request.action == PermissionAction.READ_FILE:
            # Check if reading sensitive files
            path = request.details.get("path", "")
            sensitive_patterns = [
                r"/etc/shadow",
                r"/etc/passwd",
                r"\.ssh/",
                r"\.aws/credentials",
                r"\.env",
            ]
            for pattern in sensitive_patterns:
                if re.search(pattern, path, re.IGNORECASE):
                    return RiskLevel.HIGH
            return RiskLevel.LOW
        elif request.action == PermissionAction.WRITE_FILE:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.MEDIUM

    def request_permission(self, request: PermissionRequest) -> PermissionRequest:
        """Process a permission request.

        Args:
            request: Permission request

        Returns:
            Request with decision applied
        """
        # Assess risk if not already set
        if request.risk_level == RiskLevel.MEDIUM:
            request.risk_level = self.assess_risk(request)

        # Try to match against policies
        for policy in self.policies:
            if policy.matches(request):
                request.decision = policy.decision
                request.decided_at = datetime.now()
                request.decided_by = f"policy:{policy.name}"
                if policy.decision == PermissionDecision.DENIED:
                    request.denial_reason = policy.reason
                break

        # Store request
        self.requests[request.request_id] = request

        return request

    def make_decision(
        self,
        request_id: str,
        decision: PermissionDecision,
        decided_by: str = "user",
        reason: Optional[str] = None,
    ) -> PermissionRequest:
        """Make a decision on a pending request.

        Args:
            request_id: Request identifier
            decision: Decision to make
            decided_by: Who is making the decision
            reason: Optional reason for denial

        Returns:
            Updated request

        Raises:
            KeyError: If request not found
        """
        request = self.requests[request_id]
        request.decision = decision
        request.decided_at = datetime.now()
        request.decided_by = decided_by

        if decision == PermissionDecision.DENIED and reason:
            request.denial_reason = reason

        return request

    def get_pending_requests(
        self, session_id: Optional[str] = None
    ) -> list[PermissionRequest]:
        """Get all pending permission requests.

        Args:
            session_id: Optional filter by session

        Returns:
            List of pending requests
        """
        requests = [
            r
            for r in self.requests.values()
            if r.decision
            in (PermissionDecision.PENDING, PermissionDecision.ESCALATED)
        ]

        if session_id:
            requests = [r for r in requests if r.session_id == session_id]

        return sorted(requests, key=lambda r: r.created_at)

    def get_request(self, request_id: str) -> Optional[PermissionRequest]:
        """Get a request by ID.

        Args:
            request_id: Request identifier

        Returns:
            Request if found, None otherwise
        """
        return self.requests.get(request_id)

    def clear_session_requests(self, session_id: str) -> None:
        """Clear all requests for a session.

        Args:
            session_id: Session identifier
        """
        self.requests = {
            rid: req
            for rid, req in self.requests.items()
            if req.session_id != session_id
        }
