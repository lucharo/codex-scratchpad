# Hierarchical Session Management - New Features

This document describes the new hierarchical and organizational features added to c2c.

## Overview

c2c now supports three advanced use cases through hierarchical session management:

1. **CEO Mode** - Parallel strategy comparison
2. **Orthogonal Tasks** - Independent parallel work streams
3. **Self-Organizing Agents** - Dynamic organizational structures

## New Data Model Features

### Session Tags

Sessions can now be tagged for easy organization and filtering:

```python
create_session(
    task="Implement authentication",
    tags={
        "feature": "AddAuth",
        "role": "Backend",
        "strategy": "OptionA",
        "phase": "MVP"
    }
)
```

**Use cases:**
- Group related sessions
- Filter sessions by criteria
- Track parallel work streams
- Organize by role, feature, phase, etc.

### Session Metadata

Additional custom metadata for any use case:

```python
create_session(
    task="Build API",
    metadata={
        "priority": "high",
        "team": "engineering",
        "sprint": "sprint-23"
    }
)
```

### Parent-Child Relationships

Sessions can now form hierarchical trees:

```python
# Create parent
parent = create_session(task="Lead engineering")

# Create children
child1 = create_session(
    task="Build backend",
    parent_session_id=parent.session_id
)

child2 = create_session(
    task="Build frontend",
    parent_session_id=parent.session_id
)
```

**Automatic tracking:**
- `parent_session_id` - Links to parent
- `child_session_ids` - List of all children
- `depth` - How deep in the hierarchy (0 = root)

## New MCP Tools

### 1. Enhanced create_session

Now supports:
- `parent_session_id` - For hierarchy
- `tags` - Dictionary of tags
- `metadata` - Additional custom data

```python
create_session(
    task="Implement feature",
    parent_session_id="c2c-abc123",
    tags={"feature": "Auth", "role": "Backend"},
    metadata={"priority": "high"}
)
```

### 2. get_session_tree

Get the complete hierarchical tree for a session:

```python
tree = get_session_tree(session_id="c2c-root")

# Returns:
{
    "session_id": "c2c-root",
    "task": "Build product",
    "status": "running",
    "depth": 0,
    "tags": {"project": "SaaS"},
    "children": [
        {
            "session_id": "c2c-backend",
            "task": "Build API",
            "depth": 1,
            "children": [...]
        },
        {
            "session_id": "c2c-frontend",
            "task": "Build UI",
            "depth": 1,
            "children": []
        }
    ]
}
```

**Visual output:**
```
○ c2c-root (depth=0) {project: 'SaaS'}
  Task: Build product
  Branch: claude/c2c-root-build-product
  ● c2c-backend (depth=1) {role: 'Backend'}
    Task: Build API
    Branch: claude/c2c-backend-build-api
  ● c2c-frontend (depth=1) {role: 'Frontend'}
    Task: Build UI
    Branch: claude/c2c-frontend-build-ui
```

### 3. get_sessions_by_tags

Find all sessions matching specific tags:

```python
# Find all backend sessions
backend_sessions = get_sessions_by_tags(tags={"role": "Backend"})

# Find all Option A sessions
optionA = get_sessions_by_tags(tags={"strategy": "OptionA"})

# Find all MVP phase engineering work
mvp_eng = get_sessions_by_tags(tags={"phase": "MVP", "team": "engineering"})
```

## New Helper Methods

### SessionManager API

```python
# Get session tree as dictionary
tree = session_manager.get_session_tree(session_id)

# Get all ancestors (parent, grandparent, ...)
ancestors = session_manager.get_ancestors(session_id)

# Get all descendants (children, grandchildren, ...)
descendants = session_manager.get_descendants(session_id)

# Get all root sessions (no parent)
roots = session_manager.get_root_sessions()

# Find sessions by tags
sessions = session_manager.get_sessions_by_tags({"feature": "Auth"})
```

## Use Case Examples

### CEO Mode: Multiple Strategies

```python
# Create 3 strategy options in parallel
for option in ["A", "B", "C"]:
    reviewer = create_session(
        task=f"Review approach {option}",
        tags={"feature": "AddAuth", "strategy": f"Option{option}", "role": "Reviewer"}
    )

    implementer = create_session(
        task=f"Implement approach {option}",
        tags={"feature": "AddAuth", "strategy": f"Option{option}", "role": "Implementer"},
        parent_session_id=reviewer.session_id
    )

# Compare results
for option in ["A", "B", "C"]:
    sessions = get_sessions_by_tags(tags={"strategy": f"Option{option}"})
    # Review outputs, pick winner

# Cleanup losers
losing_sessions = get_sessions_by_tags(tags={"strategy": "OptionA"})
for session in losing_sessions:
    cleanup_session(session_id=session.session_id, remove_branch=True)
```

### Orthogonal Tasks

```python
# Create parallel independent tasks
auth = create_session(
    task="Add authentication",
    tags={"project": "WebApp", "group": "A", "feature": "auth"}
)

email = create_session(
    task="Add email service",
    tags={"project": "WebApp", "group": "A", "feature": "email"}
)

# Both run in parallel, merge in any order
```

### Self-Organizing Agents

```python
# Root creates team leads
tech_lead = create_session(
    task="Lead engineering",
    tags={"project": "Startup", "role": "TechLead"}
)

# Tech lead spawns backend
backend = create_session(
    task="Build API",
    tags={"project": "Startup", "role": "Backend"},
    parent_session_id=tech_lead.session_id
)

# Backend spawns database specialist
db_specialist = create_session(
    task="Design database schema",
    tags={"project": "Startup", "role": "DatabaseSpecialist"},
    parent_session_id=backend.session_id
)

# View the org chart
tree = get_session_tree(session_id=tech_lead.session_id)
```

## Session Status Icons

When viewing sessions, status is indicated with icons:

- `○` CREATED - Session created but not started
- `●` RUNNING - Currently executing
- `✓` COMPLETED - Successfully finished
- `✗` FAILED - Completed with errors
- `⊗` TERMINATED - Manually stopped

## Branch Naming

Branches are auto-generated with pattern:
```
claude/<session-id>-<sanitized-task>
```

Examples:
- `claude/c2c-abc123-implement-authentication`
- `claude/c2c-def456-build-api-endpoint`

## Query Patterns

### Find all work for a feature
```python
get_sessions_by_tags(tags={"feature": "Authentication"})
```

### Find all reviewers
```python
get_sessions_by_tags(tags={"role": "Reviewer"})
```

### Find all failed sessions
```python
all_sessions = session_manager.list_sessions()
failed = [s for s in all_sessions if s.status == "failed"]
```

### Find all children of a session
```python
session = session_manager.get_session(session_id)
children = [session_manager.get_session(cid) for cid in session.child_session_ids]
```

## Migration from Previous Version

Old code without hierarchy still works:

```python
# Old code - still works
create_session(task="Do something")
```

New hierarchy features are **opt-in**:

```python
# New code - with hierarchy
create_session(
    task="Do something",
    parent_session_id="...",  # Optional
    tags={"feature": "..."}    # Optional
)
```

## Best Practices

1. **Use consistent tags** - Define your tagging schema upfront
2. **Limit depth** - Keep hierarchies 2-3 levels max for most use cases
3. **Clean up** - Remove failed/rejected sessions and their branches
4. **Tag for queries** - Tag sessions in a way that makes filtering easy
5. **Document relationships** - Use `parent_session_id` to track dependencies

## System Prompts

Three system prompts are provided for different use cases:

1. `docs/CEO_MODE_PROMPT.md` - Parallel strategy comparison
2. `docs/ORTHOGONAL_TASKS_PROMPT.md` - Independent parallel work
3. `docs/SELF_ORGANIZING_AGENTS_PROMPT.md` - Dynamic organization

These can be used as system prompts or Claude Skills to guide agent behavior.

## Testing

All 68 existing tests pass. New features are backwards compatible.

## Next Steps

For permission hierarchy (so sub-agents have limited permissions):
- See `docs/CODE_REVIEW.md` section on permission hierarchy
- This is a more complex feature for future implementation
- Current permissions are flat (all agents have same level)
