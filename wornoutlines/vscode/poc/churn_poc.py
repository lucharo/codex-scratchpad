#!/usr/bin/env python3
"""
Line Churn Visualizer POC
Shows how often each line in a file has been modified throughout git history.
The "worn knob" effect for code.

This is the reference implementation that proves the algorithm works.
The VS Code extension translates this logic to TypeScript.
"""

import subprocess
import os
import tempfile
import shutil
from collections import defaultdict


def run(cmd, cwd=None):
    """Run a shell command and return the result"""
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)


def create_test_repo():
    """Create a test repo with realistic commit history showing varied churn"""
    repo_dir = tempfile.mkdtemp(prefix="churn_test_")

    run("git init", cwd=repo_dir)
    run("git config user.email 'test@test.com'", cwd=repo_dir)
    run("git config user.name 'Test'", cwd=repo_dir)

    filepath = os.path.join(repo_dir, "example.py")

    # File versions showing realistic churn patterns:
    # - MAGIC_NUMBER never changes (stable structural code)
    # - TAX_RATE changes frequently (config value that gets tweaked)
    # - calculate_total body changes a few times (bug fixes)
    # - return statement changes multiple times (fought over)
    # - validate_item never changes (stable helper)
    versions = [
        # v1: Initial
        '''MAGIC_NUMBER = 42
TAX_RATE = 0.1

def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total

def validate_item(item):
    """Validate an item has required fields."""
    return hasattr(item, 'price')
''',
        # v2: Bug fix - multiply by quantity (changes line 7)
        '''MAGIC_NUMBER = 42
TAX_RATE = 0.1

def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total

def validate_item(item):
    """Validate an item has required fields."""
    return hasattr(item, 'price')
''',
        # v3: Change tax rate constant (changes line 2)
        '''MAGIC_NUMBER = 42
TAX_RATE = 0.15

def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total

def validate_item(item):
    """Validate an item has required fields."""
    return hasattr(item, 'price')
''',
        # v4: Change tax rate again (changes line 2)
        '''MAGIC_NUMBER = 42
TAX_RATE = 0.08

def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total

def validate_item(item):
    """Validate an item has required fields."""
    return hasattr(item, 'price')
''',
        # v5: Add rounding to return (changes line 8)
        '''MAGIC_NUMBER = 42
TAX_RATE = 0.08

def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return round(total, 2)

def validate_item(item):
    """Validate an item has required fields."""
    return hasattr(item, 'price')
''',
        # v6: Tweak return again (changes line 8)
        '''MAGIC_NUMBER = 42
TAX_RATE = 0.08

def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return round(total, 4)

def validate_item(item):
    """Validate an item has required fields."""
    return hasattr(item, 'price')
''',
        # v7: Back to 2 decimals (changes line 8 again!)
        '''MAGIC_NUMBER = 42
TAX_RATE = 0.08

def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return round(total, 2)

def validate_item(item):
    """Validate an item has required fields."""
    return hasattr(item, 'price')
''',
        # v8: Yet another tax rate change (line 2 again)
        '''MAGIC_NUMBER = 42
TAX_RATE = 0.0725

def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return round(total, 2)

def validate_item(item):
    """Validate an item has required fields."""
    return hasattr(item, 'price')
''',
    ]

    messages = [
        'Initial commit',
        'Fix: multiply by quantity',
        'Increase tax rate',
        'Lower tax rate',
        'Add rounding',
        'Use 4 decimal places',
        'Revert to 2 decimals',
        'Update to CA tax rate',
    ]

    for v, msg in zip(versions, messages):
        with open(filepath, 'w') as f:
            f.write(v)
        run(f"git add . && git commit -m '{msg}'", cwd=repo_dir)

    return repo_dir, filepath


def get_line_churn(filepath, repo_dir):
    """
    For each line in the current file, count how many times
    that exact content appears in the git diff history.

    KEY INSIGHT: We track by CONTENT, not line number.
    This means when code moves (refactoring, adding functions above),
    the churn history follows the content, not the position.
    """
    # Get full diff history with patches
    result = run(f"git log -p --follow -- {os.path.basename(filepath)}", cwd=repo_dir)
    diff_history = result.stdout

    # Read current file
    with open(filepath) as f:
        current_lines = f.readlines()

    churn = []
    for i, line in enumerate(current_lines):
        content = line.rstrip()
        if not content.strip():  # Skip empty lines
            churn.append((i + 1, content, 0))
            continue

        # Count appearances in diffs (added or removed)
        # Lines in diffs are prefixed with + or -
        add_count = diff_history.count(f"\n+{content}\n")
        remove_count = diff_history.count(f"\n-{content}\n")

        # Also check for lines at end of diff blocks (no trailing newline)
        add_count += diff_history.count(f"\n+{content}")
        remove_count += diff_history.count(f"\n-{content}")

        # Total churn = times added + times removed
        # A line that changes shows up as removed then added
        total = add_count + remove_count
        churn.append((i + 1, content, total))

    return churn


def visualize_churn(churn, max_bar_width=30):
    """ASCII visualization of line churn"""
    max_churn = max(c[2] for c in churn) or 1

    print("\n" + "=" * 80)
    print("LINE CHURN VISUALIZATION - 'Worn Knob' Effect")
    print("=" * 80)
    print(f"{'Line':<5} {'Churn':<6} {'Bar':<{max_bar_width+2}} Code")
    print("-" * 80)

    for line_num, content, count in churn:
        bar_len = int((count / max_churn) * max_bar_width) if max_churn > 0 else 0
        bar = "█" * bar_len + "░" * (max_bar_width - bar_len)

        # Truncate long lines
        display_content = content[:35] + "..." if len(content) > 38 else content

        # Color hint based on churn (using ANSI)
        if count >= max_churn * 0.7:
            color = "\033[91m"  # Red - hot
        elif count >= max_churn * 0.3:
            color = "\033[93m"  # Yellow - warm
        else:
            color = "\033[92m"  # Green - stable
        reset = "\033[0m"

        print(f"{line_num:<5} {color}{count:<6}{reset} [{bar}] {display_content}")

    print("-" * 80)
    print("\nInterpretation:")
    print("  🔴 High churn = frequently modified (the 'worn knob')")
    print("  🟢 Low churn  = stable code")
    print()


def generate_html_report(churn, filepath, output_path):
    """Generate an HTML visualization"""
    max_churn = max(c[2] for c in churn) or 1

    lines_html = []
    for line_num, content, count in churn:
        intensity = count / max_churn if max_churn > 0 else 0
        # Color from green (stable) to red (churny)
        r = int(255 * intensity)
        g = int(255 * (1 - intensity))
        bg_color = f"rgba({r}, {g}, 50, 0.3)"
        bar_width = int(intensity * 100)

        escaped_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if not escaped_content.strip():
            escaped_content = "&nbsp;"

        lines_html.append(f'''
        <div class="line" style="background: linear-gradient(to right, {bg_color} {bar_width}%, transparent {bar_width}%);">
            <span class="line-num">{line_num}</span>
            <span class="churn-count" title="Churn count">{count}</span>
            <code class="content">{escaped_content}</code>
        </div>''')

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Line Churn: {os.path.basename(filepath)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            margin: 0;
        }}
        h1 {{
            color: #569cd6;
            font-size: 1.4em;
        }}
        .subtitle {{
            color: #808080;
            margin-bottom: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .line {{
            display: flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 2px;
            margin: 1px 0;
        }}
        .line:hover {{
            background-color: rgba(255,255,255,0.1) !important;
        }}
        .line-num {{
            color: #858585;
            min-width: 30px;
            text-align: right;
            margin-right: 15px;
            font-size: 12px;
        }}
        .churn-count {{
            background: #333;
            color: #9cdcfe;
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 11px;
            min-width: 20px;
            text-align: center;
            margin-right: 15px;
        }}
        .content {{
            white-space: pre;
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 14px;
        }}
        .legend {{
            margin-top: 30px;
            padding: 15px;
            background: #252526;
            border-radius: 5px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 12px;
            margin-right: 5px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Line Churn Analysis</h1>
        <p class="subtitle">File: {os.path.basename(filepath)} | The "worn knob" effect - lines that change frequently</p>

        <div class="code-view">
            {"".join(lines_html)}
        </div>

        <div class="legend">
            <span class="legend-item">
                <span class="legend-color" style="background: rgba(50, 255, 50, 0.3);"></span>
                Stable (low churn)
            </span>
            <span class="legend-item">
                <span class="legend-color" style="background: rgba(255, 255, 50, 0.3);"></span>
                Moderate churn
            </span>
            <span class="legend-item">
                <span class="legend-color" style="background: rgba(255, 50, 50, 0.3);"></span>
                High churn (frequently modified)
            </span>
        </div>
    </div>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


if __name__ == "__main__":
    print("Creating test repository with commit history...")
    repo_dir, filepath = create_test_repo()

    print(f"Repository created at: {repo_dir}")
    print(f"Analyzing: {filepath}")

    print("\nCurrent file content:")
    print("-" * 40)
    with open(filepath) as f:
        print(f.read())
    print("-" * 40)

    churn = get_line_churn(filepath, repo_dir)
    visualize_churn(churn)

    # Generate HTML report
    html_path = os.path.join(repo_dir, "churn_report.html")
    generate_html_report(churn, filepath, html_path)
    print(f"HTML report generated: {html_path}")

    print(f"\nTest repo kept at: {repo_dir}")
    print("You can inspect the git history with: git log --oneline")
