- mono repo for open ai's codex agent.
- one folder per project/idea. all self contained
- prefer uv over pip/venv for project management
- stick to uv's syntax if possible (e.g. prefer uv add to uv pip install)
- initialise project with uv init
- prefer marimo over jupyter notebooks
- prefers polars over pandas

## Working with marimo

marimo notebooks are reactive Python notebooks that double as scripts and apps.

**Running:**
```bash
uvx marimo edit notebook.py   # Interactive editing
uvx marimo run notebook.py    # Run as app
uv run notebook.py            # Run as script (good integration test)
```

**Validation:**
```bash
uvx marimo check notebook.py  # Check for errors before committing
```

**Key patterns:**
- Each cell is a function decorated with `@app.cell`
- Variables returned from cells are available to other cells
- Use `_` prefix for private variables (e.g., `_conn`, `_n`) to avoid conflicts
- Imports go in cells, returned in tuple: `return (mo, pl, alt)`
- Use `mo.stop(condition)` to halt cell execution conditionally
- UI elements: `mo.ui.dropdown`, `mo.ui.table`, `mo.ui.run_button`, etc.
- No `mo.ui.time` - use dropdown with time strings instead

**Structure for larger apps:**
- Keep notebook lean (~300-400 lines) - UI and plots only
- Extract logic to modules: `api.py`, `data.py`, `stats.py`
- Import from modules inside cells: `from stats import get_route_stats`

**Inline dependencies (PEP 723):**
```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "polars", "altair"]
# ///
```
- python viz libraries in order of preference: altair, plotnine, seaborn or matplotlib

if doing front end/fulls stack work:
- prefer next.js and svelte (5 if possible) over other similar tools

- browse the internet and read the docs whener you're lost, best of luck! 
