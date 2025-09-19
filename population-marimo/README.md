# Population Marimo App

This project contains a small [marimo](https://marimo.io) notebook that explores simple population scenarios using World Bank indicators.

The repository ships with a tiny cached dataset under `data/`. When network access is available, the notebook will attempt to refresh the cache from the World Bank API; otherwise it will fall back to these CSVs.

Run the app with:

```bash
marimo run app.py
```
