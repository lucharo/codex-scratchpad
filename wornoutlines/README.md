# Worn Out Lines

Visualize which lines of code change frequently. Lines that get modified often appear highlighted - the "hotter" the color, the more churn.

## Why

<!-- What problem does this solve? Why did you build it? -->

## How It Works

Content-addressed tracking: churn follows the code itself, not line numbers. When you move a function, its history moves with it.

- Analyzes git history (VS Code) or GitHub API (Chrome)
- Counts how often each line appears in diffs
- Normalizes and visualizes with color intensity

## VS Code Extension

<!-- Installation instructions, marketplace link if published -->

### Commands

- `Worn Out Lines: Toggle` - Enable/disable visualization
- `Worn Out Lines: Refresh` - Clear cache and re-analyze

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `wornOutLines.enabled` | `true` | Enable churn visualization |
| `wornOutLines.maxOpacity` | `0.3` | Maximum highlight opacity (0.1-0.8) |
| `wornOutLines.colorScheme` | `heat` | Color scheme: `heat`, `blue`, or `mono` |

## Chrome Extension

Works on public GitHub repositories. Rate limited to 60 requests/hour (GitHub API without auth).

<!-- Installation instructions, Chrome Web Store link if published -->

### Limitations

- Public repositories only
- 60 API requests per hour
- Analyzes last 30 commits per file

## Screenshots

<!-- Add screenshots here -->

## Development

```bash
# VS Code extension
cd vscode
npm install
npm run compile
# Press F5 in VS Code to launch Extension Development Host

# Chrome extension
cd chrome
# Load unpacked extension from chrome://extensions
```

## Known Issues

<!-- Any quirks or limitations worth noting -->

## License

<!-- MIT, Apache 2.0, etc. -->
