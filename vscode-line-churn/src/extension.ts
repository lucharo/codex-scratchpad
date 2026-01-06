import * as vscode from 'vscode';
import { ChurnAnalyzer } from './churnAnalyzer';
import { DecorationProvider } from './decorationProvider';
import { CacheManager } from './cacheManager';

let analyzer: ChurnAnalyzer;
let decorationProvider: DecorationProvider;
let cache: CacheManager;
let statusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
    cache = new CacheManager();
    analyzer = new ChurnAnalyzer();
    decorationProvider = new DecorationProvider();

    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.command = 'lineChurn.toggle';
    context.subscriptions.push(statusBarItem);

    // Update decorations on active editor change
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(editor => {
            if (editor) {
                updateDecorations(editor);
            }
        })
    );

    // Update on document save (invalidate cache, file content changed)
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument(doc => {
            cache.invalidate(doc.uri.fsPath);
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document === doc) {
                updateDecorations(editor);
            }
        })
    );

    // Listen for configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('lineChurn')) {
                const editor = vscode.window.activeTextEditor;
                if (editor) {
                    // Clear cache to recompute with new settings
                    cache.clear();
                    updateDecorations(editor);
                }
            }
        })
    );

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('lineChurn.toggle', () => {
            const config = vscode.workspace.getConfiguration('lineChurn');
            const currentEnabled = config.get<boolean>('enabled', true);
            config.update('enabled', !currentEnabled, vscode.ConfigurationTarget.Global);

            if (currentEnabled) {
                // Turning off - clear decorations
                const editor = vscode.window.activeTextEditor;
                if (editor) {
                    decorationProvider.clear(editor);
                }
                statusBarItem.text = '$(eye-closed) Churn: Off';
            } else {
                statusBarItem.text = '$(eye) Churn: On';
            }
        }),
        vscode.commands.registerCommand('lineChurn.refresh', async () => {
            cache.clear();
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                statusBarItem.text = '$(sync~spin) Analyzing...';
                await updateDecorations(editor);
            }
        })
    );

    // Initial decoration for active editor
    if (vscode.window.activeTextEditor) {
        updateDecorations(vscode.window.activeTextEditor);
    }

    updateStatusBar();
}

async function updateDecorations(editor: vscode.TextEditor) {
    const config = vscode.workspace.getConfiguration('lineChurn');

    if (!config.get<boolean>('enabled', true)) {
        decorationProvider.clear(editor);
        updateStatusBar();
        return;
    }

    const filePath = editor.document.uri.fsPath;

    // Skip non-file URIs (e.g., untitled, output panels)
    if (editor.document.uri.scheme !== 'file') {
        decorationProvider.clear(editor);
        return;
    }

    // Check cache first
    let churnData = cache.get(filePath);

    if (!churnData) {
        statusBarItem.text = '$(sync~spin) Analyzing...';

        try {
            churnData = await analyzer.analyze(filePath);
            if (churnData) {
                cache.set(filePath, churnData);
            }
        } catch (error) {
            console.error('Line Churn analysis error:', error);
            statusBarItem.text = '$(warning) Churn: Error';
            return;
        }
    }

    if (churnData) {
        decorationProvider.apply(editor, churnData);
        statusBarItem.text = `$(flame) Churn: ${churnData.maxChurn} max`;
        statusBarItem.tooltip = `Line Churn Analysis\nMax churn: ${churnData.maxChurn}\nAnalyzed: ${churnData.lines.length} lines\nClick to toggle`;
    } else {
        decorationProvider.clear(editor);
        statusBarItem.text = '$(flame) Churn: N/A';
        statusBarItem.tooltip = 'No git history available for this file';
    }

    statusBarItem.show();
}

function updateStatusBar() {
    const config = vscode.workspace.getConfiguration('lineChurn');
    const enabled = config.get<boolean>('enabled', true);

    if (enabled) {
        statusBarItem.text = '$(flame) Churn';
    } else {
        statusBarItem.text = '$(eye-closed) Churn: Off';
    }
    statusBarItem.show();
}

export function deactivate() {
    decorationProvider.dispose();
    statusBarItem.dispose();
}
