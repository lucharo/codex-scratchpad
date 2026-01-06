/**
 * Line Churn VS Code Extension - Main Entry Point
 *
 * Visualizes how frequently each line of code has been modified
 * throughout git history, like worn knobs on a control panel.
 */

import * as vscode from 'vscode';
import { analyzeFile } from './churnAnalyzer';
import { DecorationProvider } from './decorationProvider';
import { CacheManager } from './cacheManager';
import { getConfig, setConfig, affectsLineChurn } from './config';
import { ChurnData, AnalysisError, AnalysisErrorType } from './types';

/** Extension state */
let cache: CacheManager;
let decorationProvider: DecorationProvider;
let statusBarItem: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;

/** Pending analysis for debouncing */
let pendingAnalysis: NodeJS.Timeout | null = null;
const DEBOUNCE_MS = 300;

/**
 * Extension activation.
 */
export function activate(context: vscode.ExtensionContext): void {
    // Initialize components
    cache = new CacheManager();
    decorationProvider = new DecorationProvider();
    outputChannel = vscode.window.createOutputChannel('Line Churn');

    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.command = 'lineChurn.toggle';
    statusBarItem.tooltip = 'Click to toggle Line Churn visualization';

    // Register disposables
    context.subscriptions.push(
        cache,
        decorationProvider,
        statusBarItem,
        outputChannel
    );

    // Register event listeners
    registerEventListeners(context);

    // Register commands
    registerCommands(context);

    // Initial update
    updateStatusBar();
    if (vscode.window.activeTextEditor) {
        void scheduleUpdate(vscode.window.activeTextEditor);
    }

    log('Extension activated');
}

/**
 * Register event listeners for editor changes.
 */
function registerEventListeners(context: vscode.ExtensionContext): void {
    // Update on active editor change
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor) {
                void scheduleUpdate(editor);
            }
        })
    );

    // Invalidate cache and update on file save
    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument((doc) => {
            cache.invalidate(doc.uri.fsPath);
            const editor = vscode.window.activeTextEditor;
            if (editor?.document === doc) {
                void scheduleUpdate(editor);
            }
        })
    );

    // Update on configuration change
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration((event) => {
            if (affectsLineChurn(event)) {
                cache.clear(); // Config change may affect analysis
                updateStatusBar();
                const editor = vscode.window.activeTextEditor;
                if (editor) {
                    void scheduleUpdate(editor);
                }
            }
        })
    );

    // Clear decorations when editor closes
    context.subscriptions.push(
        vscode.workspace.onDidCloseTextDocument((doc) => {
            cache.invalidate(doc.uri.fsPath);
        })
    );
}

/**
 * Register extension commands.
 */
function registerCommands(context: vscode.ExtensionContext): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('lineChurn.toggle', async () => {
            const config = getConfig();
            await setConfig('enabled', !config.enabled);

            const editor = vscode.window.activeTextEditor;
            if (!config.enabled && editor) {
                // Was disabled, now enabled - update
                void scheduleUpdate(editor);
            } else if (config.enabled && editor) {
                // Was enabled, now disabled - clear
                decorationProvider.clear(editor);
            }
            updateStatusBar();
        }),

        vscode.commands.registerCommand('lineChurn.refresh', async () => {
            cache.clear();
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                await updateDecorations(editor, true);
            }
        }),

        vscode.commands.registerCommand('lineChurn.showLineHistory', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                return;
            }

            const filePath = editor.document.uri.fsPath;
            const churnData = cache.get(filePath);
            if (!churnData) {
                void vscode.window.showInformationMessage(
                    'No churn data available. Try refreshing first.'
                );
                return;
            }

            const line = editor.selection.active.line;
            const lineData = churnData.lines[line];
            if (!lineData) {
                return;
            }

            const message = lineData.churnCount > 0
                ? `Line ${line + 1}: Modified ${lineData.churnCount} times in git history`
                : `Line ${line + 1}: No modifications found in git history`;

            void vscode.window.showInformationMessage(message);
        })
    );
}

/**
 * Schedule a debounced update for the editor.
 */
function scheduleUpdate(editor: vscode.TextEditor): void {
    if (pendingAnalysis) {
        clearTimeout(pendingAnalysis);
    }

    pendingAnalysis = setTimeout(() => {
        pendingAnalysis = null;
        void updateDecorations(editor, false);
    }, DEBOUNCE_MS);
}

/**
 * Update decorations for an editor.
 */
async function updateDecorations(
    editor: vscode.TextEditor,
    forceRefresh: boolean
): Promise<void> {
    const config = getConfig();

    // Check if disabled
    if (!config.enabled) {
        decorationProvider.clear(editor);
        updateStatusBar();
        return;
    }

    // Skip non-file URIs
    if (editor.document.uri.scheme !== 'file') {
        decorationProvider.clear(editor);
        statusBarItem.text = '$(flame) Churn: N/A';
        statusBarItem.show();
        return;
    }

    const filePath = editor.document.uri.fsPath;

    // Check cache first (unless forced refresh)
    let churnData: ChurnData | null = null;
    if (!forceRefresh) {
        churnData = cache.get(filePath);
    }

    // Analyze if not cached
    if (!churnData) {
        statusBarItem.text = '$(sync~spin) Analyzing...';
        statusBarItem.show();

        const result = await analyzeFile(filePath);

        if (result.success) {
            churnData = result.value;
            cache.set(filePath, churnData);
        } else {
            handleAnalysisError(result.error);
            decorationProvider.clear(editor);
            return;
        }
    }

    // Apply decorations
    decorationProvider.apply(editor, churnData);

    // Update status bar
    statusBarItem.text = `$(flame) Churn: ${churnData.maxChurn} max`;
    statusBarItem.tooltip = [
        'Line Churn Analysis',
        `Max churn: ${churnData.maxChurn}`,
        `Lines: ${churnData.lines.length}`,
        `Analyzed: ${churnData.analyzedAt.toLocaleTimeString()}`,
        '',
        'Click to toggle',
    ].join('\n');
    statusBarItem.show();
}

/**
 * Handle analysis errors with appropriate user feedback.
 */
function handleAnalysisError(error: AnalysisError): void {
    log(`Analysis error: ${error.type} - ${error.message}`);

    switch (error.type) {
        case AnalysisErrorType.NotGitRepository:
            statusBarItem.text = '$(flame) Churn: No Git';
            statusBarItem.tooltip = 'File is not in a git repository';
            break;

        case AnalysisErrorType.FileNotTracked:
            statusBarItem.text = '$(flame) Churn: Untracked';
            statusBarItem.tooltip = 'File is not tracked by git';
            break;

        case AnalysisErrorType.Timeout:
            statusBarItem.text = '$(warning) Churn: Timeout';
            statusBarItem.tooltip = 'Analysis timed out. Try reducing commit limit in settings.';
            void vscode.window.showWarningMessage(
                'Line Churn analysis timed out. Try reducing lineChurn.commitLimit in settings.'
            );
            break;

        default:
            statusBarItem.text = '$(warning) Churn: Error';
            statusBarItem.tooltip = error.message;
    }

    statusBarItem.show();
}

/**
 * Update status bar based on current configuration.
 */
function updateStatusBar(): void {
    const config = getConfig();

    if (config.enabled) {
        statusBarItem.text = '$(flame) Churn';
        statusBarItem.tooltip = 'Line Churn visualization enabled. Click to disable.';
    } else {
        statusBarItem.text = '$(eye-closed) Churn: Off';
        statusBarItem.tooltip = 'Line Churn visualization disabled. Click to enable.';
    }

    statusBarItem.show();
}

/**
 * Log a message to the output channel.
 */
function log(message: string): void {
    const timestamp = new Date().toISOString();
    outputChannel.appendLine(`[${timestamp}] ${message}`);
}

/**
 * Extension deactivation.
 */
export function deactivate(): void {
    if (pendingAnalysis) {
        clearTimeout(pendingAnalysis);
    }
    log('Extension deactivated');
}
