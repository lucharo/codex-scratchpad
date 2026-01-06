/**
 * Line Churn VS Code Extension
 * Visualize how frequently each line has been modified in git history.
 */

import * as vscode from 'vscode';
import { analyzeFile, ChurnResult } from './analyzer';

// Simple cache: filepath -> { data, timestamp }
const cache = new Map<string, { data: ChurnResult; time: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// Decoration types for each intensity bucket
let decorations: vscode.TextEditorDecorationType[] = [];

export function activate(context: vscode.ExtensionContext): void {
    // Commands
    context.subscriptions.push(
        vscode.commands.registerCommand('lineChurn.toggle', async () => {
            const config = vscode.workspace.getConfiguration('lineChurn');
            const enabled = config.get<boolean>('enabled', true);
            await config.update('enabled', !enabled, true);
        }),
        vscode.commands.registerCommand('lineChurn.refresh', () => {
            cache.clear();
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                void updateEditor(editor);
            }
        })
    );

    // Events
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(editor => {
            if (editor) {
                void updateEditor(editor);
            }
        }),
        vscode.workspace.onDidSaveTextDocument(doc => {
            cache.delete(doc.uri.fsPath);
            const editor = vscode.window.activeTextEditor;
            if (editor?.document === doc) {
                void updateEditor(editor);
            }
        }),
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('lineChurn')) {
                cache.clear();
                const editor = vscode.window.activeTextEditor;
                if (editor) {
                    void updateEditor(editor);
                }
            }
        })
    );

    // Initial update
    if (vscode.window.activeTextEditor) {
        void updateEditor(vscode.window.activeTextEditor);
    }
}

async function updateEditor(editor: vscode.TextEditor): Promise<void> {
    const config = vscode.workspace.getConfiguration('lineChurn');
    if (!config.get<boolean>('enabled', true)) {
        clearDecorations(editor);
        return;
    }

    if (editor.document.uri.scheme !== 'file') {
        clearDecorations(editor);
        return;
    }

    const filePath = editor.document.uri.fsPath;

    // Check cache
    const cached = cache.get(filePath);
    if (cached && Date.now() - cached.time < CACHE_TTL) {
        applyDecorations(editor, cached.data, config);
        return;
    }

    // Analyze
    const result = await analyzeFile(filePath);
    if (!result) {
        clearDecorations(editor);
        return;
    }

    cache.set(filePath, { data: result, time: Date.now() });
    applyDecorations(editor, result, config);
}

function applyDecorations(
    editor: vscode.TextEditor,
    data: ChurnResult,
    config: vscode.WorkspaceConfiguration
): void {
    clearDecorations(editor);

    const maxOpacity = config.get<number>('maxOpacity', 0.3);
    const scheme = config.get<string>('colorScheme', 'heat');
    const buckets = 10;

    // Group lines by intensity bucket
    const groups: number[][] = Array.from({ length: buckets }, () => []);
    for (const line of data.lines) {
        if (line.count === 0) {
            continue;
        }
        const bucket = Math.min(Math.floor(line.normalized * buckets), buckets - 1);
        groups[bucket].push(line.line);
    }

    // Create decoration for each bucket
    for (let i = 0; i < buckets; i++) {
        if (groups[i].length === 0) {
            continue;
        }

        const intensity = i / (buckets - 1);
        const color = getColor(intensity, scheme, maxOpacity);

        const type = vscode.window.createTextEditorDecorationType({
            backgroundColor: color,
            isWholeLine: true,
        });
        decorations.push(type);

        const ranges = groups[i].map(line =>
            new vscode.Range(line, 0, line, 0)
        );
        editor.setDecorations(type, ranges);
    }
}

function getColor(intensity: number, scheme: string, maxOpacity: number): string {
    const opacity = intensity * maxOpacity;

    if (scheme === 'blue') {
        const b = Math.round(100 + 155 * intensity);
        return `rgba(50, 100, ${b}, ${opacity})`;
    }
    if (scheme === 'mono') {
        const g = Math.round(200 - 150 * intensity);
        return `rgba(${g}, ${g}, ${g}, ${opacity})`;
    }
    // heat (default): green -> yellow -> red
    const r = Math.round(255 * Math.min(intensity * 2, 1));
    const g = Math.round(255 * Math.min((1 - intensity) * 2, 1));
    return `rgba(${r}, ${g}, 50, ${opacity})`;
}

function clearDecorations(editor: vscode.TextEditor): void {
    for (const d of decorations) {
        editor.setDecorations(d, []);
        d.dispose();
    }
    decorations = [];
}

export function deactivate(): void {
    cache.clear();
}
